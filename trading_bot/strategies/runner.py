"""
Backtest runner — currency-accurate replay της MA-crossover στρατηγικής πάνω σε
ιστορικά OHLCV, με ρεαλιστικά fees και position sizing 1% risk.

Σε αντίθεση με το `backtest.py` (που επιστρέφει R-multiples για τον optimizer),
εδώ μετατρέπουμε κάθε trade σε **νόμισμα ($)**: gross PnL, fee, net PnL, ώστε να
απαντάμε "πόσα trades / πόσο profit / profit after fees".

Είσοδος δεδομένων:
  --csv PATH   : CSV με στήλες open,high,low,close,volume (ή timestamp+OHLCV)
  --bars N     : αν δεν δοθεί CSV, παράγει N synthetic 1h bars (demo)

Παράμετροι στρατηγικής/ρίσκου διαβάζονται από το config.json (live params).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .backtest import _atr, _sma, synthetic_ohlcv


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    side: str
    entry: float
    exit: float
    qty: float
    gross_pnl: float
    fee: float

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fee


@dataclass
class BacktestReport:
    trades: list[Trade] = field(default_factory=list)
    start_equity: float = 10000.0
    fee_rate: float = 0.0004  # ανά side (Binance futures taker ~0.04%)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def gross_pnl(self) -> float:
        return sum(t.gross_pnl for t in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.fee for t in self.trades)

    @property
    def net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def end_equity(self) -> float:
        return self.start_equity + self.net_pnl

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl > 0)

    def summary(self) -> dict:
        net_series = np.array([t.net_pnl for t in self.trades], dtype=float)
        equity = self.start_equity + np.cumsum(net_series) if net_series.size \
            else np.array([self.start_equity])
        rets = net_series / self.start_equity if net_series.size else np.array([])
        return {
            "n_trades": self.n_trades,
            "wins": self.wins,
            "losses": self.n_trades - self.wins,
            "win_rate": round(self.wins / self.n_trades, 4) if self.n_trades else 0.0,
            "gross_pnl": round(self.gross_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(100 * self.net_pnl / self.start_equity, 3),
            "profit_factor": round(metrics.profit_factor(net_series), 3)
                if net_series.size else 0.0,
            # Sharpe θέλει εύλογο δείγμα· <3 trades -> αναξιόπιστο, δείξε 0.0
            "sharpe": round(metrics.sharpe_ratio(rets), 3) if rets.size >= 3 else 0.0,
            "max_drawdown_pct": round(100 * metrics.max_drawdown(equity), 3),
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
        }


def simulate(ohlcv: np.ndarray, sig: np.ndarray, atr_period: int,
            atr_sl_mult: float, rr_ratio: float, equity: float = 10000.0,
            risk_per_trade: float = 0.01, fee_rate: float = 0.0004,
            max_leverage: float = 1.0, max_hold: int = 0,
            spread_bps: float = 0.0, min_fee: float = 0.0,
            trail_atr: float = 0.0, adx_period: int = 14,
            adx_thr: float = 0.0) -> BacktestReport:
    """
    Κοινό execution engine για όλες τις στρατηγικές. Δέχεται έτοιμο per-bar σήμα
    `sig` (+1 long / -1 short / 0). **Ένα position τη φορά**, με:
      - ATR stop + RR target (bracket exit)· ή, αν `trail_atr`>0, **trailing stop**
        (ride-the-trend: μένει μέσα όσο πάει υπέρ σου, βγαίνει όταν γυρίσει
        trail_atr×ATR από το ακραίο — χωρίς σταθερό take-profit)
      - **ADAPTIVE exit** (αν `trail_atr`>0 ΚΑΙ `adx_thr`>0): ανά trade, αν στην
        είσοδο ADX>=adx_thr (τάση) -> trailing (ride)· αλλιώς (πλάγια) -> fixed RR
        target (γρήγορο κέρδος, υψηλό win rate). Ο bot «διαλέγει μόνος του».
      - leverage cap (notional <= max_leverage*equity) -> έλεγχος fees
      - optional time-exit μετά από `max_hold` bars (χρήσιμο σε mean-reversion)
      - 1% risk position sizing στο τρέχον equity (compounding)
      - ρεαλιστικά frictions: `spread_bps` (slippage ανά side) + `min_fee` (ελάχιστη
        προμήθεια ανά side — τιμωρεί μικρά trades/μικρά κεφάλαια)
    """
    report = BacktestReport(start_equity=equity, fee_rate=fee_rate)
    if ohlcv.ndim != 2 or ohlcv.shape[0] < 3 or sig.size != ohlcv.shape[0]:
        return report

    high, low, close = ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    atr = _atr(high, low, close, atr_period)
    n = close.size
    # ADX (ισχύς τάσης) μόνο αν χρειάζεται adaptive exit
    adx_arr = None
    if trail_atr > 0.0 and adx_thr > 0.0:
        from .signals import adx as _adx
        adx_arr = _adx(ohlcv, adx_period)

    cur_equity = equity
    pos: dict | None = None
    for i in range(1, n):
        if cur_equity <= 0:           # bankruptcy stop
            break
        # --- έλεγχος εξόδου τρέχουσας θέσης ---
        if pos is not None and i > pos["entry_idx"]:
            exit_price = None
            if pos["trail"] and not np.isnan(atr[i]):
                # TRAILING stop: ακολουθεί την ευνοϊκή ακραία τιμή (ride the trend —
                # «μένει μέσα όσο πάει υπέρ σου, βγαίνει όταν γυρίσει trail_atr×ATR»)
                if pos["side"] == "buy":
                    pos["stop"] = max(pos["stop"], pos["hw"] - trail_atr * atr[i])
                    if low[i] <= pos["stop"]:
                        exit_price = pos["stop"]
                    pos["hw"] = max(pos["hw"], high[i])
                else:
                    pos["stop"] = min(pos["stop"], pos["lw"] + trail_atr * atr[i])
                    if high[i] >= pos["stop"]:
                        exit_price = pos["stop"]
                    pos["lw"] = min(pos["lw"], low[i])
            elif pos["side"] == "buy":
                if low[i] <= pos["stop"]:
                    exit_price = pos["stop"]
                elif high[i] >= pos["target"]:
                    exit_price = pos["target"]
            else:  # sell / short
                if high[i] >= pos["stop"]:
                    exit_price = pos["stop"]
                elif low[i] <= pos["target"]:
                    exit_price = pos["target"]
            # time-exit: κλείσε στο close αν κρατήθηκε αρκετά
            if exit_price is None and max_hold > 0 and \
                    (i - pos["entry_idx"]) >= max_hold:
                exit_price = float(close[i])
            if exit_price is not None:
                report.trades.append(_close(pos, exit_price, i, fee_rate,
                                            spread_bps, min_fee))
                cur_equity += report.trades[-1].net_pnl
                pos = None

        # --- άνοιγμα νέας θέσης αν είμαστε flat ---
        if pos is None and sig[i] != 0 and not np.isnan(atr[i]) and atr[i] > 0:
            entry = float(close[i])
            stop_dist = atr_sl_mult * float(atr[i])
            if stop_dist <= 0:
                continue
            qty = (cur_equity * risk_per_trade) / stop_dist  # 1% risk sizing
            # leverage cap: notional <= max_leverage * equity (αποφυγή fee-bleed)
            qty = min(qty, (max_leverage * cur_equity) / entry)
            if sig[i] == 1:                                   # LONG
                stop, target = entry - stop_dist, entry + rr_ratio * stop_dist
                side = "buy"
            else:                                             # SHORT
                stop, target = entry + stop_dist, entry - rr_ratio * stop_dist
                side = "sell"
            # exit mode: σταθερό trailing, ή ADAPTIVE (τάση->trail, πλάγια->bracket)
            use_trail = trail_atr > 0.0
            if adx_arr is not None:
                a = adx_arr[i]
                use_trail = bool(not np.isnan(a) and a >= adx_thr)
            pos = {"side": side, "entry": entry, "stop": stop, "target": target,
                   "qty": qty, "entry_idx": i, "hw": entry, "lw": entry,
                   "trail": use_trail}

    if pos is not None:  # mark-to-market τυχόν ανοιχτής θέσης
        report.trades.append(_close(pos, float(close[-1]), n - 1, fee_rate,
                                    spread_bps, min_fee))
    return report


def run(ohlcv: np.ndarray, fast_ma: int, slow_ma: int, atr_period: int,
       atr_sl_mult: float, rr_ratio: float, equity: float = 10000.0,
       risk_per_trade: float = 0.01, fee_rate: float = 0.0004,
       max_leverage: float = 1.0) -> BacktestReport:
    """Backward-compatible: bidirectional MA crossover πάνω στο κοινό simulate()."""
    from .signals import ma_crossover
    sig = ma_crossover(ohlcv, fast_ma, slow_ma)
    return simulate(ohlcv, sig, atr_period, atr_sl_mult, rr_ratio, equity,
                   risk_per_trade, fee_rate, max_leverage)


def _close(pos: dict, exit_price: float, exit_idx: int, fee_rate: float,
          spread_bps: float = 0.0, min_fee: float = 0.0) -> Trade:
    qty, entry = pos["qty"], pos["entry"]
    hs = spread_bps / 10000.0 / 2.0   # half-spread slippage ανά side
    if pos["side"] == "buy":           # αγοράζεις ψηλότερα, πουλάς χαμηλότερα
        eff_entry, eff_exit = entry * (1 + hs), exit_price * (1 - hs)
        gross = (eff_exit - eff_entry) * qty
    else:                              # short: πουλάς χαμηλότερα, καλύπτεις ψηλότερα
        eff_entry, eff_exit = entry * (1 - hs), exit_price * (1 + hs)
        gross = (eff_entry - eff_exit) * qty
    # προμήθεια ανά side με ελάχιστο (fixed broker cost τιμωρεί μικρά trades)
    fee = max(fee_rate * qty * entry, min_fee) + \
        max(fee_rate * qty * exit_price, min_fee)
    return Trade(pos["entry_idx"], exit_idx, pos["side"], entry, exit_price,
                qty, gross, fee)


def load_csv(path: str) -> np.ndarray:
    """Διαβάζει CSV -> (N,5) [open,high,low,close,volume]. Δέχεται προαιρετικό
    leading timestamp/date column· αγνοεί header αν υπάρχει."""
    rows: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            # βρες τις 5 αριθμητικές στήλες (παράκαμψε timestamp/date στην αρχή)
            nums = []
            for cell in r:
                try:
                    nums.append(float(cell))
                except ValueError:
                    continue
            if len(nums) >= 5:
                rows.append(nums[-5:])
    return np.array(rows, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Currency-accurate backtest runner")
    ap.add_argument("--csv", help="CSV με OHLCV (open,high,low,close,volume)")
    ap.add_argument("--live", action="store_true",
                   help="τράβα πραγματικά δεδομένα από public πηγή (Coinbase/CryptoCompare)")
    ap.add_argument("--symbol", default="BTC-USD", help="π.χ. BTC-USD (live)")
    ap.add_argument("--start", default="", help="YYYY-MM-DD (live)")
    ap.add_argument("--end", default="", help="YYYY-MM-DD (live)")
    ap.add_argument("--interval", default="1h", help="1m|5m|15m|1h|6h|1d (live)")
    ap.add_argument("--bars", type=int, default=72,
                   help="synthetic 1h bars αν δεν δοθεί CSV/live (default 72 = 3 ημέρες)")
    ap.add_argument("--seed", type=int, default=24, help="seed synthetic data")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--equity", type=float, default=None,
                   help="override start equity (αλλιώς από config)")
    ap.add_argument("--fee", type=float, default=None,
                   help="fee rate ανά side (default: config costs ή 0.04%%)")
    ap.add_argument("--spread-bps", type=float, default=None,
                   help="slippage ανά side σε bps (default: config costs ή 0)")
    ap.add_argument("--min-fee", type=float, default=None,
                   help="ελάχιστη προμήθεια ανά side — τιμωρεί μικρά κεφάλαια "
                        "(default: config costs ή 0)")
    ap.add_argument("--leverage", type=float, default=None,
                   help="max leverage cap (notional<=lev*equity· default 1x = spot)")
    ap.add_argument("--mode", default=None,
                   help="crossover|trend|mean_reversion|regime (override config)")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    strat, risk = cfg["strategy"], cfg["risk"]
    costs = cfg.get("costs", {})
    equity = args.equity if args.equity is not None else risk["account_equity"]
    fee = args.fee if args.fee is not None else float(costs.get("fee_rate", 0.0004))
    spread = args.spread_bps if args.spread_bps is not None \
        else float(costs.get("spread_bps", 0.0))
    min_fee = args.min_fee if args.min_fee is not None \
        else float(costs.get("min_fee", 0.0))

    if args.live:
        from .data_feed import fetch
        ohlcv, source = fetch(args.symbol, args.start, args.end, args.interval)
        source = f"REAL {source}"
    elif args.csv:
        ohlcv = load_csv(args.csv)
        source = f"CSV {args.csv} ({ohlcv.shape[0]} bars)"
    else:
        ohlcv = synthetic_ohlcv(args.bars, seed=args.seed)
        source = f"SYNTHETIC {args.bars} 1h bars (seed={args.seed}) — DEMO, όχι πραγματική αγορά"

    leverage = args.leverage if args.leverage is not None \
        else risk.get("max_leverage", 1.0)
    from .signals import build_signal
    strat = dict(strat)
    if args.mode:
        strat["mode"] = args.mode
    sig, max_hold = build_signal(ohlcv, strat)
    exit_cfg = cfg.get("exit", {})
    emode = exit_cfg.get("mode", "bracket")
    trail = exit_cfg.get("trail_atr", 0.0) if emode in ("trailing", "adaptive") else 0.0
    adx_t = exit_cfg.get("adx_threshold", 0.0) if emode == "adaptive" else 0.0
    report = simulate(
        ohlcv, sig, strat["atr_period"], strat["atr_sl_mult"], risk["rr_ratio"],
        equity=equity, risk_per_trade=risk["risk_per_trade"], fee_rate=fee,
        max_leverage=leverage, max_hold=max_hold, spread_bps=spread, min_fee=min_fee,
        trail_atr=trail, adx_period=int(strat.get("adx_period", 14)), adx_thr=adx_t,
    )
    s = report.summary()

    print("=" * 58)
    print(" BACKTEST REPORT")
    print("=" * 58)
    print(f" Data source     : {source}")
    print(f" Strategy        : {strat.get('mode', 'crossover')}  "
          f"(fastMA={strat['fast_ma']} slowMA={strat['slow_ma']} "
          f"ATR={strat['atr_period']} SLx{strat['atr_sl_mult']} RR={risk['rr_ratio']})")
    print(f" Risk/trade      : {risk['risk_per_trade']*100:.2f}%  | fee/side: {fee*100:.3f}%"
          f"  | slip: {spread:g}bps | min-fee: {min_fee:g} | max lev: {leverage:g}x")
    _exit_desc = (f"{emode}: τάση->trail {trail:g}xATR, πλάγια->RR {risk['rr_ratio']:g} "
                  f"(ADX>={adx_t:g})" if emode == "adaptive"
                  else (f"trailing {trail:g}xATR" if trail else f"bracket RR {risk['rr_ratio']:g}"))
    print(f" Exit            : {_exit_desc}")
    print("-" * 58)
    print(f" Trades          : {s['n_trades']}  (wins {s['wins']} / losses {s['losses']})")
    print(f" Win rate        : {s['win_rate']*100:.1f}%")
    print(f" Gross PnL       : {s['gross_pnl']:+.2f}")
    print(f" Total fees      : -{s['total_fees']:.2f}")
    print(f" Net PnL (a.fees): {s['net_pnl']:+.2f}  ({s['return_pct']:+.2f}%)")
    print(f" Profit factor   : {s['profit_factor']}")
    print(f" Sharpe          : {s['sharpe']}")
    print(f" Max drawdown    : {s['max_drawdown_pct']:.2f}%")
    print(f" Equity          : {s['start_equity']:.2f} -> {s['end_equity']:.2f}")
    print("=" * 58)


if __name__ == "__main__":
    main()
