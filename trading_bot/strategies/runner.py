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
            "sharpe": round(metrics.sharpe_ratio(rets), 3) if rets.size else 0.0,
            "max_drawdown_pct": round(100 * metrics.max_drawdown(equity), 3),
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
        }


def run(ohlcv: np.ndarray, fast_ma: int, slow_ma: int, atr_period: int,
       atr_sl_mult: float, rr_ratio: float, equity: float = 10000.0,
       risk_per_trade: float = 0.01, fee_rate: float = 0.0004) -> BacktestReport:
    """
    Long-only MA crossover με ATR stop. Κάθε trade ρισκάρει `risk_per_trade` του
    *τρέχοντος* equity· qty από risk/stop-distance· fee = fee_rate*notional*2
    (entry+exit). Επιστρέφει αναλυτικό report.
    """
    report = BacktestReport(start_equity=equity, fee_rate=fee_rate)
    if ohlcv.ndim != 2 or ohlcv.shape[0] < int(slow_ma) + 2:
        return report

    high, low, close = ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    fast = _sma(close, fast_ma)
    slow = _sma(close, slow_ma)
    atr = _atr(high, low, close, atr_period)
    n = close.size

    cross_up = (fast[:-1] <= slow[:-1]) & (fast[1:] > slow[1:])
    entries = np.where(cross_up)[0] + 1

    cur_equity = equity
    for i in entries:
        if i >= n or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        entry = float(close[i])
        stop_dist = atr_sl_mult * float(atr[i])
        if stop_dist <= 0:
            continue
        stop = entry - stop_dist
        target = entry + rr_ratio * stop_dist
        risk_amount = cur_equity * risk_per_trade
        qty = risk_amount / stop_dist          # 1% risk sizing
        notional = qty * entry

        exit_price, exit_idx = None, n - 1
        for j in range(i + 1, n):
            if low[j] <= stop:
                exit_price, exit_idx = stop, j
                break
            if high[j] >= target:
                exit_price, exit_idx = target, j
                break
        if exit_price is None:                 # ανοιχτό στο τέλος -> κλείσε στο last close
            exit_price = float(close[-1])

        gross = (exit_price - entry) * qty
        fee = fee_rate * notional + fee_rate * (qty * exit_price)  # entry + exit
        report.trades.append(Trade(i, exit_idx, "buy", entry, exit_price, qty,
                                   gross, fee))
        cur_equity += gross - fee              # compounding του equity για το sizing
    return report


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
    ap.add_argument("--fee", type=float, default=0.0004,
                   help="fee rate ανά side (default 0.04%% Binance futures taker)")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    strat, risk = cfg["strategy"], cfg["risk"]
    equity = args.equity if args.equity is not None else risk["account_equity"]

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

    report = run(
        ohlcv, strat["fast_ma"], strat["slow_ma"], strat["atr_period"],
        strat["atr_sl_mult"], risk["rr_ratio"], equity=equity,
        risk_per_trade=risk["risk_per_trade"], fee_rate=args.fee,
    )
    s = report.summary()

    print("=" * 58)
    print(" BACKTEST REPORT")
    print("=" * 58)
    print(f" Data source     : {source}")
    print(f" Params          : fastMA={strat['fast_ma']} slowMA={strat['slow_ma']} "
          f"ATR={strat['atr_period']} SLx{strat['atr_sl_mult']} RR={risk['rr_ratio']}")
    print(f" Risk/trade      : {risk['risk_per_trade']*100:.2f}%  | fee/side: {args.fee*100:.3f}%")
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
