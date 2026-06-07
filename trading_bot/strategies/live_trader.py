"""
live_trader — ο «εγκέφαλος» του auto bot: ξεκινάει, διαβάζει candles, και
αποφασίζει μόνος του ENTRY / EXIT / STOP-LOSS / TAKE-PROFIT.

Λογική ανά νέα (κλεισμένη) μπάρα:
  1) Αν είμαστε ΣΕ ΘΕΣΗ:
       - hit Stop-Loss   -> EXIT (ζημία)
       - hit Take-Profit -> EXIT (κέρδος)
       - σήμα αντιστροφής -> EXIT (signal flip)
       - πέρασε max_hold  -> EXIT (time-exit)
  2) Αν είμαστε FLAT και υπάρχει σήμα -> ENTER:
       entry = τιμή κλεισίματος
       stop  = entry ∓ (atr_sl_mult × ATR)          (ATR-based stop-loss)
       target= entry ± (rr_ratio × απόσταση stop)    (take-profit)
       qty   = (equity × risk%) / απόσταση stop      (1% risk sizing, lev cap)

Δύο τρόποι εκτέλεσης:
  • REPLAY (demo): παίζει ιστορικές μπάρες μιας περιόδου μπάρα-μπάρα και δείχνει
    τις αποφάσεις — σαν να «έζησε» ο bot εκείνες τις ημέρες.
  • LIVE: κάθε X δευτερόλεπτα τραβάει νέες μπάρες και αποφασίζει σε πραγματικό
    χρόνο (dry-run/paper by default).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np

from .backtest import _atr
from .data_feed import fetch
from .signals import build_signal


@dataclass
class Position:
    side: str            # "buy" | "sell"
    entry: float
    stop: float
    target: float
    qty: float
    bar: int


class LiveTrader:
    """Stateful auto-trader: μία ανοιχτή θέση τη φορά, με SL/TP/flip/time exit."""

    def __init__(self, strat: dict, risk: dict, equity: float = 10000.0,
                fee_rate: float = 0.0004, max_leverage: float = 1.0,
                log=print) -> None:
        self.strat = strat
        self.atr_period = int(strat.get("atr_period", 14))
        self.atr_sl_mult = float(strat.get("atr_sl_mult", 1.5))
        self.rr = float(risk.get("rr_ratio", 2.0))
        self.risk_pct = float(risk.get("risk_per_trade", 0.01))
        self.fee = fee_rate
        self.lev = max_leverage
        self.equity = equity
        self.start_equity = equity
        self.log = log
        self.pos: Position | None = None
        self.closed: list[dict] = []

    # ---- μία μπάρα: η καρδιά του bot --------------------------------
    def on_bar(self, ohlcv: np.ndarray, label: str = "") -> None:
        """Καλείται με όλα τα candles μέχρι & συμπεριλαμβανομένης της νέας μπάρας."""
        i = ohlcv.shape[0] - 1
        high, low, close = ohlcv[i, 1], ohlcv[i, 2], ohlcv[i, 3]
        sig, max_hold = build_signal(ohlcv, self.strat)
        cur = int(sig[i])

        # 1) διαχείριση ανοιχτής θέσης
        if self.pos is not None:
            p, reason, px = self.pos, None, None
            if p.side == "buy":
                if low <= p.stop:
                    reason, px = "STOP-LOSS", p.stop
                elif high >= p.target:
                    reason, px = "TAKE-PROFIT", p.target
            else:
                if high >= p.stop:
                    reason, px = "STOP-LOSS", p.stop
                elif low <= p.target:
                    reason, px = "TAKE-PROFIT", p.target
            if reason is None and cur != 0 and \
                    ((p.side == "buy" and cur == -1) or (p.side == "sell" and cur == 1)):
                reason, px = "SIGNAL-FLIP", float(close)
            if reason is None and max_hold > 0 and (i - p.bar) >= max_hold:
                reason, px = "TIME-EXIT", float(close)
            if reason is not None:
                self._exit(px, reason, label)

        # 2) είσοδος αν είμαστε flat
        if self.pos is None and cur != 0:
            atr = _atr(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], self.atr_period)
            if np.isnan(atr[i]) or atr[i] <= 0:
                return
            self._enter(cur, float(close), float(atr[i]), i, label)

    def _enter(self, sig: int, entry: float, atr: float, bar: int,
              label: str) -> None:
        dist = self.atr_sl_mult * atr
        qty = min((self.equity * self.risk_pct) / dist,
                 (self.lev * self.equity) / entry)   # 1% risk + leverage cap
        if sig == 1:
            side, stop, target = "buy", entry - dist, entry + self.rr * dist
        else:
            side, stop, target = "sell", entry + dist, entry - self.rr * dist
        self.pos = Position(side, entry, stop, target, qty, bar)
        risk_amt = qty * dist
        self.log(f"{label}  ▶ ENTER {side.upper():4s} @ {entry:.2f}  "
                f"SL {stop:.2f}  TP {target:.2f}  qty {qty:.6f}  "
                f"(risk ${risk_amt:.2f})")

    def _exit(self, px: float, reason: str, label: str) -> None:
        p = self.pos
        assert p is not None
        gross = (px - p.entry) * p.qty if p.side == "buy" \
            else (p.entry - px) * p.qty
        fee = self.fee * p.qty * p.entry + self.fee * p.qty * px
        net = gross - fee
        self.equity += net
        self.closed.append({"side": p.side, "entry": p.entry, "exit": px,
                            "reason": reason, "net": net})
        emoji = "✅" if net > 0 else "❌"
        self.log(f"{label}  ◀ EXIT  {reason:11s} @ {px:.2f}  "
                f"PnL ${net:+.2f} {emoji}  equity ${self.equity:.2f}")
        self.pos = None

    def summary(self) -> dict:
        wins = sum(1 for t in self.closed if t["net"] > 0)
        net = sum(t["net"] for t in self.closed)
        return {"trades": len(self.closed), "wins": wins,
                "losses": len(self.closed) - wins,
                "net": round(net, 2),
                "return_pct": round(100 * net / self.start_equity, 2),
                "end_equity": round(self.equity, 2)}


# ---- REPLAY (demo): «ζήσε» ιστορικές μπάρες ------------------------
def replay(symbol: str, start: str, end: str, interval: str, strat: dict,
          risk: dict, equity: float, fee: float, lev: float) -> None:
    from datetime import datetime, timezone
    ohlcv, src = fetch(symbol, start, end, interval)
    print(f"REPLAY {src} | strategy={strat.get('mode')} | equity ${equity:.0f} "
          f"| fee {fee*100:.3f}%/side\n")
    warm = max(int(strat.get("mr_lookback", 50)), int(strat.get("slow_ma", 26)),
              int(strat.get("trend_ma", 200)) if strat.get("mode") in
              ("trend", "regime", "regime_donchian") else 0) + 5
    trader = LiveTrader(strat, risk, equity, fee, lev)
    n = ohlcv.shape[0]
    for i in range(warm, n):
        # ετικέτα μπάρας (αύξων αριθμός — οι πραγματικές ώρες εξαρτώνται από feed)
        label = f"bar {i:4d}"
        trader.on_bar(ohlcv[:i + 1], label)
    # κλείσε ανοιχτή θέση στο τέλος
    if trader.pos is not None:
        trader._exit(float(ohlcv[-1, 3]), "EOD-CLOSE", f"bar {n-1:4d}")
    s = trader.summary()
    print("\n" + "=" * 56)
    print(f" ΣΥΝΟΨΗ: {s['trades']} trades (W{s['wins']}/L{s['losses']}) | "
          f"NET ${s['net']:+.2f} ({s['return_pct']:+.2f}%) | "
          f"equity ${s['end_equity']:.2f}")
    print("=" * 56)


# ---- LIVE: poll πραγματικά δεδομένα κάθε X δευτερόλεπτα -----------
def live(symbol: str, interval: str, poll: int, strat: dict, risk: dict,
        equity: float, fee: float, lev: float, lookback_bars: int = 300) -> None:
    from datetime import datetime, timezone, timedelta
    print(f"LIVE {symbol} {interval} | poll {poll}s | strategy={strat.get('mode')} "
          f"| DRY-RUN (paper)\n")
    trader = LiveTrader(strat, risk, equity, fee, lev)
    last_n = 0
    while True:
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            ohlcv, _ = fetch(symbol, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            print(f"[live] fetch error: {exc}; retry in {poll}s")
            time.sleep(poll)
            continue
        # δράση μόνο όταν εμφανιστεί ΝΕΑ κλεισμένη μπάρα
        if ohlcv.shape[0] > last_n:
            last_n = ohlcv.shape[0]
            ts = now.strftime("%H:%M")
            trader.on_bar(ohlcv, ts)
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto bot trader (entry/exit/SL)")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--mode", default=None, help="override strategy.mode")
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--replay", action="store_true", help="replay ιστορικών μπαρών")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--live", action="store_true", help="live polling (paper)")
    ap.add_argument("--poll", type=int, default=60, help="live poll seconds")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    strat, risk = dict(cfg["strategy"]), cfg["risk"]
    if args.mode:
        strat["mode"] = args.mode
    equity = args.equity if args.equity is not None else risk.get("account_equity", 10000.0)
    lev = risk.get("max_leverage", 1.0)

    if args.live:
        live(args.symbol, args.interval, args.poll, strat, risk, equity, args.fee, lev)
    else:
        if not (args.start and args.end):
            raise SystemExit("replay χρειάζεται --start και --end (YYYY-MM-DD)")
        replay(args.symbol, args.start, args.end, args.interval, strat, risk,
               equity, args.fee, lev)


if __name__ == "__main__":
    main()
