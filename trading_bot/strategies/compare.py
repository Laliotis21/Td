"""
compare — σύγκριση στρατηγικών σε πολλά σύμβολα/περιόδους, στο ίδιο execution
engine (fees + 1x leverage cap), out-of-sample (fixed params, χωρίς optimization).

Στρατηγικές:
  crossover      : MA 12/26 (trend-following, bidirectional)
  trend_filtered : MA 12/26 ΜΕ φίλτρο τάσης 200-MA
  mean_reversion : z-score 50/2.0 (αντίθετα στα άκρα), time-exit 24 bars

Χρήση:
  python -m strategies.compare --symbols BTC-USD,ETH-USD,SOL-USD \
      --start 2026-05-01 --end 2026-06-01 --interval 1h
"""
from __future__ import annotations

import argparse

import numpy as np

from . import runner, signals
from .data_feed import fetch

FEE = 0.0004
EQUITY = 10000.0
RISK = 0.01
LEV = 1.0
ATR = 14


def _run_strategies(ohlcv: np.ndarray) -> dict[str, dict]:
    out: dict[str, dict] = {}

    sig = signals.ma_crossover(ohlcv, 12, 26)
    out["crossover"] = runner.simulate(ohlcv, sig, ATR, 1.5, 2.0, EQUITY, RISK,
                                       FEE, LEV).summary()

    sig = signals.ma_crossover_trend(ohlcv, 12, 26, 200)
    out["trend_filtered"] = runner.simulate(ohlcv, sig, ATR, 1.5, 2.0, EQUITY,
                                            RISK, FEE, LEV).summary()

    sig = signals.mean_reversion(ohlcv, 50, 2.0)
    out["mean_reversion"] = runner.simulate(ohlcv, sig, ATR, 2.0, 1.0, EQUITY,
                                            RISK, FEE, LEV, max_hold=24).summary()
    return out


def _row(name: str, s: dict) -> str:
    return (f"  {name:15s} | trades {s['n_trades']:3d} | win {s['win_rate']*100:5.1f}% "
            f"| NET {s['net_pnl']:+9.2f} ({s['return_pct']:+6.2f}%) "
            f"| PF {s['profit_factor']:5.2f} | Sharpe {s['sharpe']:7.2f} "
            f"| DD {s['max_drawdown_pct']:4.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare strategies across symbols")
    ap.add_argument("--symbols", default="BTC-USD,ETH-USD,SOL-USD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--interval", default="1h")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    agg: dict[str, list[float]] = {"crossover": [], "trend_filtered": [],
                                   "mean_reversion": []}

    print(f"Period {args.start} -> {args.end} @ {args.interval} "
          f"| fee {FEE*100:.3f}%/side | lev {LEV:g}x | equity ${EQUITY:.0f}\n")
    for sym in symbols:
        try:
            ohlcv, src = fetch(sym, args.start, args.end, args.interval)
        except Exception as exc:  # noqa: BLE001
            print(f"[{sym}] fetch failed: {exc}")
            continue
        if ohlcv.shape[0] < 210:
            print(f"[{sym}] λίγα bars ({ohlcv.shape[0]}) — το trend-filter 200 "
                  f"θέλει ιστορικό· τα νούμερα ενδεικτικά.")
        print(f"### {sym}  ({ohlcv.shape[0]} bars)")
        res = _run_strategies(ohlcv)
        for name, s in res.items():
            print(_row(name, s))
            agg[name].append(s["return_pct"])
        print()

    print("=" * 70)
    print(" ΜΕΣΟΣ net return % ανά στρατηγική (out-of-sample, fixed params):")
    for name, vals in agg.items():
        if vals:
            print(f"  {name:15s} : {np.mean(vals):+6.2f}%  "
                  f"(ανά σύμβολο: {', '.join(f'{v:+.1f}' for v in vals)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
