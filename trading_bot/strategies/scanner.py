"""
scanner — σκανάρει ένα **universe** instruments (crypto, stocks, ETFs) και
εντοπίζει (α) ποια έχουν ιστορικό edge με τη στρατηγική και (β) ποια δίνουν
**live σήμα τώρα** (στην τελευταία μπάρα).

Φιλοσοφία: αντί να βελτιστοποιούμε μία στρατηγική σε ένα σύμβολο, εφαρμόζουμε
την ίδια (robust) στρατηγική σε πολλά μη-συσχετισμένα instruments· η διασπορά
μετατρέπει ένα μικρό per-symbol edge σε portfolio edge. Αυτό είναι το brain
ενός μελλοντικού ScannerAgent που τροφοδοτεί τον Orchestrator.

Δεδομένα: Yahoo Finance (public, no-key) — καλύπτει crypto/stocks/ETFs.

Χρήση:
  python -m strategies.scanner                      # default mixed universe, daily
  python -m strategies.scanner --symbols AAPL,SPY,BTC-USD --mode mean_reversion
  python -m strategies.scanner --range 1y --interval 1d --top 15
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import runner
from .data_feed import fetch_yahoo
from .signals import build_signal

# default mixed universe: crypto · mega-cap stocks · ETFs
DEFAULT_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",        # crypto
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",       # stocks
    "SPY", "QQQ", "GLD", "TLT", "IWM",                             # ETFs
]

FEE = 0.0004
EQUITY = 10000.0


def _asset_class(sym: str) -> str:
    if sym.endswith("-USD"):
        return "crypto"
    if sym in {"SPY", "QQQ", "GLD", "TLT", "IWM", "DIA", "EEM", "XLF", "XLE"}:
        return "etf"
    return "stock"


def scan(symbols: list[str], strat: dict, risk: dict, interval: str,
        range_: str) -> list[dict]:
    rows: list[dict] = []
    lev = risk.get("max_leverage", 1.0)
    for sym in symbols:
        try:
            ohlcv = fetch_yahoo(sym, interval, range_)
        except Exception as exc:  # noqa: BLE001
            print(f"[scan] {sym} fetch failed: {exc}")
            continue
        if ohlcv.shape[0] < 60:
            continue
        sig, max_hold = build_signal(ohlcv, strat)
        rep = runner.simulate(ohlcv, sig, strat["atr_period"],
                             strat["atr_sl_mult"], risk["rr_ratio"],
                             equity=EQUITY, risk_per_trade=risk["risk_per_trade"],
                             fee_rate=FEE, max_leverage=lev, max_hold=max_hold)
        s = rep.summary()
        live = int(sig[-1])  # σήμα στην πιο πρόσφατη μπάρα
        rows.append({
            "symbol": sym, "asset": _asset_class(sym), "bars": ohlcv.shape[0],
            "net_pct": s["return_pct"], "win": s["win_rate"], "pf": s["profit_factor"],
            "sharpe": s["sharpe"], "trades": s["n_trades"],
            "live": "BUY" if live == 1 else "SELL" if live == -1 else "-",
        })
    rows.sort(key=lambda r: r["net_pct"], reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-instrument signal scanner")
    ap.add_argument("--symbols", default="", help="comma-separated (default mixed universe)")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--range", dest="range_", default="6mo",
                   help="Yahoo range: 3mo|6mo|1y|2y ...")
    ap.add_argument("--mode", default=None, help="override strategy.mode")
    ap.add_argument("--top", type=int, default=0, help="δείξε μόνο top-N (0=όλα)")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    strat, risk = dict(cfg["strategy"]), cfg["risk"]
    if args.mode:
        strat["mode"] = args.mode
    universe = [s.strip() for s in args.symbols.split(",") if s.strip()] \
        or DEFAULT_UNIVERSE

    print(f"Scanning {len(universe)} instruments | strategy={strat.get('mode')} "
          f"| {args.interval}/{args.range_} | fee {FEE*100:.3f}%/side\n")
    rows = scan(universe, strat, risk, args.interval, args.range_)
    shown = rows[:args.top] if args.top else rows

    print(f"  {'symbol':9s} {'asset':6s} {'bars':>4s} {'net%':>8s} "
          f"{'win%':>6s} {'PF':>5s} {'Sharpe':>7s} {'trades':>6s}  live")
    print("  " + "-" * 66)
    for r in shown:
        flag = f"  >>> {r['live']}" if r["live"] != "-" else ""
        print(f"  {r['symbol']:9s} {r['asset']:6s} {r['bars']:4d} "
              f"{r['net_pct']:+8.2f} {r['win']*100:5.1f}% {r['pf']:5.2f} "
              f"{r['sharpe']:7.2f} {r['trades']:6d}{flag}")

    live = [r for r in rows if r["live"] != "-"]
    profitable = [r for r in rows if r["net_pct"] > 0]
    live_str = ", ".join("{}:{}".format(r["symbol"], r["live"]) for r in live)
    prof_str = ", ".join(r["symbol"] for r in profitable)
    act = [r for r in live if r["net_pct"] > 0]   # live σήμα ΚΑΙ ιστορικό edge
    act_str = ", ".join("{}:{}".format(r["symbol"], r["live"]) for r in act)
    print("\n" + "=" * 68)
    print(f" Ιστορικά κερδοφόρα (net%>0): {len(profitable)}/{len(rows)} "
          f"-> {prof_str or '—'}")
    print(f" LIVE σήματα τώρα: {len(live)} -> {live_str or '—'}")
    print(f" ⭐ ACTIONABLE (live + ιστορικό edge): {act_str or '—'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
