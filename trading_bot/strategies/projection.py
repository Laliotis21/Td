"""
projection — προβολή κεφαλαίου βάσει των ΠΡΑΓΜΑΤΙΚΩΝ backtested αποδόσεων του
scanner (όχι αυθαίρετων υποθέσεων). Δείχνει:

  1) "Blind" portfolio: trade ΟΛΟ το universe equal-weight (ρεαλιστικό, χωρίς
     hindsight) — τι θα έβγαζες αν δεν ήξερες ποια θα δούλευαν.
  2) "Hindsight" portfolio: μόνο τα ιστορικά κερδοφόρα & στατιστικά πιο αξιόπιστα
     (net%>0 ΚΑΙ trades>=min) — ΠΡΟΣΟΧΗ: selection bias, αισιόδοξο άνω όριο.

⚠️  Είναι προβολή ΠΑΡΕΛΘΟΝΤΩΝ backtest, ΟΧΙ πρόβλεψη. Past performance ≠ future.

Χρήση: python -m strategies.projection --capital 100 --interval 1d --range 6mo
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .scanner import DEFAULT_UNIVERSE, scan


def _portfolio_return(rows: list[dict]) -> float:
    """Equal-weight portfolio period-return (μέσος των per-symbol net%)/100."""
    if not rows:
        return 0.0
    return float(np.mean([r["net_pct"] for r in rows]) / 100.0)


def _project(capital: float, r_period: float, periods_per_year: float,
            years: float) -> list[tuple[float, float]]:
    """Compounding προβολή· επιστρέφει [(year_fraction, equity), ...]."""
    out = []
    total_periods = int(round(periods_per_year * years))
    eq = capital
    for p in range(1, total_periods + 1):
        eq *= (1.0 + r_period)
        out.append((p / periods_per_year, eq))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Capital projection from backtest")
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--capitals", default="100,1000,10000",
                   help="comma-separated επίπεδα κεφαλαίου για σύγκριση scaling")
    ap.add_argument("--spread-bps", type=float, default=5.0,
                   help="slippage ανά side σε bps (default 5 = 0.05%%)")
    ap.add_argument("--min-fee", type=float, default=1.0,
                   help="ελάχιστη προμήθεια ανά side σε $ (fixed broker cost)")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--range", dest="range_", default="6mo")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--min-trades", type=int, default=4,
                   help="ελάχιστα trades για στατιστική αξιοπιστία (hindsight set)")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    strat, risk = dict(cfg["strategy"]), cfg["risk"]
    if args.mode:
        strat["mode"] = args.mode
    universe = [s.strip() for s in args.symbols.split(",") if s.strip()] \
        or DEFAULT_UNIVERSE

    span_years = {"3mo": 0.25, "6mo": 0.5, "1y": 1.0, "2y": 2.0, "5y": 5.0}\
        .get(args.range_, 0.5)
    ppyr = 1.0 / span_years  # περίοδοι ανά έτος (annualization)

    capitals = [float(c) for c in args.capitals.split(",")] if args.capitals \
        else [args.capital]

    print(f"Backtesting {len(universe)} instruments "
          f"(strategy={strat.get('mode')}, {args.interval}/{args.range_}) "
          f"| frictions: spread {args.spread_bps:.0f}bps, min fee ${args.min_fee:.2f}/side\n")

    for cap in capitals:
        # ξανατρέχουμε το backtest ΣΤΟ συγκεκριμένο κεφάλαιο (το min_fee κάνει
        # τις αποδόσεις capital-dependent — μικρό κεφάλαιο τιμωρείται)
        rows = scan(universe, strat, risk, args.interval, args.range_,
                   equity=cap, spread_bps=args.spread_bps, min_fee=args.min_fee)
        blind = rows
        hind = [r for r in rows if r["net_pct"] > 0 and r["trades"] >= args.min_trades]

        print("#" * 70)
        print(f"#  ΚΕΦΑΛΑΙΟ ${cap:,.0f}")
        print("#" * 70)
        for tag, basket in [("BLIND (όλο το universe)", blind),
                            (f"HINDSIGHT (κερδοφόρα & trades>={args.min_trades})", hind)]:
            r = _portfolio_return(basket)
            ann = (1.0 + r) ** ppyr - 1.0
            print(f"\n  {tag}: {len(basket)} instruments | "
                  f"{args.range_} return {r*100:+.2f}% -> annualized {ann*100:+.2f}%")
            if basket:
                for yr in (1, 3):
                    eq = cap * (1.0 + r) ** (ppyr * yr)
                    print(f"     ${cap:,.0f} -> μετά {yr} έτος/η: ${eq:,.2f} "
                          f"(κέρδος ${eq-cap:+,.2f})")
        print()

    print("⚠️  ΕΠΙΦΥΛΑΞΕΙΣ:")
    print("  • Προβολή ΠΑΡΕΛΘΟΝΤΩΝ backtest, ΟΧΙ πρόβλεψη — το μέλλον θα διαφέρει.")
    print("  • HINDSIGHT = selection bias (διαλέχτηκαν οι νικητές εκ των υστέρων).")
    print("  • Τώρα ΣΥΜΠΕΡΙΛΑΜΒΑΝΟΝΤΑΙ spread + min fee -> μικρό κεφάλαιο "
          "τιμωρείται δυσανάλογα.")
    print("  • Λίγα trades/περίοδο -> χαμηλή στατιστική σημαντικότητα.")


if __name__ == "__main__":
    main()
