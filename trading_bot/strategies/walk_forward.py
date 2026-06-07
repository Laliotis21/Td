"""
walk_forward — out-of-sample demo του retraining loop σε πραγματικά δεδομένα.

  TRAIN  : βελτιστοποίηση παραμέτρων (scipy differential_evolution) σε ένα
           ιστορικό παράθυρο.
  TEST   : εφαρμογή των βέλτιστων παραμέτρων σε επόμενο, *αόρατο* παράθυρο.

Έτσι βλέπουμε αν το optimization γενικεύει (out-of-sample) αντί να κάνει overfit.
Χρήση:
  python -m strategies.walk_forward --symbol BTC-USD \
      --train-start 2026-05-01 --train-end 2026-05-24 \
      --test-start 2026-05-24  --test-end 2026-05-27 --interval 15m
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import differential_evolution, minimize

from . import objective, runner
from .data_feed import fetch


def _net_cost(x: np.ndarray, ohlcv: np.ndarray, atr_period: int, equity: float,
            risk: float, fee: float, lev: float) -> float:
    """
    Fee-aware objective: ελαχιστοποιεί το αρνητικό του net return (μετά fees),
    με ποινή για drawdown και για πολύ λίγα trades. Έτσι ο optimizer στοχεύει
    απευθείας στο "profit after fees", όχι σε raw R-multiples.
    """
    p = objective.decode(x)
    rep = runner.run(ohlcv, p["fast_ma"], p["slow_ma"], atr_period,
                    p["atr_sl_mult"], p["rr_ratio"], equity=equity,
                    risk_per_trade=risk, fee_rate=fee, max_leverage=lev)
    s = rep.summary()
    if s["n_trades"] < 3:
        return 1e3
    return -(s["return_pct"] - 0.5 * s["max_drawdown_pct"])


def optimize(ohlcv: np.ndarray, atr_period: int, equity: float, risk: float,
            fee: float, lev: float) -> tuple[dict, float]:
    args = (ohlcv, atr_period, equity, risk, fee, lev)
    res = differential_evolution(
        _net_cost, objective.PARAM_BOUNDS, args=args,
        maxiter=60, popsize=15, tol=1e-3, seed=42, polish=False)
    ref = minimize(_net_cost, res.x, args=args, method="Nelder-Mead",
                  options={"maxiter": 300, "xatol": 1e-2, "fatol": 1e-3})
    best_x = ref.x if ref.fun <= res.fun else res.x
    return objective.decode(np.asarray(best_x)), -float(min(ref.fun, res.fun))


def _report(tag: str, ohlcv, p, atr_period, equity, risk, fee, lev=1.0):
    rep = runner.run(ohlcv, p["fast_ma"], p["slow_ma"], atr_period,
                    p["atr_sl_mult"], p["rr_ratio"], equity=equity,
                    risk_per_trade=risk, fee_rate=fee, max_leverage=lev)
    s = rep.summary()
    print(f"\n--- {tag} ---")
    print(f"  params   : fastMA={p['fast_ma']} slowMA={p['slow_ma']} "
          f"SLx{p['atr_sl_mult']:.2f} RR={p['rr_ratio']:.2f}")
    print(f"  trades   : {s['n_trades']} (W{s['wins']}/L{s['losses']}, "
          f"win {s['win_rate']*100:.1f}%)")
    print(f"  gross    : {s['gross_pnl']:+.2f} | fees -{s['total_fees']:.2f} "
          f"| NET {s['net_pnl']:+.2f} ({s['return_pct']:+.2f}%)")
    print(f"  PF {s['profit_factor']} | Sharpe {s['sharpe']} | "
          f"MaxDD {s['max_drawdown_pct']:.2f}% | equity {s['end_equity']:.2f}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward optimize & test")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    risk_cfg = cfg["risk"]
    equity = risk_cfg["account_equity"]
    risk = risk_cfg["risk_per_trade"]
    atr_period = cfg["strategy"]["atr_period"]
    fee = 0.0004
    lev = risk_cfg.get("max_leverage", 1.0)

    train, ts = fetch(args.symbol, args.train_start, args.train_end, args.interval)
    test, es = fetch(args.symbol, args.test_start, args.test_end, args.interval)
    print(f"TRAIN: {ts}\nTEST : {es}  | max leverage {lev:g}x, fee {fee*100:.3f}%/side")

    default = {"fast_ma": cfg["strategy"]["fast_ma"],
               "slow_ma": cfg["strategy"]["slow_ma"],
               "atr_sl_mult": cfg["strategy"]["atr_sl_mult"],
               "rr_ratio": risk_cfg["rr_ratio"]}
    best, score = optimize(train, atr_period, equity, risk, fee, lev)
    print(f"\noptimized net-objective score (train): {score:.3f}")

    print("\n" + "=" * 58)
    _report("TEST · DEFAULT params (baseline)", test, default, atr_period,
            equity, risk, fee, lev)
    _report("TEST · OPTIMIZED params (out-of-sample)", test, best, atr_period,
            equity, risk, fee, lev)
    _report("TRAIN · OPTIMIZED params (in-sample, ref)", train, best, atr_period,
            equity, risk, fee, lev)
    print("=" * 58)


if __name__ == "__main__":
    main()
