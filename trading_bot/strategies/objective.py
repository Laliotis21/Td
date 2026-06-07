"""
Objective function για τον optimizer.

Δοθέντος ενός διανύσματος παραμέτρων και ιστορικού OHLCV, τρέχει το backtest και
επιστρέφει ένα **κόστος προς ελαχιστοποίηση** (= -score). Ο score συνδυάζει Sharpe
με ποινή για μεγάλο drawdown, ώστε ο optimizer να μην κυνηγά απλώς απόδοση.

Διάνυσμα παραμέτρων x = [fast_ma, slow_ma, atr_sl_mult, rr_ratio].
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .backtest import run_backtest

# bounds που χρησιμοποιεί ο differential_evolution
PARAM_BOUNDS = [
    (3, 30),     # fast_ma
    (15, 100),   # slow_ma
    (0.5, 4.0),  # atr_sl_mult
    (1.0, 4.0),  # rr_ratio
]


def decode(x: np.ndarray) -> dict[str, float]:
    """Μετατρέπει το διάνυσμα του optimizer σε named παραμέτρους."""
    fast_ma = int(round(x[0]))
    slow_ma = int(round(x[1]))
    return {
        "fast_ma": fast_ma,
        "slow_ma": max(slow_ma, fast_ma + 1),  # εγγύηση fast < slow
        "atr_sl_mult": float(x[2]),
        "rr_ratio": float(x[3]),
    }


def score(x: np.ndarray, ohlcv: np.ndarray, atr_period: int = 14) -> float:
    """Score (higher=better): Sharpe - 2*MaxDrawdown, με ποινή για λίγα trades."""
    p = decode(x)
    pnls = run_backtest(
        ohlcv, p["fast_ma"], p["slow_ma"], atr_period,
        p["atr_sl_mult"], p["rr_ratio"],
    )
    if pnls.size < 5:                    # πολύ λίγα trades -> αναξιόπιστο
        return -10.0
    m = metrics.summarize(pnls)
    return m["sharpe"] - 2.0 * m["max_drawdown"]


def objective(x: np.ndarray, ohlcv: np.ndarray, atr_period: int = 14) -> float:
    """Κόστος προς ελαχιστοποίηση από τον scipy optimizer."""
    return -score(x, ohlcv, atr_period)
