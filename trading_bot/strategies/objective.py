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


# ======================================================================
# Fee-aware, strategy-aware objective (walk-forward) — νεότερος μηχανισμός
# που χρησιμοποιεί ο OptimizationAgent. Σε αντίθεση με το `objective()`
# παραπάνω (που βελτιστοποιεί ΜΟΝΟ crossover σε R-multiples, χωρίς fees), εδώ:
#   * χτίζεται το σήμα της ΠΡΑΓΜΑΤΙΚΗΣ στρατηγικής (config mode) μέσω build_signal
#   * τρέχει ο currency-accurate simulator με fees (runner.simulate)
#   * το score είναι net return% μείον ποινή drawdown -> "profit after fees"
# ======================================================================

# tunable παράμετροι ανά mode (η σειρά ορίζει το διάνυσμα x του optimizer)
PARAM_SPECS: dict[str, list[str]] = {
    "crossover":       ["fast_ma", "slow_ma", "atr_sl_mult", "rr_ratio"],
    "trend":           ["fast_ma", "slow_ma", "trend_ma", "atr_sl_mult", "rr_ratio"],
    "donchian":        ["donchian_lb", "atr_sl_mult", "rr_ratio"],
    "mean_reversion":  ["mr_lookback", "mr_z", "atr_sl_mult", "rr_ratio"],
    "regime":          ["fast_ma", "slow_ma", "adx_threshold", "atr_sl_mult", "rr_ratio"],
    "regime_donchian": ["donchian_lb", "adx_threshold", "atr_sl_mult", "rr_ratio"],
}

_BOUNDS: dict[str, tuple[float, float]] = {
    "fast_ma": (3, 30),
    "slow_ma": (15, 100),
    "trend_ma": (50, 250),
    "donchian_lb": (10, 60),
    "mr_lookback": (20, 100),
    "mr_z": (1.0, 3.5),
    "adx_threshold": (15.0, 40.0),
    "atr_sl_mult": (0.5, 4.0),
    "rr_ratio": (1.0, 4.0),
}
_INT_PARAMS = {"fast_ma", "slow_ma", "trend_ma", "donchian_lb", "mr_lookback"}


def param_bounds(mode: str) -> list[tuple[float, float]]:
    """Bounds για τον differential_evolution, ανάλογα με τη στρατηγική."""
    return [_BOUNDS[name] for name in PARAM_SPECS.get(mode, PARAM_SPECS["crossover"])]


def decode_mode(x: np.ndarray, mode: str) -> dict[str, float]:
    """Διάνυσμα optimizer -> named params της συγκεκριμένης στρατηγικής."""
    names = PARAM_SPECS.get(mode, PARAM_SPECS["crossover"])
    out: dict[str, float] = {}
    for name, val in zip(names, np.asarray(x, dtype=float)):
        out[name] = int(round(val)) if name in _INT_PARAMS else float(val)
    if "fast_ma" in out and "slow_ma" in out:        # εγγύηση fast < slow
        out["slow_ma"] = max(out["slow_ma"], out["fast_ma"] + 1)
    return out


def simulate_summary(x: np.ndarray, ohlcv: np.ndarray, mode: str,
                    base_strat: dict, atr_period: int = 14,
                    equity: float = 10000.0, risk: float = 0.01,
                    fee: float = 0.0004, lev: float = 1.0) -> dict:
    """Fee-aware backtest της στρατηγικής `mode` με params x -> summary dict."""
    from . import runner
    from .signals import build_signal
    p = decode_mode(x, mode)
    strat = {**base_strat, **p, "mode": mode}
    sig, max_hold = build_signal(ohlcv, strat)
    return runner.simulate(
        ohlcv, sig, atr_period, float(p["atr_sl_mult"]), float(p["rr_ratio"]),
        equity=equity, risk_per_trade=risk, fee_rate=fee, max_leverage=lev,
        max_hold=max_hold).summary()


def net_score(x: np.ndarray, ohlcv: np.ndarray, mode: str, base_strat: dict,
             atr_period: int = 14, equity: float = 10000.0, risk: float = 0.01,
             fee: float = 0.0004, lev: float = 1.0, dd_penalty: float = 0.5) -> float:
    """Score (higher=better): net return% − dd_penalty·maxDD%, ποινή για λίγα trades."""
    s = simulate_summary(x, ohlcv, mode, base_strat, atr_period, equity, risk, fee, lev)
    if s["n_trades"] < 5:
        return -1e3
    return float(s["return_pct"] - dd_penalty * s["max_drawdown_pct"])


def net_cost(x: np.ndarray, ohlcv: np.ndarray, mode: str, base_strat: dict,
            atr_period: int = 14, equity: float = 10000.0, risk: float = 0.01,
            fee: float = 0.0004, lev: float = 1.0) -> float:
    """Κόστος προς ελαχιστοποίηση (= −net_score) για scipy optimizers."""
    return -net_score(x, ohlcv, mode, base_strat, atr_period, equity, risk, fee, lev)
