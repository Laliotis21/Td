"""
Έλεγχοι ορθότητας για τα performance metrics και το backtest/objective.

Τρέξε:  python -m tests.test_metrics   (από το trading_bot/)
Δεν απαιτεί pytest — απλά asserts.
"""
from __future__ import annotations

import numpy as np

from strategies import metrics
from strategies.backtest import run_backtest, synthetic_ohlcv
from strategies.objective import PARAM_BOUNDS, decode, objective


def test_profit_factor() -> None:
    pnls = np.array([2.0, -1.0, 2.0, -1.0])  # κέρδη=4, ζημιές=2 -> PF=2.0
    assert abs(metrics.profit_factor(pnls) - 2.0) < 1e-9


def test_win_rate() -> None:
    pnls = np.array([1.0, -1.0, 1.0, 1.0])   # 3/4
    assert abs(metrics.win_rate(pnls) - 0.75) < 1e-9


def test_max_drawdown() -> None:
    equity = np.array([100.0, 120.0, 90.0, 110.0])  # peak 120 -> trough 90 = 25%
    assert abs(metrics.max_drawdown(equity) - 0.25) < 1e-9


def test_sharpe_positive_for_uptrend() -> None:
    rng = np.random.default_rng(0)
    rets = 0.01 + rng.normal(0, 0.002, 50)  # θετικό mean με μικρή μεταβλητότητα
    assert metrics.sharpe_ratio(rets) > 0


def test_sharpe_zero_for_constant_returns() -> None:
    # μηδενική μεταβλητότητα -> Sharpe undefined· επιστρέφουμε 0.0 με ασφάλεια
    assert metrics.sharpe_ratio(np.full(50, 0.01)) == 0.0


def test_summarize_keys() -> None:
    pnls = np.array([1.0, -1.0, 2.0, -1.0, 1.0])
    s = metrics.summarize(pnls)
    for k in ("sharpe", "win_rate", "profit_factor", "max_drawdown",
              "n_trades", "total_pnl"):
        assert k in s


def test_backtest_runs() -> None:
    ohlcv = synthetic_ohlcv(400)
    pnls = run_backtest(ohlcv, fast_ma=10, slow_ma=30, atr_period=14,
                       atr_sl_mult=1.5, rr_ratio=2.0)
    assert isinstance(pnls, np.ndarray)


def test_objective_finite() -> None:
    ohlcv = synthetic_ohlcv(400)
    x = np.array([10, 30, 1.5, 2.0])
    val = objective(x, ohlcv, 14)
    assert np.isfinite(val)


def test_decode_keeps_fast_below_slow() -> None:
    p = decode(np.array([25, 20, 1.5, 2.0]))  # fast>slow στο input
    assert p["fast_ma"] < p["slow_ma"]


def test_bounds_shape() -> None:
    assert len(PARAM_BOUNDS) == 4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
