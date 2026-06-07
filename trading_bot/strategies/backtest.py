"""
Vectorized backtest μιας απλής MA-crossover στρατηγικής με ATR-based stops.

Δέχεται OHLCV (numpy) και παραμέτρους, και επιστρέφει σειρά PnL ανά trade. Είναι
σκόπιμα ελαφρύ & numpy-only ώστε ο optimizer να το τρέχει χιλιάδες φορές με
χαμηλό latency μέσα σε `asyncio.to_thread`.
"""
from __future__ import annotations

import numpy as np


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if x.size < window:
        return np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(x.size, np.nan, dtype=float)
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int) -> np.ndarray:
    period = max(1, int(period))
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return _sma(tr, period)


def run_backtest(ohlcv: np.ndarray, fast_ma: int, slow_ma: int,
                atr_period: int, atr_sl_mult: float, rr_ratio: float) -> np.ndarray:
    """
    ohlcv: array σχήματος (N, 5) με στήλες [open, high, low, close, volume].
    Επιστρέφει 1D array με PnL (σε R-multiples) ανά κλεισμένο trade.

    Λογική: long όταν fast SMA διασταυρώνει πάνω από slow SMA· stop = atr_sl_mult*ATR,
    target = rr_ratio * (entry - stop). Το κάθε trade κλείνει στο πρώτο εκ των SL/TP.
    """
    if ohlcv.ndim != 2 or ohlcv.shape[1] < 4 or ohlcv.shape[0] < int(slow_ma) + 2:
        return np.array([], dtype=float)

    high, low, close = ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    fast = _sma(close, fast_ma)
    slow = _sma(close, slow_ma)
    atr = _atr(high, low, close, atr_period)

    cross_up = (fast[:-1] <= slow[:-1]) & (fast[1:] > slow[1:])
    entries = np.where(cross_up)[0] + 1  # index της επόμενης μπάρας

    pnls: list[float] = []
    n = close.size
    for i in entries:
        if i >= n or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        entry = close[i]
        stop = entry - atr_sl_mult * atr[i]
        risk = entry - stop
        if risk <= 0:
            continue
        target = entry + rr_ratio * risk
        # προσομοίωση: σκάναρε μπροστά μέχρι SL ή TP
        outcome = 0.0
        for j in range(i + 1, n):
            if low[j] <= stop:
                outcome = -1.0          # -1R
                break
            if high[j] >= target:
                outcome = rr_ratio      # +rr_ratio R
                break
        pnls.append(outcome)

    return np.array(pnls, dtype=float)


def synthetic_ohlcv(n: int = 500, seed: int = 7) -> np.ndarray:
    """Παράγει συνθετικό OHLCV (geometric random walk) για demo/δοκιμές."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    vol = rng.uniform(100, 1000, n)
    return np.column_stack([open_, high, low, close, vol])
