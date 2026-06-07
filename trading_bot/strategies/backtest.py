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

    Bidirectional, ένα position τη φορά:
      cross-up   -> LONG  (stop = atr_sl_mult*ATR κάτω, target = rr_ratio*risk πάνω)
      cross-down -> SHORT (stop πάνω, target κάτω)
    Κάθε trade -> -1R (stop) ή +rr_ratio R (target). Νέα είσοδος μόνο αφού κλείσει
    η προηγούμενη θέση.
    """
    if ohlcv.ndim != 2 or ohlcv.shape[1] < 4 or ohlcv.shape[0] < int(slow_ma) + 2:
        return np.array([], dtype=float)

    high, low, close = ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    fast = _sma(close, fast_ma)
    slow = _sma(close, slow_ma)
    atr = _atr(high, low, close, atr_period)
    n = close.size

    sig = np.zeros(n, dtype=int)
    up = (fast[:-1] <= slow[:-1]) & (fast[1:] > slow[1:])
    dn = (fast[:-1] >= slow[:-1]) & (fast[1:] < slow[1:])
    sig[1:][up] = 1
    sig[1:][dn] = -1

    pnls: list[float] = []
    pos: dict | None = None
    for i in range(1, n):
        if pos is not None and i > pos["i"]:
            out = None
            if pos["long"]:
                if low[i] <= pos["stop"]:
                    out = -1.0
                elif high[i] >= pos["target"]:
                    out = rr_ratio
            else:
                if high[i] >= pos["stop"]:
                    out = -1.0
                elif low[i] <= pos["target"]:
                    out = rr_ratio
            if out is not None:
                pnls.append(out)
                pos = None
        if pos is None and sig[i] != 0 and not np.isnan(atr[i]) and atr[i] > 0:
            entry = close[i]
            dist = atr_sl_mult * atr[i]
            if dist <= 0:
                continue
            long = sig[i] == 1
            stop = entry - dist if long else entry + dist
            target = entry + rr_ratio * dist if long else entry - rr_ratio * dist
            pos = {"long": long, "stop": stop, "target": target, "i": i}

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
