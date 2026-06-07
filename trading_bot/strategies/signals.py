"""
Signal generators — κάθε στρατηγική παράγει per-bar σήμα εισόδου:
  +1 = open LONG, -1 = open SHORT, 0 = τίποτα.

Ο simulator (runner.simulate) είναι κοινός για όλες, ώστε να συγκρίνονται δίκαια
στο ίδιο execution engine (ATR stop + RR target, leverage cap, fees, optional
time-exit).
"""
from __future__ import annotations

import numpy as np

from .backtest import _sma


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    window = max(2, int(window))
    out = np.full(x.size, np.nan, dtype=float)
    for i in range(window - 1, x.size):
        out[i] = x[i - window + 1:i + 1].std(ddof=1)
    return out


def ma_crossover(ohlcv: np.ndarray, fast_ma: int, slow_ma: int) -> np.ndarray:
    """Κλασικό crossover: +1 cross-up, -1 cross-down (trend-following)."""
    close = ohlcv[:, 3]
    fast, slow = _sma(close, fast_ma), _sma(close, slow_ma)
    n = close.size
    sig = np.zeros(n, dtype=int)
    up = (fast[:-1] <= slow[:-1]) & (fast[1:] > slow[1:])
    dn = (fast[:-1] >= slow[:-1]) & (fast[1:] < slow[1:])
    sig[1:][up] = 1
    sig[1:][dn] = -1
    return sig


def ma_crossover_trend(ohlcv: np.ndarray, fast_ma: int, slow_ma: int,
                      trend_ma: int) -> np.ndarray:
    """
    Crossover ΜΕ trend filter: long μόνο όταν close > trend MA (π.χ. 200),
    short μόνο όταν close < trend MA. Φιλτράρει counter-trend σήματα.
    """
    close = ohlcv[:, 3]
    trend = _sma(close, trend_ma)
    sig = ma_crossover(ohlcv, fast_ma, slow_ma)
    above = close > trend
    # ακύρωσε longs σε downtrend & shorts σε uptrend (και όπου trend=nan)
    sig[(sig == 1) & ~(above == True)] = 0   # noqa: E712 (above μπορεί nan)
    sig[(sig == -1) & ~(above == False)] = 0  # noqa: E712
    return sig


def mean_reversion(ohlcv: np.ndarray, lookback: int = 50,
                  z_entry: float = 2.0) -> np.ndarray:
    """
    Mean-reversion (z-score / Bollinger): +1 (buy) όταν η τιμή είναι πολύ ΚΑΤΩ
    από τον κινητό μέσο (z < -z_entry, oversold), -1 (sell/short) όταν πολύ ΠΑΝΩ
    (z > +z_entry, overbought). Στοιχηματίζει σε επιστροφή στον μέσο.
    """
    close = ohlcv[:, 3]
    mean = _sma(close, lookback)
    sd = _rolling_std(close, lookback)
    z = (close - mean) / np.where((sd == 0) | np.isnan(sd), np.nan, sd)
    sig = np.zeros(close.size, dtype=int)
    sig[z < -z_entry] = 1     # oversold -> long
    sig[z > z_entry] = -1     # overbought -> short
    sig[np.isnan(z)] = 0
    return sig
