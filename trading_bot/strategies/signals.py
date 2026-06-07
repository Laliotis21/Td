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


def _wilder(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing (RMA) — χρησιμοποιείται στο ADX."""
    out = np.full(x.size, np.nan, dtype=float)
    if x.size < period:
        return out
    out[period - 1] = np.nanmean(x[:period])
    for i in range(period, x.size):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def adx(ohlcv: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average Directional Index (Wilder) — μέτρο **ισχύος τάσης**.
    ADX >= ~25 -> trending· ADX < ~20 -> range/χωρίς τάση. Επιστρέφει array
    μήκους N (leading NaN στο warm-up).
    """
    high, low, close = ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    n = close.size
    if n < 2 * period + 1:
        return np.full(n, np.nan, dtype=float)
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_close = close[:-1]
    tr = np.maximum.reduce([high[1:] - low[1:],
                            np.abs(high[1:] - prev_close),
                            np.abs(low[1:] - prev_close)])
    atr = _wilder(tr, period)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = 100 * _wilder(plus_dm, period) / atr
        minus_di = 100 * _wilder(minus_dm, period) / atr
        denom = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / np.where(denom == 0, np.nan, denom)
    adx_arr = _wilder(dx, period)
    return np.concatenate([[np.nan], adx_arr])   # realign σε μήκος N


def regime_switch(ohlcv: np.ndarray, fast_ma: int = 12, slow_ma: int = 26,
                 trend_ma: int = 200, mr_lookback: int = 50, mr_z: float = 2.0,
                 adx_period: int = 14, adx_threshold: float = 25.0) -> np.ndarray:
    """
    Regime-adaptive: ανά bar, αν ADX >= threshold (τάση) χρησιμοποιεί το
    **trend-filtered crossover**· αλλιώς (range) χρησιμοποιεί **mean-reversion**.
    Έτσι παίρνει την κατάλληλη στρατηγική για το εκάστοτε καθεστώς αγοράς.
    """
    a = adx(ohlcv, adx_period)
    trend_sig = ma_crossover_trend(ohlcv, fast_ma, slow_ma, trend_ma)
    mr_sig = mean_reversion(ohlcv, mr_lookback, mr_z)
    sig = np.where(a >= adx_threshold, trend_sig, mr_sig)
    sig[np.isnan(a)] = 0
    return sig.astype(int)


def build_signal(ohlcv: np.ndarray, s: dict) -> tuple[np.ndarray, int]:
    """
    Dispatcher: επιστρέφει (signal, max_hold) με βάση το `s["mode"]` από το config.
    Επιτρέπει στον Orchestrator/Optimizer να αλλάζει στρατηγική **live** μέσω
    hot-reload, χωρίς αλλαγή κώδικα.
    """
    mode = s.get("mode", "regime")
    fast, slow = int(s.get("fast_ma", 12)), int(s.get("slow_ma", 26))
    if mode == "crossover":
        return ma_crossover(ohlcv, fast, slow), 0
    if mode == "trend":
        return ma_crossover_trend(ohlcv, fast, slow, int(s.get("trend_ma", 200))), 0
    if mode == "mean_reversion":
        return (mean_reversion(ohlcv, int(s.get("mr_lookback", 50)),
                              float(s.get("mr_z", 2.0))),
                int(s.get("mr_max_hold", 24)))
    # default: regime switch
    return (regime_switch(ohlcv, fast, slow, int(s.get("trend_ma", 200)),
                         int(s.get("mr_lookback", 50)), float(s.get("mr_z", 2.0)),
                         int(s.get("adx_period", 14)),
                         float(s.get("adx_threshold", 25.0))),
            int(s.get("mr_max_hold", 24)))


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
