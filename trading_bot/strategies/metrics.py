"""
Performance metrics (numpy, vectorized) για αξιολόγηση στρατηγικών/agents.

Χρησιμοποιούνται από τον Optimization agent ώστε να μετρά αντικειμενικά κάθε
στρατηγική: Sharpe Ratio, Win Rate, Profit Factor, Maximum Drawdown.
"""
from __future__ import annotations

import numpy as np

# trading periods ανά έτος (π.χ. 252 ημέρες) — annualization factor για Sharpe
TRADING_PERIODS = 252


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0,
                periods: int = TRADING_PERIODS) -> float:
    """Annualized Sharpe ratio από σειρά per-trade/per-period returns."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods
    sd = excess.std(ddof=1)
    # μηδενική (ή αριθμητικά αμελητέα) μεταβλητότητα -> Sharpe undefined.
    # Επιστρέφουμε 0.0· αποτρέπει και explosion από float-noise σε all-win σειρές.
    if sd < 1e-12:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / sd)


def win_rate(pnls: np.ndarray) -> float:
    """Ποσοστό κερδοφόρων trades (0..1)."""
    p = np.asarray(pnls, dtype=float)
    if p.size == 0:
        return 0.0
    return float((p > 0).sum() / p.size)


def profit_factor(pnls: np.ndarray) -> float:
    """Άθροισμα κερδών / |άθροισμα ζημιών|. inf αν δεν υπάρχουν ζημιές."""
    p = np.asarray(pnls, dtype=float)
    gross_profit = p[p > 0].sum()
    gross_loss = -p[p < 0].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Μέγιστο drawdown (θετικός αριθμός, 0..1) πάνω σε καμπύλη equity.
    Δέχεται είτε equity values είτε cumulative returns > 0.
    """
    e = np.asarray(equity_curve, dtype=float)
    if e.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(e)
    drawdowns = (running_max - e) / np.where(running_max == 0, 1, running_max)
    return float(drawdowns.max())


def equity_from_pnls(pnls: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Φτιάχνει equity curve από διαδοχικά PnL (additive)."""
    p = np.asarray(pnls, dtype=float)
    return start + np.cumsum(p)


def summarize(pnls: np.ndarray, risk_fraction: float = 0.01) -> dict[str, float]:
    """
    Συγκεντρωτικό dict με όλα τα metrics για μια σειρά PnL σε R-multiples.

    Τα PnL είναι ήδη risk-normalized (R-multiples: -1R loss, +rrR win), οπότε
    αποτελούν απευθείας τη σειρά per-trade returns για το Sharpe. Για το equity
    curve / drawdown συνθέτουμε **πολλαπλασιαστικά** με το `risk_fraction` (1%
    ανά trade), ώστε το equity να μένει θετικό και το drawdown φραγμένο στο [0,1].
    """
    p = np.asarray(pnls, dtype=float)
    if p.size == 0:
        return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "max_drawdown": 0.0, "n_trades": 0.0, "total_pnl": 0.0}
    equity = np.cumprod(1.0 + p * risk_fraction)
    return {
        "sharpe": sharpe_ratio(p),          # Sharpe των R-multiples (system quality)
        "win_rate": win_rate(p),
        "profit_factor": profit_factor(p),
        "max_drawdown": max_drawdown(equity),
        "n_trades": float(p.size),
        "total_pnl": float(p.sum()),
    }
