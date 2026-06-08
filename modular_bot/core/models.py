"""
Κοινά data models — μοιράζονται από το Core και τα Execution modules.

Είναι το «κοινό λεξιλόγιο»: η στρατηγική παράγει `TradeSetup`, ο RiskManager το
μετατρέπει σε `SizedOrder`, και κάθε execution module επιστρέφει `OrderResult`.
Έτσι το ίδιο αντικείμενο ταξιδεύει αναλλοίωτο σε Crypto ΚΑΙ σε Μετοχές.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TradeSetup:
    """Το «σήμα» από τη στρατηγική, ΠΡΙΝ το position sizing (asset-agnostic)."""
    symbol: str                # "BTC/USDT" (crypto) ή "AAPL" (stock)
    side: str                  # "buy" (long) | "sell" (short)
    entry: float
    stop_loss: float
    take_profit: float

    @property
    def is_long(self) -> bool:
        return self.side.lower() == "buy"


@dataclass(frozen=True)
class SizedOrder:
    """Έτοιμη εντολή — μετά τον υπολογισμό ποσότητας & τον έλεγχο R:R."""
    symbol: str
    side: str
    quantity: float
    entry: float
    stop_loss: float
    take_profit: float
    risk_amount: float         # πόσα € ρισκάρουμε (= 1% του κεφαλαίου)
    rr_ratio: float

    @property
    def exit_side(self) -> str:
        """Αντίθετη πλευρά — για τα SL/TP (reduce-only) orders."""
        return "sell" if self.side.lower() == "buy" else "buy"


@dataclass
class OrderResult:
    """Τυποποιημένο αποτέλεσμα εκτέλεσης από οποιοδήποτε execution module."""
    ok: bool
    broker: str
    symbol: str = ""
    quantity: float = 0.0
    entry_id: str = ""
    stop_id: str = ""
    take_id: str = ""
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
