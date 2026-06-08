"""
Abstract Execution Module — το «συμβόλαιο» (interface) που υλοποιούν ΟΛΑ τα
execution backends (Crypto/CCXT, Stocks/ib_insync, …).

Εδώ κρύβεται η σπονδυλωτότητα: ο TradingBot μιλά ΜΟΝΟ μέσω αυτής της διεπαφής
και δεν ξέρει (ούτε τον νοιάζει) τι υπάρχει από κάτω. Για να προσθέσεις νέο
asset class (π.χ. forex, futures), φτιάχνεις μια νέα υλοποίηση αυτής της κλάσης.
"""
from __future__ import annotations

import abc

from core.models import OrderResult, SizedOrder


class ExecutionModule(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def connect(self) -> None:
        """Σύνδεση με το API. Πρέπει να πετά σαφές σφάλμα αν αποτύχει."""

    @abc.abstractmethod
    def get_capital(self) -> float:
        """Συνολικό κεφάλαιο/equity του λογαριασμού (στο νόμισμα βάσης)."""

    @abc.abstractmethod
    def place_bracket_order(self, order: SizedOrder) -> OrderResult:
        """Entry + Stop-Loss + Take-Profit ως bracket (ή ισοδύναμος συνδυασμός)."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Καθαρό κλείσιμο σύνδεσης."""

    # --- context manager: καθαρό connect/disconnect ---
    def __enter__(self) -> "ExecutionModule":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
