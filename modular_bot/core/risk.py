"""
CORE RISK MANAGEMENT — η κοινή λογική για Crypto ΚΑΙ Μετοχές.

Εντελώς ανεξάρτητη από ανταλλακτήριο/broker. Κάνει δύο πράγματα:

  1) Position Sizing με τον κανόνα του 1%:
        Quantity = (Capital * risk_per_trade) / |Entry - Stop Loss|
     (απόλυτη τιμή ώστε να δουλεύει και σε short θέσεις)

  2) Έλεγχο ελάχιστης αναλογίας Risk-to-Reward (default 1:3) πριν εγκριθεί το trade.
     Αν δεν περνά -> RiskRejected (το trade αγνοείται, ΧΩΡΙΣ να σταλεί εντολή).
"""
from __future__ import annotations

from .models import SizedOrder, TradeSetup


class RiskRejected(Exception):
    """Το trade απορρίφθηκε από το risk gate (π.χ. R:R < ελάχιστο, κακή γεωμετρία)."""


class RiskManager:
    def __init__(self, risk_per_trade: float = 0.01, min_rr: float = 3.0) -> None:
        if not 0.0 < risk_per_trade < 1.0:
            raise ValueError("risk_per_trade πρέπει να είναι στο (0,1) — π.χ. 0.01 = 1%")
        if min_rr <= 0:
            raise ValueError("min_rr πρέπει να είναι θετικό (π.χ. 3.0 = 1:3)")
        self.risk_per_trade = risk_per_trade
        self.min_rr = min_rr

    # --- βασικές μετρικές ----------------------------------------------
    @staticmethod
    def stop_distance(setup: TradeSetup) -> float:
        d = abs(setup.entry - setup.stop_loss)
        if d <= 0:
            raise RiskRejected("Μηδενική απόσταση stop (entry == stop_loss)")
        return d

    def risk_reward(self, setup: TradeSetup) -> float:
        """Reward / Risk. π.χ. entry 100, SL 96, TP 112 -> 12/4 = 3.0 (1:3)."""
        reward = abs(setup.take_profit - setup.entry)
        return reward / self.stop_distance(setup)

    # --- position sizing (ο κανόνας του 1%) ----------------------------
    def position_size(self, capital: float, setup: TradeSetup) -> float:
        """Quantity = (Capital * 1%) / (Entry - Stop Loss)."""
        if capital <= 0:
            raise RiskRejected(f"Μη έγκυρο κεφάλαιο: {capital}")
        return (capital * self.risk_per_trade) / self.stop_distance(setup)

    # --- έγκριση & σύνθεση εντολής -------------------------------------
    def build_order(self, capital: float, setup: TradeSetup) -> SizedOrder:
        """
        Επικυρώνει γεωμετρία + R:R και υπολογίζει ποσότητα. Πετά RiskRejected
        αν δεν περνά. Επιστρέφει SizedOrder (κοινή για όλα τα execution modules).
        """
        self._validate_geometry(setup)
        rr = self.risk_reward(setup)
        if rr < self.min_rr:
            raise RiskRejected(
                f"R:R {rr:.2f} < ελάχιστο {self.min_rr:.2f} — το trade αγνοείται")
        qty = self.position_size(capital, setup)
        if qty <= 0:
            raise RiskRejected(f"Μη έγκυρη ποσότητα: {qty}")
        return SizedOrder(
            symbol=setup.symbol, side=setup.side, quantity=qty,
            entry=setup.entry, stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            risk_amount=capital * self.risk_per_trade, rr_ratio=rr,
        )

    @staticmethod
    def _validate_geometry(setup: TradeSetup) -> None:
        """Long: stop < entry < tp. Short: tp < entry < stop. Αλλιώς λάθος setup."""
        if setup.is_long:
            if not (setup.stop_loss < setup.entry < setup.take_profit):
                raise RiskRejected("Long setup: απαιτείται stop < entry < take_profit")
        else:
            if not (setup.take_profit < setup.entry < setup.stop_loss):
                raise RiskRejected("Short setup: απαιτείται take_profit < entry < stop")
