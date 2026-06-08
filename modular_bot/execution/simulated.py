"""
SIMULATED EXECUTION MODULE — paper trading ΧΩΡΙΣ εξωτερικό API.

Υλοποιεί την ΙΔΙΑ `ExecutionModule` διεπαφή, αλλά δεν συνδέεται πουθενά: κρατά ένα
ρυθμιζόμενο κεφάλαιο και «εκτελεί» το bracket τοπικά (logging + OrderResult). Ιδανικό
για να τρέχεις/τεστάρεις τη ροή με €X κεφάλαιο σε crypto ΚΑΙ stocks, χωρίς keys/TWS.

Επίσης: δείχνει πόσο εύκολα προστίθεται νέο backend — κληρονομείς ExecutionModule,
υλοποιείς 4 μεθόδους, τέλος. Το core risk μένει αναλλοίωτο.
"""
from __future__ import annotations

import logging

from core.models import OrderResult, SizedOrder

from .base import ExecutionModule

log = logging.getLogger("sim")


class SimulatedExecution(ExecutionModule):
    name = "sim"

    def __init__(self, capital: float = 100.0, asset_kind: str = "crypto") -> None:
        self.capital = capital
        self.asset_kind = asset_kind     # "crypto" (fractional) | "stocks" (integer)
        self._n = 0

    def connect(self) -> None:
        log.info("simulated %s account — capital €%.2f", self.asset_kind, self.capital)

    def get_capital(self) -> float:
        return self.capital

    def place_bracket_order(self, order: SizedOrder) -> OrderResult:
        qty = order.quantity
        # stocks: ΑΚΕΡΑΙΕΣ μετοχές (όπως το πραγματικό IB module)
        if self.asset_kind == "stocks":
            qty = float(int(order.quantity))
            if qty < 1:
                return OrderResult(
                    ok=False, broker=self.name, symbol=order.symbol,
                    error=(f"qty {order.quantity:.4f} < 1 μετοχή — €{self.capital:.0f} "
                           "πολύ λίγα για αυτό το stop/τιμή (θες fractional ή φθηνότερη μετοχή)"))
        notional = qty * order.entry
        self._n += 1
        log.info("[SIM] BRACKET #%d %s %s qty=%s @ %.2f | SL=%.2f TP=%.2f | "
                 "notional=€%.2f risk=€%.2f R:R=%.1f", self._n, order.side.upper(),
                 order.symbol, qty, order.entry, order.stop_loss, order.take_profit,
                 notional, order.risk_amount, order.rr_ratio)
        return OrderResult(
            ok=True, broker=self.name, symbol=order.symbol, quantity=qty,
            entry_id=f"sim{self._n}-E", stop_id=f"sim{self._n}-SL",
            take_id=f"sim{self._n}-TP",
            raw={"notional": notional, "risk": order.risk_amount, "rr": order.rr_ratio})

    def disconnect(self) -> None:
        pass
