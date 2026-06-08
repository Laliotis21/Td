"""
STOCKS EXECUTION MODULE — μέσω ib_insync (Interactive Brokers, Paper Trading).

Δέχεται έτοιμη `SizedOrder` από το Core και:
  - μετατρέπει την ποσότητα σε **ΑΚΕΡΑΙΟ** αριθμό μετοχών (οι μετοχές δεν αγοράζονται
    σε δεκαδικά· αν βγει < 1 μετοχή -> απορρίπτει με σαφές μήνυμα),
  - στέλνει **BRACKET**: Parent MARKET order + Child STOP (SL) + Child LIMIT (TP),
    με OCA group ώστε όταν γεμίσει το ένα child να ακυρώνεται αυτόματα το άλλο.

Προαπαιτούμενο: να τρέχει το TWS ή το IB Gateway σε **Paper** mode και να δέχεται
API συνδέσεις. Default ports: TWS paper=7497, IB Gateway paper=4002.
"""
from __future__ import annotations

import os

from core.models import OrderResult, SizedOrder

from .base import ExecutionModule


class StocksExecution(ExecutionModule):
    name = "stocks"

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                client_id: int = 1, currency: str = "USD",
                exchange: str = "SMART") -> None:
        self.host = host
        self.port = int(os.getenv("IB_PORT", port))
        self.client_id = client_id
        self.currency = currency
        self.exchange = exchange
        self.ib = None

    # --- σύνδεση -------------------------------------------------------
    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError("Λείπει το ib_insync — pip install ib_insync") from exc
        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
        except Exception as exc:            # ConnectionRefused αν δεν τρέχει TWS/Gateway
            raise RuntimeError(
                f"Σύνδεση IB απέτυχε ({self.host}:{self.port}). Τρέχει το TWS/IB "
                f"Gateway σε Paper mode με ενεργό API; → {exc}") from exc

    # --- κεφάλαιο ------------------------------------------------------
    def get_capital(self) -> float:
        try:
            rows = self.ib.accountSummary()
        except Exception as exc:
            raise RuntimeError(f"accountSummary απέτυχε: {exc}") from exc
        for row in rows:
            if row.tag == "NetLiquidation":
                return float(row.value)
        return 0.0

    # --- bracket order -------------------------------------------------
    def place_bracket_order(self, order: SizedOrder) -> OrderResult:
        if self.ib is None:
            return OrderResult(ok=False, broker=self.name, error="δεν έγινε connect()")
        from ib_insync import LimitOrder, MarketOrder, Stock, StopOrder

        qty = int(order.quantity)           # ΑΚΕΡΑΙΕΣ μετοχές
        if qty < 1:
            return OrderResult(
                ok=False, broker=self.name, symbol=order.symbol,
                error=(f"Ποσότητα {order.quantity:.4f} < 1 μετοχή — αύξησε κεφάλαιο "
                       "ή βάλε στενότερο stop (μεγαλύτερη απόσταση = λιγότερες μετοχές)"))

        action = "BUY" if order.side.lower() == "buy" else "SELL"
        exit_action = "SELL" if action == "BUY" else "BUY"
        try:
            contract = Stock(order.symbol, self.exchange, self.currency)
            self.ib.qualifyContracts(contract)
            oca = f"oca_{order.symbol}_{self.ib.client.getReqId()}"

            # Parent: MARKET (δεν μεταδίδεται ακόμα)
            parent = MarketOrder(action, qty)
            parent.orderId = self.ib.client.getReqId()
            parent.transmit = False

            # Child 1: TAKE-PROFIT (LIMIT), κρεμασμένο στον parent
            take = LimitOrder(exit_action, qty, round(order.take_profit, 2))
            take.orderId = self.ib.client.getReqId()
            take.parentId = parent.orderId
            take.ocaGroup, take.ocaType = oca, 1
            take.transmit = False

            # Child 2: STOP-LOSS (STOP) — το τελευταίο transmit στέλνει ΟΛΟ το bracket
            stop = StopOrder(exit_action, qty, round(order.stop_loss, 2))
            stop.orderId = self.ib.client.getReqId()
            stop.parentId = parent.orderId
            stop.ocaGroup, stop.ocaType = oca, 1
            stop.transmit = True

            for o in (parent, take, stop):
                self.ib.placeOrder(contract, o)
            return OrderResult(
                ok=True, broker=self.name, symbol=order.symbol, quantity=qty,
                entry_id=str(parent.orderId), stop_id=str(stop.orderId),
                take_id=str(take.orderId))
        except Exception as exc:            # qualify / placement / liquidity / margin
            return OrderResult(ok=False, broker=self.name, symbol=order.symbol,
                              error=f"IB bracket απέτυχε: {exc}")

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
        self.ib = None
