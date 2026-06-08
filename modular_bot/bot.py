"""
TradingBot — ο ενορχηστρωτής. ΚΟΙΝΗ ροή για Crypto & Μετοχές:

    TradeSetup ──► RiskManager (1% sizing + έλεγχος R:R≥1:3) ──► SizedOrder
                                                                     │
                                              ExecutionModule.place_bracket_order
                                            (CryptoExecution Ή StocksExecution)

Η εναλλαγή asset class γίνεται ΜΟΝΟ στο `build_bot()` — όλη η υπόλοιπη λογική
(risk, sizing, ροή) είναι ίδια και κοινή. Αυτό είναι το νόημα του «modular».
"""
from __future__ import annotations

import logging

from core.models import OrderResult, TradeSetup
from core.risk import RiskManager, RiskRejected
from execution.base import ExecutionModule

log = logging.getLogger("bot")


class TradingBot:
    def __init__(self, execution: ExecutionModule, risk: RiskManager) -> None:
        self.execution = execution
        self.risk = risk

    def start(self) -> None:
        self.execution.connect()
        log.info("connected via '%s' execution module", self.execution.name)

    def stop(self) -> None:
        self.execution.disconnect()

    def execute_signal(self, setup: TradeSetup) -> OrderResult:
        """Ένα σήμα -> risk gate/sizing -> bracket order. Κοινό για ΟΛΑ τα assets."""
        try:
            capital = self.execution.get_capital()
            order = self.risk.build_order(capital, setup)   # 1% sizing + R:R check
        except RiskRejected as exc:
            log.warning("trade rejected (%s): %s", setup.symbol, exc)
            return OrderResult(ok=False, broker=self.execution.name,
                              symbol=setup.symbol, error=f"risk: {exc}")
        except Exception as exc:                            # π.χ. get_capital network
            log.error("pre-trade error (%s): %s", setup.symbol, exc)
            return OrderResult(ok=False, broker=self.execution.name,
                              symbol=setup.symbol, error=str(exc))
        log.info("APPROVED %s %s qty=%.6f R:R=%.2f risk=%.2f", order.side,
                 order.symbol, order.quantity, order.rr_ratio, order.risk_amount)
        return self.execution.place_bracket_order(order)


def build_bot(mode: str, risk_per_trade: float = 0.01, min_rr: float = 3.0,
             **execution_kwargs) -> TradingBot:
    """
    Factory — γυρνά TradingBot με το σωστό execution module ανάλογα με το `mode`:
      mode="crypto" -> CryptoExecution (CCXT)
      mode="stocks" -> StocksExecution (ib_insync)
    Τα `execution_kwargs` περνάνε στον constructor του module (π.χ. exchange_id, port).
    """
    risk = RiskManager(risk_per_trade=risk_per_trade, min_rr=min_rr)
    mode = mode.lower()
    if mode == "crypto":
        from execution.crypto_ccxt import CryptoExecution
        execution: ExecutionModule = CryptoExecution(**execution_kwargs)
    elif mode == "stocks":
        from execution.stocks_ib import StocksExecution
        execution = StocksExecution(**execution_kwargs)
    else:
        raise ValueError(f"Άγνωστο mode '{mode}' — επίλεξε 'crypto' ή 'stocks'")
    return TradingBot(execution, risk)
