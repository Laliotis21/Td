"""
CRYPTO EXECUTION MODULE — μέσω CCXT (Binance/Bybit futures, sandbox/testnet).

Δέχεται έτοιμη `SizedOrder` από το Core και στέλνει bracket ως **συνδυασμό**:
    market entry  +  reduce-only STOP_MARKET (SL)  +  reduce-only TAKE_PROFIT_MARKET (TP)

Γιατί έτσι: στα crypto futures (Binance/Bybit) τα reduce-only conditional orders
είναι ο πιο φορητός τρόπος για bracket μέσω του unified CCXT API. (Σε SPOT θα
χρειαζόσουν OCO — δες σχόλιο στο place_bracket_order.)

Testnet/Sandbox: `client.set_sandbox_mode(True)` -> χτυπά τα testnet endpoints.
"""
from __future__ import annotations

import os

from core.models import OrderResult, SizedOrder

from .base import ExecutionModule


class CryptoExecution(ExecutionModule):
    name = "crypto"

    def __init__(self, exchange_id: str = "binance", api_key: str = "",
                secret: str = "", testnet: bool = True, quote: str = "USDT",
                default_type: str = "future") -> None:
        self.exchange_id = exchange_id
        self.api_key = api_key or os.getenv("CRYPTO_API_KEY", "")
        self.secret = secret or os.getenv("CRYPTO_API_SECRET", "")
        self.testnet = testnet
        self.quote = quote                  # νόμισμα βάσης (margin), π.χ. USDT
        self.default_type = default_type    # "future" | "spot"
        self.client = None
        self._ccxt = None                   # κρατάμε το module για τα exception types

    # --- σύνδεση -------------------------------------------------------
    def connect(self) -> None:
        try:
            import ccxt
        except ImportError as exc:
            raise RuntimeError("Λείπει το ccxt — εγκατάστησε: pip install ccxt") from exc
        self._ccxt = ccxt
        try:
            exchange_cls = getattr(ccxt, self.exchange_id)
        except AttributeError as exc:
            raise RuntimeError(f"Άγνωστο exchange '{self.exchange_id}'") from exc

        self.client = exchange_cls({
            "apiKey": self.api_key,
            "secret": self.secret,
            "enableRateLimit": True,
            "options": {"defaultType": self.default_type},
        })
        if self.testnet:
            self.client.set_sandbox_mode(True)
        try:
            self.client.load_markets()
        except Exception as exc:            # network / auth / geo-block
            raise RuntimeError(f"CCXT σύνδεση/load_markets απέτυχε: {exc}") from exc

    # --- κεφάλαιο ------------------------------------------------------
    def get_capital(self) -> float:
        try:
            balance = self.client.fetch_balance()
        except Exception as exc:
            raise RuntimeError(f"fetch_balance απέτυχε: {exc}") from exc
        return float(balance.get("total", {}).get(self.quote, 0.0))

    # --- bracket order -------------------------------------------------
    def place_bracket_order(self, order: SizedOrder) -> OrderResult:
        if self.client is None:
            return OrderResult(ok=False, broker=self.name, error="δεν έγινε connect()")
        ccxt = self._ccxt
        sym, qty, exit_side = order.symbol, order.quantity, order.exit_side
        try:
            # 1) MARKET entry
            entry = self.client.create_order(sym, "market", order.side, qty)
            # 2) STOP-LOSS (reduce-only stop-market)
            sl = self.client.create_order(
                sym, "STOP_MARKET", exit_side, qty, None,
                {"stopPrice": order.stop_loss, "reduceOnly": True})
            # 3) TAKE-PROFIT (reduce-only take-profit-market)
            tp = self.client.create_order(
                sym, "TAKE_PROFIT_MARKET", exit_side, qty, None,
                {"stopPrice": order.take_profit, "reduceOnly": True})
            # SPOT note: αντί για τα παραπάνω, θα έκανες market entry + create_order
            #            τύπου 'OCO' (αν το υποστηρίζει το exchange) για SL/TP μαζί.
            return OrderResult(
                ok=True, broker=self.name, symbol=sym, quantity=qty,
                entry_id=str(entry.get("id", "")), stop_id=str(sl.get("id", "")),
                take_id=str(tp.get("id", "")),
                raw={"entry": entry, "stop": sl, "take": tp})
        except ccxt.InsufficientFunds as exc:
            return OrderResult(ok=False, broker=self.name, symbol=sym,
                              error=f"Ανεπαρκές υπόλοιπο/ρευστότητα: {exc}")
        except ccxt.NetworkError as exc:
            return OrderResult(ok=False, broker=self.name, symbol=sym,
                              error=f"Πρόβλημα δικτύου: {exc}")
        except ccxt.BaseError as exc:       # κάθε άλλο σφάλμα ανταλλακτηρίου
            return OrderResult(ok=False, broker=self.name, symbol=sym,
                              error=f"Σφάλμα ανταλλακτηρίου: {exc}")

    def disconnect(self) -> None:
        self.client = None                  # CCXT REST -> δεν κρατά persistent socket
