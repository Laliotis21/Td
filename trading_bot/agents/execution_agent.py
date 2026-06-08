"""
ExecutionAgent — εκτελεί εγκεκριμένες εντολές στο Binance με αυστηρό risk control.

- Λαμβάνει `OrderIntent` από το bus (TOPIC_ORDER).
- Σε DRY_RUN (default) προσομοιώνει το order και το καταγράφει στο journal/DB.
- Σε live mode καλεί το python-binance (σε `asyncio.to_thread`) με retry/backoff
  για rate limits (-1003) και network timeouts.
- Πάντα τοποθετεί Stop-Loss / Take-Profit και ελέγχει διαθέσιμο margin.

Το position sizing γίνεται upstream στον Orchestrator (1% risk)· εδώ γίνεται ο
τελικός έλεγχος margin και η αποστολή.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import replace

from core.bus import AsyncMessageBus
from core.database import Database
from core.journal import Journal
from core.messages import TOPIC_ORDER, TOPIC_RESULT, ExecutionResult, OrderIntent

from .base import BaseAgent

_MAX_RETRIES = 4


class ExecutionAgent(BaseAgent):
    name = "execution"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal) -> None:
        super().__init__(bus, db, journal)
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        # spot (testnet.binance.vision) ή futures (testnet.binancefuture.com)
        self.market = os.getenv("BINANCE_MARKET", "spot").lower()
        self.quote = os.getenv("BINANCE_QUOTE", "USDT")
        self._client = None if self.dry_run else self._init_binance()

    def _norm(self, ticker: str) -> str:
        """'BTC-USD' / 'BTC/USD' -> 'BTCUSDT' (Binance format)."""
        t = ticker.upper().replace("-", "").replace("/", "")
        if t.endswith("USD") and not t.endswith("USDT") and self.quote == "USDT":
            t = t[:-3] + "USDT"
        return t

    def _init_binance(self):
        key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_API_SECRET")
        if not key or not secret:
            self.log.warning("Binance keys missing -> forcing DRY_RUN")
            self.dry_run = True
            return None
        # FUTURES: δικός μας signed REST client (παρακάμπτει το spot ping του
        # python-binance που γεω-μπλοκάρεται με 451 σε πολλά clouds)
        if self.market == "futures":
            from core.binance_futures import BinanceFutures
            client = BinanceFutures(key, secret, testnet=self.testnet)
            try:
                bal = client.available_balance(self.quote)
                self.log.info("Binance futures testnet=%s AUTH OK — %.2f %s available",
                              self.testnet, bal, self.quote)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("Binance futures auth failed (%s) -> DRY_RUN", exc)
                self.dry_run = True
                return None
            return client
        # SPOT: python-binance (δουλεύει μόνο όπου το testnet.binance.vision είναι
        # προσβάσιμο — π.χ. από το μηχάνημά σου, όχι από geo-blocked cloud)
        try:
            from binance.client import Client
        except ImportError:
            self.log.warning("python-binance not installed -> forcing DRY_RUN")
            self.dry_run = True
            return None
        try:
            client = Client(key, secret, testnet=self.testnet)
            acct = client.get_account()
            bal = next((float(b["free"]) for b in acct["balances"]
                        if b["asset"] == self.quote), 0.0)
            self.log.info("Binance spot testnet=%s AUTH OK — %.2f %s",
                          self.testnet, bal, self.quote)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Binance spot auth/connectivity failed (%s) -> DRY_RUN", exc)
            self.dry_run = True
            return None
        return client

    # --- margin check --------------------------------------------------
    async def _has_margin(self, order: OrderIntent) -> bool:
        if self.dry_run or self._client is None:
            return True
        try:
            notional = order.qty * order.entry
            if self.market == "futures":
                avail = await asyncio.to_thread(
                    self._client.available_balance, self.quote)
                return avail >= notional  # 1x· για leverage ρύθμισε ξεχωριστά
            acct = await asyncio.to_thread(self._client.get_account)
            usdt = next((float(b["free"]) for b in acct["balances"]
                         if b["asset"] == self.quote), 0.0)
            return usdt >= notional
        except Exception as exc:  # noqa: BLE001
            self.log.warning("margin check failed: %s", exc)
            return False

    # --- live order με retry/backoff ----------------------------------
    async def _send_live(self, order: OrderIntent) -> ExecutionResult:
        from binance.exceptions import BinanceAPIException

        side = "BUY" if order.side in {"buy", "yes"} else "SELL"
        sym = self._norm(order.ticker)
        for attempt in range(_MAX_RETRIES):
            try:
                if self.market == "futures":
                    await self._send_futures(order, side, sym)
                else:
                    await self._send_spot(order, side, sym)
                return ExecutionResult(
                    order_id=order.id, ticker=order.ticker, side=order.side,
                    qty=order.qty, entry=order.entry, stop_loss=order.stop_loss,
                    take_profit=order.take_profit, status="filled", dry_run=False,
                )
            except BinanceAPIException as exc:
                if exc.code == -1003 and attempt < _MAX_RETRIES - 1:  # rate limit
                    wait = 2 ** (attempt + 1)
                    self.log.warning("Binance rate-limit, retry in %ss", wait)
                    await asyncio.sleep(wait)
                    continue
                return ExecutionResult(
                    order_id=order.id, ticker=order.ticker, side=order.side,
                    qty=order.qty, entry=order.entry, stop_loss=order.stop_loss,
                    take_profit=order.take_profit, status="error", dry_run=False,
                    reason=str(exc),
                )
            except (TimeoutError, ConnectionError) as exc:
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                return ExecutionResult(
                    order_id=order.id, ticker=order.ticker, side=order.side,
                    qty=order.qty, entry=order.entry, stop_loss=order.stop_loss,
                    take_profit=order.take_profit, status="error", dry_run=False,
                    reason=str(exc),
                )
        return ExecutionResult(
            order_id=order.id, ticker=order.ticker, side=order.side, qty=order.qty,
            entry=order.entry, stop_loss=order.stop_loss,
            take_profit=order.take_profit, status="error", dry_run=False,
            reason="max retries exceeded",
        )

    async def _send_spot(self, order: OrderIntent, side: str, sym: str) -> None:
        """Spot market entry + OCO (SL/TP). testnet.binance.vision."""
        await asyncio.to_thread(
            self._client.create_order,  # type: ignore[union-attr]
            symbol=sym, side=side, type="MARKET", quantity=round(order.qty, 6))
        await asyncio.to_thread(
            self._client.create_oco_order,  # type: ignore[union-attr]
            symbol=sym, side="SELL" if side == "BUY" else "BUY",
            quantity=round(order.qty, 6),
            price=round(order.take_profit, 2),
            stopPrice=round(order.stop_loss, 2),
            stopLimitPrice=round(order.stop_loss, 2), stopLimitTimeInForce="GTC")

    async def _send_futures(self, order: OrderIntent, side: str, sym: str) -> None:
        """
        Futures: MARKET entry + LIMIT take-profit (reduceOnly). Το stop-loss
        μπαίνει ως STOP_MARKET· αν το testnet το απορρίψει (-4120), γίνεται
        **bot-managed** (ο LiveTrader το κλείνει client-side) αντί να σκάσει.
        """
        from core.binance_futures import BinanceFuturesError

        opp = "SELL" if side == "BUY" else "BUY"
        qty = round(order.qty, 3)
        await asyncio.to_thread(self._client.market_order, sym, side, qty)
        # take-profit ως LIMIT reduce-only (υποστηρίζεται στο testnet)
        await asyncio.to_thread(self._client.limit_reduce, sym, opp, qty,
                               round(order.take_profit, 1))
        # stop-loss: δοκίμασε exchange-side· fallback σε bot-managed
        try:
            await asyncio.to_thread(self._client.stop_market, sym, opp, qty,
                                   round(order.stop_loss, 1))
        except BinanceFuturesError as exc:
            if exc.code == self._client.STOP_UNSUPPORTED:
                self.log.warning("exchange stop-loss unsupported (-4120) -> "
                                "bot-managed SL @ %.1f", order.stop_loss)
            else:
                raise

    async def _execute(self, order: OrderIntent) -> ExecutionResult:
        if not await self._has_margin(order):
            return ExecutionResult(
                order_id=order.id, ticker=order.ticker, side=order.side,
                qty=order.qty, entry=order.entry, stop_loss=order.stop_loss,
                take_profit=order.take_profit, status="rejected", dry_run=self.dry_run,
                reason="insufficient margin",
            )
        if self.dry_run:
            return ExecutionResult(
                order_id=order.id, ticker=order.ticker, side=order.side,
                qty=order.qty, entry=order.entry, stop_loss=order.stop_loss,
                take_profit=order.take_profit, status="simulated", dry_run=True,
            )
        return await self._send_live(order)

    # --- main loop -----------------------------------------------------
    async def run(self) -> None:
        q = self.bus.subscribe(TOPIC_ORDER)
        self.log.info("execution agent ready (dry_run=%s, market=%s, testnet=%s)",
                      self.dry_run, self.market, self.testnet)
        while True:
            order: OrderIntent = await q.get()
            result = await self._execute(order)
            opened = result.status in {"filled", "simulated"}
            trade_id = await self.db.insert_trade(
                order.ticker, order.side, order.qty, order.entry,
                order.stop_loss, order.take_profit,
                status="open" if opened else "rejected",
                strategy=order.strategy, dry_run=result.dry_run,
            )
            if opened:                       # δώσε στο PositionManager το DB id
                result = replace(result, trade_id=trade_id)
            await self.record(
                f"order.{result.status}",
                f"{order.side} {order.qty:.6f} {order.ticker} @ {order.entry} "
                f"SL={order.stop_loss} TP={order.take_profit}",
                {"order_id": order.id, "reason": result.reason},
            )
            self.bus.publish(TOPIC_RESULT, result)
