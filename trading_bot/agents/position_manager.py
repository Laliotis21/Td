"""
PositionManagerAgent — κλείνει ανοιχτές θέσεις σε SL/TP και καταγράφει realized PnL.

Το live execution skeleton άνοιγε θέσεις αλλά δεν τις έκλεινε ποτέ (κανένα realized
PnL). Αυτός ο agent συμπληρώνει τον βρόχο:

  ExecutionAgent ──[result: open]──► PositionManager (poll τιμή ανά PM_POLL)
                                          │  price ≤ SL  ή  ≥ TP ;
                                          ▼
                                     close_trade(pnl) ──[position.closed]──► Orchestrator
                                          └──► Journal / SQLite (status=closed, pnl)

- PAPER (dry-run): τιμή από public Coinbase ticker· κλείσιμο υπολογιστικά (P&L σε $).
- LIVE futures: τιμή/κλείσιμο μέσω BinanceFutures (testnet) — market reduce-only.

Έτσι ο OptimizationAgent διαβάζει επιτέλους ΠΡΑΓΜΑΤΙΚΑ closed trades με PnL, και ο
Orchestrator ελευθερώνει slots (open_positions) + ενημερώνει το daily drawdown.
"""
from __future__ import annotations

import asyncio
import os

from core.bus import AsyncMessageBus
from core.database import Database
from core.journal import Journal
from core.messages import (
    TOPIC_POSITION_CLOSED,
    TOPIC_RESULT,
    ExecutionResult,
    PositionClosed,
)

from .base import BaseAgent


class PositionManagerAgent(BaseAgent):
    name = "positions"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal,
                poll_interval: float | None = None) -> None:
        super().__init__(bus, db, journal)
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.market = os.getenv("BINANCE_MARKET", "spot").lower()
        self.fee_rate = float(os.getenv("POSITION_FEE", "0.0004"))
        self.poll = poll_interval if poll_interval is not None \
            else float(os.getenv("PM_POLL", "10"))
        self._open: dict[int, dict] = {}     # trade_id -> position dict
        self._realized = 0.0
        self._client = self._init_live_client()

    # --- live client (testnet futures) ---------------------------------
    def _init_live_client(self):
        if self.dry_run or self.market != "futures":
            return None
        key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
        if not (key and secret):
            return None
        from core.binance_futures import BinanceFutures
        return BinanceFutures(
            key, secret, testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true")

    # --- price source --------------------------------------------------
    def _coinbase_symbol(self, ticker: str) -> str:
        """'BTCUSDT'/'BTC/USD' -> 'BTC-USD' (Coinbase product)."""
        t = ticker.upper().replace("-", "").replace("/", "")
        for q in ("USDT", "USDC", "USD"):
            if t.endswith(q):
                return f"{t[:-len(q)]}-USD"
        return ticker

    def _price(self, ticker: str) -> float | None:
        """Τρέχουσα τιμή — live futures client ή public Coinbase ticker."""
        try:
            if self._client is not None:
                return self._client.price(ticker.upper().replace("-", ""))
            from strategies.data_feed import spot_price
            return spot_price(self._coinbase_symbol(ticker))
        except Exception as exc:  # noqa: BLE001
            self.log.warning("price fetch failed for %s: %s", ticker, exc)
            return None

    # --- lifecycle -----------------------------------------------------
    async def run(self) -> None:
        results_q = self.bus.subscribe(TOPIC_RESULT)
        self.log.info("position manager ready (dry_run=%s, market=%s, poll=%ss)",
                      self.dry_run, self.market, self.poll)
        await asyncio.gather(self._intake(results_q), self._monitor())

    async def _intake(self, q: asyncio.Queue) -> None:
        """Καταγράφει νέες ανοιχτές θέσεις από τα ExecutionResults."""
        while True:
            res: ExecutionResult = await q.get()
            if res.status in {"filled", "simulated"} and res.trade_id > 0:
                self._open[res.trade_id] = {
                    "ticker": res.ticker, "side": res.side, "qty": res.qty,
                    "entry": res.entry, "sl": res.stop_loss, "tp": res.take_profit,
                    "dry_run": res.dry_run,
                }
                self.log.info("tracking position #%d %s %s qty=%.6f entry=%.2f "
                              "SL=%.2f TP=%.2f", res.trade_id, res.side, res.ticker,
                              res.qty, res.entry, res.stop_loss, res.take_profit)

    async def _monitor(self) -> None:
        """Poll loop: ελέγχει SL/TP για κάθε ανοιχτή θέση."""
        while True:
            await asyncio.sleep(self.poll)
            for tid in list(self._open):
                pos = self._open.get(tid)
                if pos is None:
                    continue
                price = await asyncio.to_thread(self._price, pos["ticker"])
                if price is None:
                    continue
                hit = self._check(pos, price)
                if hit is not None:
                    await self._close(tid, pos, *hit)

    @staticmethod
    def _check(pos: dict, price: float) -> tuple[float, str] | None:
        """(exit_price, reason) αν χτυπήθηκε SL/TP στην τρέχουσα τιμή, αλλιώς None."""
        long = pos["side"] in {"buy", "yes"}
        if long:
            if price <= pos["sl"]:
                return pos["sl"], "stop_loss"
            if price >= pos["tp"]:
                return pos["tp"], "take_profit"
        else:
            if price >= pos["sl"]:
                return pos["sl"], "stop_loss"
            if price <= pos["tp"]:
                return pos["tp"], "take_profit"
        return None

    async def _close(self, tid: int, pos: dict, exit_price: float,
                    reason: str) -> None:
        long = pos["side"] in {"buy", "yes"}
        qty, entry = pos["qty"], pos["entry"]
        gross = (exit_price - entry) * qty if long else (entry - exit_price) * qty
        fee = self.fee_rate * qty * (entry + exit_price)
        pnl = gross - fee
        # live futures: market-close (reduce-only) + cancel residual orders
        if self._client is not None and not pos["dry_run"]:
            sym = pos["ticker"].upper().replace("-", "")
            try:
                await asyncio.to_thread(self._client.close_position, sym)
                await asyncio.to_thread(self._client.cancel_all, sym)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("live close failed #%d: %s", tid, exc)
        await self.db.close_trade(tid, exit_price, pnl)
        self._realized += pnl
        self._open.pop(tid, None)
        await self.record(
            "position.closed",
            f"{reason} {pos['side']} {pos['ticker']} entry={entry:.2f} "
            f"exit={exit_price:.2f} PnL={pnl:+.2f} (realized total {self._realized:+.2f})",
            {"trade_id": tid, "pnl": pnl, "reason": reason},
        )
        self.bus.publish(TOPIC_POSITION_CLOSED, PositionClosed(
            trade_id=tid, ticker=pos["ticker"], side=pos["side"], qty=qty,
            entry=entry, exit=exit_price, pnl=pnl, reason=reason,
            dry_run=pos["dry_run"]))
