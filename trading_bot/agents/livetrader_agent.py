"""
LiveTraderAgent — ο «εγκέφαλος» που τρέχει 24/7, σκανάρει ένα universe instruments
(stocks/ETFs/crypto), διαβάζει candles, και όταν μια στρατηγική δίνει σήμα εισόδου
στέλνει `Signal` στο bus. Από εκεί ακολουθεί η υπάρχουσα ροή:

    LiveTraderAgent ──[signal]──► Orchestrator (risk gate + 1% sizing + SL/TP)
                                      └──[order]──► ExecutionAgent (paper/live)
                                                        └──► Journal / SQLite

Δηλαδή το sizing και τα Stop-Loss/Take-Profit τα υπολογίζει ο Orchestrator (μία
πηγή αλήθειας), ενώ αυτός ο agent αποφασίζει ΠΟΥ/ΠΟΤΕ υπάρχει σήμα. Για να μη
«σπαμάρει», στέλνει σήμα μόνο όταν αλλάζει η κατεύθυνση ανά σύμβολο (fresh entry).
"""
from __future__ import annotations

import asyncio
import os

from core.bus import AsyncMessageBus
from core.config_manager import ConfigManager
from core.database import Database
from core.journal import Journal
from core.messages import TOPIC_SIGNAL, Signal
from strategies.data_feed import fetch_yahoo
from strategies.signals import build_signal

from .base import BaseAgent

_YRANGE = {"1d": "6mo", "1h": "3mo", "30m": "1mo", "15m": "1mo",
          "5m": "5d", "1m": "5d"}


class LiveTraderAgent(BaseAgent):
    name = "livetrader"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal,
                config: ConfigManager) -> None:
        super().__init__(bus, db, journal)
        self.config = config
        self.universe = [s.strip() for s in
                        os.getenv("LIVE_UNIVERSE", "GLD,TLT,IWM,NVDA").split(",")
                        if s.strip()]
        self.interval = os.getenv("LIVE_INTERVAL", "1d")
        self.poll = int(os.getenv("LIVE_POLL", "300"))
        self._last_sig: dict[str, int] = {}     # κατεύθυνση που έχει ήδη σταλεί

    async def run(self) -> None:
        self.log.info("livetrader scanning %s (%s, every %ss)",
                      self.universe, self.interval, self.poll)
        await self.record("livetrader.start",
                         f"universe={self.universe} interval={self.interval}")
        while True:
            try:
                await self._scan()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — μη πεθάνει ο agent
                self.log.exception("livetrader scan error: %s", exc)
            await asyncio.sleep(self.poll)

    async def _scan(self) -> None:
        cfg = await self.config.get()
        strat = cfg.get("strategy", {})
        yrange = _YRANGE.get(self.interval, "6mo")
        for symbol in self.universe:
            try:
                ohlcv = await asyncio.to_thread(fetch_yahoo, symbol,
                                               self.interval, yrange)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("fetch %s failed: %s", symbol, exc)
                continue
            if ohlcv.shape[0] < 60:
                continue
            sig, _ = build_signal(ohlcv, strat)
            cur = int(sig[-1])
            prev = self._last_sig.get(symbol, 0)
            if cur != 0 and cur != prev:           # fresh entry (άλλαξε κατεύθυνση)
                action = "buy" if cur == 1 else "sell"
                price = float(ohlcv[-1, 3])
                s = Signal(source=self.name, ticker=symbol, action=action,  # type: ignore[arg-type]
                          price=price, meta={"strategy": strat.get("mode", "?"),
                                            "interval": self.interval})
                await self.db.insert_signal(self.name, symbol, action, price,
                                           {"interval": self.interval})
                await self.record("signal.emitted",
                                 f"{action} {symbol} @ {price:.2f} "
                                 f"({strat.get('mode')})", {"signal_id": s.id})
                self.bus.publish(TOPIC_SIGNAL, s)
            self._last_sig[symbol] = cur
