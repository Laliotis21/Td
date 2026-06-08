"""
main.py — entrypoint του Multi-Agent Trading Bot.

Στήνει τα κοινά κομμάτια (bus, DB, config, journal) και εκκινεί όλους τους agents
ως ανεξάρτητα asyncio tasks μέσα στο ίδιο event loop. Καθαρό shutdown σε
SIGINT/SIGTERM με ακύρωση όλων των tasks.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal

from dotenv import load_dotenv

from agents.execution_agent import ExecutionAgent
from agents.livetrader_agent import LiveTraderAgent
from agents.optimization_agent import OptimizationAgent
from agents.orchestrator import Orchestrator
from agents.polymarket_agent import PolymarketAgent
from agents.position_manager import PositionManagerAgent
from agents.tradingview_agent import TradingViewAgent
from core.bus import AsyncMessageBus
from core.config_manager import ConfigManager
from core.database import Database
from core.journal import Journal, setup_logging

log = logging.getLogger("main")


async def main() -> None:
    load_dotenv()
    setup_logging(os.getenv("LOG_PATH", "trading.log"))

    # --- shared infrastructure ---
    bus = AsyncMessageBus()
    db = Database(os.getenv("DB_PATH", "trading.db"))
    await db.connect()
    config = ConfigManager(os.getenv("CONFIG_PATH", "config.json"))
    await config.load()
    journal = Journal(db)

    # --- agents ---
    agents = [
        TradingViewAgent(bus, db, journal),
        PolymarketAgent(bus, db, journal, config),
        LiveTraderAgent(bus, db, journal, config),
        ExecutionAgent(bus, db, journal),
        PositionManagerAgent(bus, db, journal, config),
        Orchestrator(bus, db, journal, config),
        OptimizationAgent(bus, db, journal, config),
    ]

    tasks = [asyncio.create_task(a.run(), name=a.name) for a in agents]
    log.info("started %d agents: %s", len(tasks),
             ", ".join(a.name for a in agents))

    # --- graceful shutdown ---
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # π.χ. Windows

    done, pending = await asyncio.wait(
        [*tasks, asyncio.create_task(stop.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    log.info("shutting down...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await db.close()
    log.info("bye.")


if __name__ == "__main__":
    asyncio.run(main())
