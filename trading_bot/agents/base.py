"""
BaseAgent — αφηρημένη βάση για κάθε agent.

Κάθε agent κρατά αναφορά στο bus, στη DB και στον journal, και υλοποιεί μια
async `run()` που είναι ο κύριος βρόχος ζωής του (τρέχει ως asyncio.Task).
"""
from __future__ import annotations

import abc
import logging

from core.bus import AsyncMessageBus
from core.database import Database
from core.journal import Journal


class BaseAgent(abc.ABC):
    name: str = "agent"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal) -> None:
        self.bus = bus
        self.db = db
        self.journal = journal
        self.log = logging.getLogger(self.name)

    async def record(self, decision: str, rationale: str = "",
                    payload: dict | None = None) -> None:
        """Καταγραφή απόφασης στο ενιαίο Journal (log + DB)."""
        await self.journal.record(self.name, decision, rationale, payload)

    @abc.abstractmethod
    async def run(self) -> None:
        """Κύριος βρόχος του agent. Πρέπει να σέβεται το asyncio.CancelledError."""
        raise NotImplementedError
