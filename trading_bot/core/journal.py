"""
Journal — ενιαίο σύστημα καταγραφής αποφάσεων.

Κάθε σημαντική απόφαση agent καταγράφεται (α) στο αρχείο `trading.log` ως
structured γραμμή και (β) στον πίνακα `agent_decisions` της SQLite. Έτσι ο
Optimization agent μπορεί αργότερα να διαβάσει το ίδιο "ημερολόγιο" είτε ως
κείμενο είτε ως δομημένα δεδομένα.
"""
from __future__ import annotations

import logging
from typing import Any

from .database import Database


def setup_logging(log_path: str, level: int = logging.INFO) -> None:
    """Ρυθμίζει root logger -> κονσόλα + αρχείο trading.log."""
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)


class Journal:
    """Helper που γράφει ταυτόχρονα σε log αρχείο και DB."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.log = logging.getLogger("journal")

    async def record(self, agent: str, decision: str, rationale: str = "",
                    payload: dict[str, Any] | None = None) -> None:
        self.log.info("[%s] %s | %s", agent, decision, rationale)
        await self.db.journal(agent, decision, rationale, payload)
