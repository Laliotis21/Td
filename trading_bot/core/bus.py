"""
AsyncMessageBus — ελαφρύ in-process pub/sub πάνω σε asyncio.Queue.

Κάθε agent κάνει `subscribe(topic)` και παίρνει μια ουρά· κάθε publisher κάνει
`publish(topic, msg)`. Το fan-out είναι μη-μπλοκάρον: ο publisher δεν περιμένει
τους subscribers, οπότε ένα αργό κατάντη στάδιο δεν μπλοκάρει π.χ. το webhook
handler (κρίσιμο για να μη χάνονται webhooks κατά το hot-reload).
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger("bus")


class AsyncMessageBus:
    def __init__(self, max_queue: int = 1000) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._max_queue = max_queue

    def subscribe(self, topic: str) -> asyncio.Queue:
        """Επιστρέφει μια νέα ουρά εγγεγραμμένη στο topic."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers[topic].append(q)
        return q

    def publish(self, topic: str, message: Any) -> None:
        """
        Δημοσιεύει σε όλους τους subscribers χωρίς να μπλοκάρει.
        Αν μια ουρά είναι γεμάτη, το μήνυμα πέφτει με warning αντί να κρεμάσει
        τον publisher (back-pressure safety).
        """
        for q in self._subscribers.get(topic, ()):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                log.warning("dropping message on full queue for topic=%s", topic)

    async def publish_async(self, topic: str, message: Any) -> None:
        """Παραλλαγή που περιμένει χώρο στην ουρά (όταν θες εγγυημένη παράδοση)."""
        for q in self._subscribers.get(topic, ()):
            await q.put(message)
