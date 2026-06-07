"""
ConfigManager — φόρτωση/αποθήκευση του config.json με ασφαλές hot-reload.

Σχεδιαστική αρχή: το live config κρατιέται ως ένα immutable dict πίσω από έναν
`asyncio.Lock`. Το hot-reload κάνει **atomic swap** του reference — οι αναγνώστες
(Orchestrator, Polymarket agent) διαβάζουν πάντα είτε το παλιό είτε το νέο config,
ποτέ μισο-γραμμένο. Έτσι ο Orchestrator συνεχίζει να σερβίρει webhooks ενώ ο
Optimization agent γράφει νέες παραμέτρους.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import tempfile
from typing import Any

log = logging.getLogger("config")


class ConfigManager:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._config: dict[str, Any] = {}

    async def load(self) -> dict[str, Any]:
        """(Επανα)φορτώνει το config.json από τον δίσκο και κάνει atomic swap."""
        data = await asyncio.to_thread(self._read_file)
        async with self._lock:
            self._config = data
        log.info("config loaded (version=%s)", data.get("version"))
        return data

    def _read_file(self) -> dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def get(self) -> dict[str, Any]:
        """Επιστρέφει βαθύ αντίγραφο του τρέχοντος config (safe για αναγνώστες)."""
        async with self._lock:
            return copy.deepcopy(self._config)

    def snapshot(self) -> dict[str, Any]:
        """
        Lock-free αντίγραφο για hot paths (π.χ. webhook handler).
        Ασφαλές γιατί το `self._config` αντικαθίσταται ατομικά ως reference.
        """
        return copy.deepcopy(self._config)

    @property
    def version(self) -> int:
        return int(self._config.get("version", 0))

    async def save(self, new_config: dict[str, Any]) -> int:
        """
        Γράφει νέο config (atomic, μέσω temp file + os.replace), αυξάνει το
        version, και κάνει swap στη μνήμη. Επιστρέφει το νέο version.
        Καλείται από τον Optimization agent.
        """
        async with self._lock:
            new_config = copy.deepcopy(new_config)
            new_config["version"] = int(self._config.get("version", 0)) + 1
            await asyncio.to_thread(self._write_atomic, new_config)
            self._config = new_config
            log.info("config saved & swapped (version=%s)", new_config["version"])
            return new_config["version"]

    def _write_atomic(self, data: dict[str, Any]) -> None:
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)  # atomic στο ίδιο filesystem
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
