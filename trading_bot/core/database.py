"""
Database — αποθήκευση ιστορικού/audit σε SQLite (async μέσω aiosqlite).

Schema:
  signals          : κάθε εισερχόμενο σήμα (TradingView/Polymarket)
  trades           : κάθε εντολή/trade (open & closed) με SL/TP/PnL
  agent_decisions  : το Trading Journal — μία εγγραφή ανά απόφαση agent
  params_history   : ιστορικό deployαρισμένων παραμέτρων από τον optimizer
  llm_lessons      : προβλέψεις Polymarket vs πραγματικό αποτέλεσμα (few-shot πηγή)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiosqlite

log = logging.getLogger("db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    source    TEXT NOT NULL,
    ticker    TEXT NOT NULL,
    action    TEXT NOT NULL,
    price     REAL,
    raw_json  TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    ticker     TEXT NOT NULL,
    side       TEXT NOT NULL,
    qty        REAL NOT NULL,
    entry      REAL NOT NULL,
    sl         REAL,
    tp         REAL,
    exit_price REAL,
    pnl        REAL,
    status     TEXT NOT NULL,        -- open | closed | rejected
    strategy   TEXT,
    dry_run    INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    agent       TEXT NOT NULL,
    decision    TEXT NOT NULL,
    rationale   TEXT,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS params_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    version      INTEGER NOT NULL,
    params_json  TEXT NOT NULL,
    score        REAL,
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS llm_lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    market_id   TEXT NOT NULL,
    predicted   REAL,                -- predicted prob YES
    actual      REAL,                -- 1.0 / 0.0 όταν λυθεί η αγορά
    lesson_text TEXT
);
"""


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        log.info("database ready at %s", self.path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database not connected"
        return self._db

    # --- writes --------------------------------------------------------
    async def insert_signal(self, source: str, ticker: str, action: str,
                            price: float, raw: dict[str, Any]) -> None:
        await self.conn.execute(
            "INSERT INTO signals (ts, source, ticker, action, price, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), source, ticker, action, price, json.dumps(raw)),
        )
        await self.conn.commit()

    async def insert_trade(self, ticker: str, side: str, qty: float, entry: float,
                          sl: float, tp: float, status: str, strategy: str,
                          dry_run: bool) -> int:
        cur = await self.conn.execute(
            "INSERT INTO trades (ts, ticker, side, qty, entry, sl, tp, status, "
            "strategy, dry_run) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), ticker, side, qty, entry, sl, tp, status, strategy,
             int(dry_run)),
        )
        await self.conn.commit()
        return cur.lastrowid

    async def close_trade(self, trade_id: int, exit_price: float, pnl: float) -> None:
        await self.conn.execute(
            "UPDATE trades SET exit_price=?, pnl=?, status='closed' WHERE id=?",
            (exit_price, pnl, trade_id),
        )
        await self.conn.commit()

    async def journal(self, agent: str, decision: str, rationale: str = "",
                      payload: dict[str, Any] | None = None) -> None:
        await self.conn.execute(
            "INSERT INTO agent_decisions (ts, agent, decision, rationale, payload_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (time.time(), agent, decision, rationale, json.dumps(payload or {})),
        )
        await self.conn.commit()

    async def insert_params(self, version: int, params: dict[str, Any],
                           score: float, metrics: dict[str, Any]) -> None:
        await self.conn.execute(
            "INSERT INTO params_history (ts, version, params_json, score, metrics_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (time.time(), version, json.dumps(params), score, json.dumps(metrics)),
        )
        await self.conn.commit()

    async def insert_lesson(self, market_id: str, predicted: float,
                           actual: float, lesson: str) -> None:
        await self.conn.execute(
            "INSERT INTO llm_lessons (ts, market_id, predicted, actual, lesson_text)"
            " VALUES (?, ?, ?, ?, ?)",
            (time.time(), market_id, predicted, actual, lesson),
        )
        await self.conn.commit()

    # --- reads (κυρίως για τον Optimization agent) ---------------------
    async def count_closed_trades(self) -> int:
        cur = await self.conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE status='closed'")
        row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def fetch_closed_trades(self, limit: int = 1000) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT * FROM trades WHERE status='closed' ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def fetch_failed_lessons(self, limit: int = 20) -> list[dict[str, Any]]:
        """Προβλέψεις όπου το predicted ήταν στη λάθος πλευρά του 0.5."""
        cur = await self.conn.execute(
            "SELECT * FROM llm_lessons WHERE actual IS NOT NULL "
            "AND ((predicted >= 0.5 AND actual < 0.5) OR "
            "     (predicted < 0.5 AND actual >= 0.5)) "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]
