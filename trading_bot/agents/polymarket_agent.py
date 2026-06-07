"""
PolymarketAgent — παρακολουθεί markets στο Polymarket μέσω του CLOB API και
παίρνει αποφάσεις με τη βοήθεια του Claude (AsyncAnthropic).

Ροή ανά κύκλο:
  1. Τραβά τα order books των markets (CLOB, httpx) -> implied probability YES.
  2. Στέλνει στον Claude το market + implied prob + (placeholder) news context.
  3. Ο Claude επιστρέφει structured απόφαση: true_prob, side, edge, rationale.
  4. Αν edge >= min_edge (από hot-reloaded config) -> publish Signal στο bus.

Το **system prompt διαβάζεται κάθε κύκλο από το config** (config["llm"]), οπότε
όταν ο Optimization agent κάνει prompt-tuning, ο agent το παίρνει χωρίς restart.
Όλες οι κλήσεις στον Claude είναι async και έχουν error handling για rate limits.
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx

from core.bus import AsyncMessageBus
from core.config_manager import ConfigManager
from core.database import Database
from core.journal import Journal
from core.messages import TOPIC_SIGNAL, Signal

from .base import BaseAgent

# δομημένο σχήμα απόφασης που ζητάμε από τον Claude (structured outputs)
_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "true_prob": {"type": "number"},     # εκτιμώμενη πιθανότητα YES (0..1)
        "side": {"type": "string", "enum": ["yes", "no", "skip"]},
        "edge": {"type": "number"},           # |true_prob - implied|
        "rationale": {"type": "string"},
    },
    "required": ["true_prob", "side", "edge", "rationale"],
    "additionalProperties": False,
}


class PolymarketAgent(BaseAgent):
    name = "polymarket"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal,
                config: ConfigManager, poll_interval: float = 30.0) -> None:
        super().__init__(bus, db, journal)
        self.config = config
        self.poll_interval = poll_interval
        self.clob_url = os.getenv("POLYMARKET_CLOB_URL",
                                  "https://clob.polymarket.com").rstrip("/")
        self.markets = [m for m in os.getenv("POLYMARKET_MARKETS", "").split(",")
                        if m.strip()]
        self._client: object | None = self._init_llm()

    # --- LLM client ----------------------------------------------------
    def _init_llm(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            self.log.warning("ANTHROPIC_API_KEY missing -> Polymarket LLM disabled")
            return None
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            self.log.warning("anthropic SDK not installed -> LLM disabled")
            return None
        return AsyncAnthropic(api_key=key)

    # --- CLOB polling --------------------------------------------------
    async def _fetch_orderbook(self, http: httpx.AsyncClient,
                              market_id: str) -> dict | None:
        """Τραβά το order book ενός market. Επιστρέφει implied prob YES."""
        try:
            resp = await http.get(f"{self.clob_url}/book",
                                  params={"token_id": market_id}, timeout=10.0)
            resp.raise_for_status()
            book = resp.json()
            # midpoint του best bid/ask ως implied probability
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            implied = (best_bid + best_ask) / 2
            return {"market_id": market_id, "implied_prob": implied,
                    "best_bid": best_bid, "best_ask": best_ask}
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            self.log.warning("CLOB fetch failed for %s: %s", market_id, exc)
            return None

    # --- Claude decision ----------------------------------------------
    async def _decide(self, market: dict, system_prompt: str,
                     news: str = "") -> dict | None:
        if self._client is None:
            return None
        user = (
            f"Market id: {market['market_id']}\n"
            f"Market-implied probability of YES: {market['implied_prob']:.3f}\n"
            f"Best bid/ask: {market['best_bid']:.3f}/{market['best_ask']:.3f}\n"
            f"Recent news context: {news or 'none provided'}\n\n"
            "Return your calibrated estimate and trade decision."
        )
        try:
            # Opus 4.8: adaptive thinking, high effort, structured JSON output.
            # Καθόλου temperature/budget_tokens (επιστρέφουν 400 στο 4.8).
            resp = await self._client.messages.create(  # type: ignore[attr-defined]
                model="claude-opus-4-8",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "high",
                    "format": {"type": "json_schema", "schema": _DECISION_SCHEMA},
                },
                system=system_prompt,
                messages=[{"role": "user", "content": user}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return json.loads(text)
        except Exception as exc:  # anthropic.RateLimitError/APIError κ.λπ.
            self._handle_llm_error(exc)
            return None

    def _handle_llm_error(self, exc: Exception) -> None:
        try:
            import anthropic
            if isinstance(exc, anthropic.RateLimitError):
                self.log.warning("Claude rate-limited; backing off")
                return
            if isinstance(exc, anthropic.APIError):
                self.log.warning("Claude API error: %s", exc)
                return
        except ImportError:
            pass
        self.log.warning("LLM decision failed: %s", exc)

    # --- main loop -----------------------------------------------------
    async def run(self) -> None:
        self.log.info("polymarket agent started (markets=%s)", self.markets or "—")
        async with httpx.AsyncClient() as http:
            while True:
                try:
                    await self._cycle(http)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — να μη πεθάνει ο agent
                    self.log.exception("polymarket cycle error: %s", exc)
                await asyncio.sleep(self.poll_interval)

    async def _cycle(self, http: httpx.AsyncClient) -> None:
        cfg = await self.config.get()
        pm_cfg = cfg.get("polymarket", {})
        llm_cfg = cfg.get("llm", {})
        system_prompt = self._compose_prompt(llm_cfg)
        min_edge = float(pm_cfg.get("min_edge", 0.05))

        for market_id in self.markets:
            book = await self._fetch_orderbook(http, market_id)
            if book is None:
                continue
            decision = await self._decide(book, system_prompt)
            if decision is None:
                continue

            edge = float(decision.get("edge", 0.0))
            side = decision.get("side", "skip")
            await self.record(
                "market.analyzed",
                f"{market_id} side={side} edge={edge:.3f}",
                {"book": book, "decision": decision},
            )
            if side in {"yes", "no"} and edge >= min_edge:
                sig = Signal(
                    source=self.name, ticker=market_id, action=side,  # type: ignore[arg-type]
                    price=book["implied_prob"], confidence=min(edge, 1.0),
                    meta={"rationale": decision.get("rationale", "")},
                )
                self.bus.publish(TOPIC_SIGNAL, sig)
                await self.record("signal.emitted", f"{side} {market_id}",
                                 {"signal_id": sig.id})

    def _compose_prompt(self, llm_cfg: dict) -> str:
        """Συνθέτει το system prompt + few-shot lessons (prompt-tuned από optimizer)."""
        base = llm_cfg.get("system_prompt", "")
        lessons = llm_cfg.get("few_shot_lessons", [])
        if not lessons:
            return base
        examples = "\n".join(f"- {l}" for l in lessons)
        return (f"{base}\n\nLessons learned from past mistakes "
                f"(avoid repeating these):\n{examples}")
