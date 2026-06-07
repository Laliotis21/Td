"""
TradingViewAgent — ελαφρύς FastAPI server που δέχεται alerts από TradingView.

- Εκθέτει POST /webhook που δέχεται JSON payload από TradingView alerts.
- Επικυρώνει HMAC-SHA256 υπογραφή (header X-Signature) με κοινό secret.
- Επικυρώνει σχήμα (ticker/action/price) μέσω Pydantic.
- Μετατρέπει σε `Signal` και κάνει **μη-μπλοκάρον** publish στο bus.

Ο server τρέχει μέσα στο ίδιο asyncio loop (uvicorn.Server.serve()) ώστε να
μοιράζεται bus/DB με τους υπόλοιπους agents.
"""
from __future__ import annotations

import hashlib
import hmac
import os

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, field_validator

from core.bus import AsyncMessageBus
from core.database import Database
from core.journal import Journal
from core.messages import TOPIC_SIGNAL, Signal

from .base import BaseAgent


class TVAlert(BaseModel):
    ticker: str
    action: str               # buy | sell | close
    price: float
    strategy: str = "tradingview"

    @field_validator("action")
    @classmethod
    def _valid_action(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"buy", "sell", "close"}:
            raise ValueError("action must be buy|sell|close")
        return v

    @field_validator("ticker")
    @classmethod
    def _norm_ticker(cls, v: str) -> str:
        return v.upper().strip()


class TradingViewAgent(BaseAgent):
    name = "tradingview"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal) -> None:
        super().__init__(bus, db, journal)
        self.secret = os.getenv("TV_WEBHOOK_SECRET", "").encode()
        self.host = os.getenv("WEBHOOK_HOST", "0.0.0.0")
        self.port = int(os.getenv("WEBHOOK_PORT", "8000"))
        self.app = self._build_app()

    def _verify(self, body: bytes, signature: str | None) -> bool:
        if not self.secret:
            return True  # κανένα secret ρυθμισμένο -> dev mode
        if not signature:
            return False
        expected = hmac.new(self.secret, body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="TradingView Webhook Agent")

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "agent": self.name}

        @app.post("/webhook")
        async def webhook(request: Request,
                         x_signature: str | None = Header(default=None)) -> dict:
            raw = await request.body()
            if not self._verify(raw, x_signature):
                raise HTTPException(status_code=401, detail="bad signature")
            try:
                alert = TVAlert.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=422, detail=str(exc))

            sig = Signal(
                source=self.name,
                ticker=alert.ticker,
                action=alert.action,  # type: ignore[arg-type]
                price=alert.price,
                meta={"strategy": alert.strategy},
            )
            # persist + journal + μη-μπλοκάρον publish (δεν χάνουμε webhooks)
            await self.db.insert_signal(self.name, sig.ticker, sig.action,
                                        sig.price, alert.model_dump())
            await self.record("signal.received",
                             f"{sig.action} {sig.ticker} @ {sig.price}",
                             {"signal_id": sig.id})
            self.bus.publish(TOPIC_SIGNAL, sig)
            return {"accepted": True, "signal_id": sig.id}

        return app

    async def run(self) -> None:
        config = uvicorn.Config(self.app, host=self.host, port=self.port,
                                log_level="warning", loop="asyncio")
        server = uvicorn.Server(config)
        self.log.info("webhook server listening on %s:%s", self.host, self.port)
        await server.serve()
