"""
Orchestrator — ο κεντρικός εγκέφαλος που συντονίζει τους source agents με το
Execution, εφαρμόζοντας τους γενικούς κανόνες ρίσκου.

- Subscribe στο TOPIC_SIGNAL (από TradingView & Polymarket).
- Για κάθε σήμα: έλεγχος risk gate (max open positions, daily drawdown),
  υπολογισμός position sizing 1% risk + Stop-Loss / Take-Profit, και έκδοση
  `OrderIntent` στο Execution.
- Subscribe στο TOPIC_CONFIG_RELOAD: όταν ο Optimization agent deployαρίσει νέες
  παραμέτρους, ο Orchestrator τις φορτώνει **hot** (atomic) χωρίς downtime — τα
  εισερχόμενα σήματα συνεχίζουν να εξυπηρετούνται κανονικά.

Το position sizing χρησιμοποιεί τις στρατηγικές παραμέτρους (atr_sl_mult, rr) από
το live config, οπότε αλλάζει δυναμικά μετά από κάθε retraining.
"""
from __future__ import annotations

import asyncio

from core.bus import AsyncMessageBus
from core.config_manager import ConfigManager
from core.database import Database
from core.journal import Journal
from core.messages import (
    TOPIC_CONFIG_RELOAD,
    TOPIC_ORDER,
    TOPIC_RESULT,
    TOPIC_SIGNAL,
    ConfigReload,
    ExecutionResult,
    OrderIntent,
    Signal,
)

from .base import BaseAgent


class Orchestrator(BaseAgent):
    name = "orchestrator"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal,
                config: ConfigManager) -> None:
        super().__init__(bus, db, journal)
        self.config = config
        self._open_positions = 0
        self._daily_pnl = 0.0

    async def run(self) -> None:
        sig_q = self.bus.subscribe(TOPIC_SIGNAL)
        reload_q = self.bus.subscribe(TOPIC_CONFIG_RELOAD)
        result_q = self.bus.subscribe(TOPIC_RESULT)
        mode = (await self.config.get()).get("strategy", {}).get("mode", "regime")
        self.log.info("orchestrator online (config v%s, strategy=%s)",
                      self.config.version, mode)
        await self.record("strategy.active", f"mode={mode}")
        await asyncio.gather(
            self._handle_signals(sig_q),
            self._handle_reloads(reload_q),
            self._handle_results(result_q),
        )

    # --- hot-reload ----------------------------------------------------
    async def _handle_reloads(self, q: asyncio.Queue) -> None:
        while True:
            msg: ConfigReload = await q.get()
            await self.config.load()  # atomic swap στο config manager
            mode = (await self.config.get()).get("strategy", {}).get("mode", "regime")
            await self.record("config.reloaded",
                             f"version {msg.version}, strategy={mode}")

    # --- results / drawdown tracking ----------------------------------
    async def _handle_results(self, q: asyncio.Queue) -> None:
        while True:
            res: ExecutionResult = await q.get()
            if res.status in {"filled", "simulated"}:
                self._open_positions += 1

    # --- signal -> risk gate -> order ---------------------------------
    async def _handle_signals(self, q: asyncio.Queue) -> None:
        while True:
            sig: Signal = await q.get()
            cfg = await self.config.get()
            order = self._evaluate(sig, cfg)
            if order is None:
                continue
            self.bus.publish(TOPIC_ORDER, order)

    def _evaluate(self, sig: Signal, cfg: dict) -> OrderIntent | None:
        risk = cfg.get("risk", {})
        strat = cfg.get("strategy", {})

        # --- γενικοί κανόνες ρίσκου ---
        if self._open_positions >= int(risk.get("max_open_positions", 5)):
            self._reject(sig, "max open positions reached")
            return None
        equity = float(risk.get("account_equity", 10000.0))
        if self._daily_pnl <= -abs(equity * float(risk.get("max_daily_drawdown", 0.06))):
            self._reject(sig, "daily drawdown limit hit")
            return None
        if sig.action == "close":
            return None  # close handling εκτός scope του skeleton

        # --- position sizing: 1% risk ---
        risk_per_trade = float(risk.get("risk_per_trade", 0.01))
        rr = float(risk.get("rr_ratio", 2.0))
        atr_sl_mult = float(strat.get("atr_sl_mult", 1.5))
        entry = float(sig.price)

        # προσέγγιση stop distance ως ποσοστό (atr proxy) όταν δεν έχουμε live ATR
        stop_dist = entry * 0.01 * atr_sl_mult
        if stop_dist <= 0:
            self._reject(sig, "invalid stop distance")
            return None

        long = sig.action in {"buy", "yes"}
        stop_loss = entry - stop_dist if long else entry + stop_dist
        take_profit = entry + rr * stop_dist if long else entry - rr * stop_dist
        qty = (equity * risk_per_trade) / stop_dist  # risk-based sizing

        return OrderIntent(
            signal_id=sig.id, ticker=sig.ticker, side=sig.action, qty=qty,
            entry=entry, stop_loss=stop_loss, take_profit=take_profit,
            strategy=sig.meta.get("strategy", sig.source),
            meta={"confidence": sig.confidence},
        )

    def _reject(self, sig: Signal, reason: str) -> None:
        self.log.info("signal %s rejected: %s", sig.id, reason)
        # fire-and-forget journal (μέσα σε async context δεν θέλουμε await εδώ)
        asyncio.create_task(
            self.record("signal.rejected", reason, {"signal_id": sig.id}))
