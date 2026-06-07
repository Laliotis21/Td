"""
OptimizationAgent — ο μηχανισμός Συνεχούς Επανεκπαίδευσης & Αυτο-Βελτίωσης.

Τρέχει περιοδικά (κάθε OPTIMIZE_EVERY_HOURS) ή μετά από OPTIMIZE_EVERY_TRADES
κλεισμένα trades. Σε κάθε γύρο:

  1. DATA INGESTION   : διαβάζει closed trades (SQLite) + OHLCV (Binance, ή
                        synthetic σε dry-run) μέσω asyncio.to_thread.
  2. EVALUATION       : υπολογίζει Sharpe / Win Rate / Profit Factor / Max DD
                        ανά στρατηγική (strategies/metrics.py).
  3. QUANT OPTIMIZATION: scipy.optimize.differential_evolution (GA-like global
                        search) + local refine με `minimize` πάνω στο objective.
                        Προαιρετικό rolling-window sklearn μοντέλο για δυναμικό R:R.
  4. LLM PROMPT-TUNING : μαζεύει αποτυχημένες Polymarket προβλέψεις, παράγει
                        "lessons learned", και ζητά από τον Claude βελτιωμένο
                        system prompt (few-shot από τα ίδια τα λάθη).
  5. DEPLOYMENT        : γράφει νέο config.json (version+1) + params_history και
                        κάνει publish(TOPIC_CONFIG_RELOAD) -> hot-reload χωρίς
                        να σταματήσει το bot.

Χαμηλό latency: όλο το CPU-bound optimization τρέχει σε thread (asyncio.to_thread)
ώστε να μη μπλοκάρει το event loop / τα webhooks.
"""
from __future__ import annotations

import argparse
import asyncio
import os

import numpy as np
from scipy.optimize import differential_evolution, minimize

from core.bus import AsyncMessageBus
from core.config_manager import ConfigManager
from core.database import Database
from core.journal import Journal
from core.messages import TOPIC_CONFIG_RELOAD, ConfigReload
from strategies import metrics, objective
from strategies.backtest import synthetic_ohlcv

from .base import BaseAgent


class OptimizationAgent(BaseAgent):
    name = "optimizer"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal,
                config: ConfigManager) -> None:
        super().__init__(bus, db, journal)
        self.config = config
        self.every_trades = int(os.getenv("OPTIMIZE_EVERY_TRADES", "50"))
        self.every_hours = float(os.getenv("OPTIMIZE_EVERY_HOURS", "168"))
        self._last_trade_count = 0

    # ===================================================================
    # 1. DATA INGESTION
    # ===================================================================
    async def _load_ohlcv(self, symbol: str = "BTCUSDT") -> np.ndarray:
        """OHLCV από Binance (live) ή synthetic (dry-run / χωρίς keys)."""
        if os.getenv("DRY_RUN", "true").lower() == "true" \
                or not os.getenv("BINANCE_API_KEY"):
            return synthetic_ohlcv(500)
        try:
            from binance.client import Client
            client = Client(os.getenv("BINANCE_API_KEY"),
                           os.getenv("BINANCE_API_SECRET"),
                           testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true")
            raw = await asyncio.to_thread(
                client.get_klines, symbol=symbol, interval="1h", limit=500)
            arr = np.array([[float(k[1]), float(k[2]), float(k[3]),
                             float(k[4]), float(k[5])] for k in raw])
            return arr
        except Exception as exc:  # noqa: BLE001
            self.log.warning("OHLCV fetch failed (%s); using synthetic", exc)
            return synthetic_ohlcv(500)

    # ===================================================================
    # 2. EVALUATION
    # ===================================================================
    async def _evaluate_live(self) -> dict[str, float]:
        """Metrics από τα πραγματικά κλεισμένα trades του journal."""
        trades = await self.db.fetch_closed_trades(limit=1000)
        pnls = np.array([float(t["pnl"]) for t in trades
                         if t.get("pnl") is not None], dtype=float)
        if pnls.size == 0:
            return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                    "max_drawdown": 0.0, "n_trades": 0.0, "total_pnl": 0.0}
        return metrics.summarize(pnls)

    # ===================================================================
    # 3. QUANT OPTIMIZATION (scipy)
    # ===================================================================
    def _optimize_sync(self, ohlcv: np.ndarray,
                      atr_period: int) -> tuple[dict, float]:
        """Blocking optimization — καλείται μέσα σε asyncio.to_thread."""
        # global search (GA-like)
        result = differential_evolution(
            objective.objective, bounds=objective.PARAM_BOUNDS,
            args=(ohlcv, atr_period), maxiter=40, popsize=12, tol=1e-3,
            seed=42, polish=False,
        )
        # local refine γύρω από το global optimum
        refined = minimize(
            objective.objective, result.x, args=(ohlcv, atr_period),
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-2, "fatol": 1e-3},
        )
        best_x = refined.x if refined.fun <= result.fun else result.x
        best_score = -float(min(refined.fun, result.fun))
        return objective.decode(np.asarray(best_x)), best_score

    def _rolling_rr_model(self, ohlcv: np.ndarray) -> float | None:
        """
        Προαιρετικό rolling-window sklearn μοντέλο: μαθαίνει σχέση
        volatility -> βέλτιστο R:R και προτείνει δυναμικό rr. Επιστρέφει None
        αν δεν υπάρχουν αρκετά δεδομένα ή λείπει το sklearn.
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            return None
        close = ohlcv[:, 3]
        if close.size < 60:
            return None
        # feature: rolling volatility· target: forward return magnitude (proxy R:R)
        rets = np.diff(np.log(close))
        win = 20
        X, y = [], []
        for i in range(win, rets.size - 1):
            vol = rets[i - win:i].std()
            fwd = abs(rets[i + 1])
            X.append([vol])
            y.append(fwd)
        if len(X) < 30:
            return None
        model = GradientBoostingRegressor(n_estimators=50, max_depth=2)
        model.fit(np.array(X), np.array(y))
        cur_vol = rets[-win:].std()
        pred = float(model.predict([[cur_vol]])[0])
        # map σε εύλογο R:R εύρος [1.0, 4.0]
        return float(np.clip(1.0 + pred * 100.0, 1.0, 4.0))

    # ===================================================================
    # 4. LLM PROMPT-TUNING (Claude)
    # ===================================================================
    async def _tune_prompt(self, base_prompt: str) -> tuple[str, list[str]]:
        """
        Διαβάζει αποτυχημένες προβλέψεις, φτιάχνει lessons learned, και ζητά από
        τον Claude βελτιωμένο system prompt. Επιστρέφει (new_prompt, lessons).
        Αν δεν υπάρχουν λάθη ή λείπει το API key, επιστρέφει το base αμετάβλητο.
        """
        failures = await self.db.fetch_failed_lessons(limit=20)
        if not failures:
            return base_prompt, []

        lessons = [
            f"Market {f['market_id']}: predicted YES={f['predicted']:.2f} but "
            f"actual={f['actual']:.0f}. {f.get('lesson_text') or ''}".strip()
            for f in failures
        ]

        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return base_prompt, lessons  # κρατάμε τα lessons ως few-shot, χωρίς LLM rewrite
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=key)
            joined = "\n".join(f"- {l}" for l in lessons)
            resp = await client.messages.create(
                model="claude-opus-4-8",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                system=("You are a prompt engineer improving a Polymarket trading "
                        "analyst's system prompt. Given the current prompt and a list "
                        "of past prediction mistakes, rewrite the system prompt so the "
                        "analyst avoids those failure modes. Return ONLY the improved "
                        "system prompt text."),
                messages=[{"role": "user", "content":
                           f"Current prompt:\n{base_prompt}\n\nMistakes:\n{joined}"}],
            )
            new_prompt = next((b.text for b in resp.content if b.type == "text"),
                             base_prompt).strip()
            return (new_prompt or base_prompt), lessons
        except Exception as exc:  # noqa: BLE001
            self.log.warning("prompt tuning via LLM failed: %s", exc)
            return base_prompt, lessons

    # ===================================================================
    # 5. ONE FULL RETRAINING ROUND
    # ===================================================================
    async def optimize_once(self) -> int:
        await self.record("retrain.start", "beginning optimization round")
        cfg = await self.config.get()
        atr_period = int(cfg.get("strategy", {}).get("atr_period", 14))

        # 1. ingest
        ohlcv = await self._load_ohlcv()
        # 2. evaluate live performance (για το journal/audit)
        live_metrics = await self._evaluate_live()
        # 3. quant optimization (CPU-bound -> thread)
        best_params, best_score = await asyncio.to_thread(
            self._optimize_sync, ohlcv, atr_period)
        dyn_rr = await asyncio.to_thread(self._rolling_rr_model, ohlcv)
        if dyn_rr is not None:
            best_params["rr_ratio"] = dyn_rr
        # 4. prompt tuning
        new_prompt, lessons = await self._tune_prompt(
            cfg.get("llm", {}).get("system_prompt", ""))

        # συνθέτω νέο config
        new_cfg = dict(cfg)
        new_cfg["strategy"] = {
            **cfg.get("strategy", {}),
            "fast_ma": best_params["fast_ma"],
            "slow_ma": best_params["slow_ma"],
            "atr_sl_mult": best_params["atr_sl_mult"],
        }
        new_cfg["risk"] = {**cfg.get("risk", {}), "rr_ratio": best_params["rr_ratio"]}
        new_cfg["llm"] = {
            **cfg.get("llm", {}),
            "system_prompt": new_prompt,
            "few_shot_lessons": lessons,
        }

        # 5. deploy + hot-reload
        version = await self.config.save(new_cfg)
        await self.db.insert_params(version, best_params, best_score,
                                   {"live": live_metrics, "dynamic_rr": dyn_rr})
        await self.record(
            "retrain.deployed",
            f"v{version} score={best_score:.3f} params={best_params}",
            {"live_metrics": live_metrics, "lessons": len(lessons)},
        )
        self.bus.publish(TOPIC_CONFIG_RELOAD, ConfigReload(version=version))
        return version

    # ===================================================================
    # main loop — trigger σε trades ή χρόνο
    # ===================================================================
    async def run(self) -> None:
        self.log.info("optimizer running (every %s trades / %s h)",
                      self.every_trades, self.every_hours)
        self._last_trade_count = await self.db.count_closed_trades()
        while True:
            await asyncio.sleep(min(self.every_hours * 3600, 3600))
            try:
                n = await self.db.count_closed_trades()
                if n - self._last_trade_count >= self.every_trades:
                    self._last_trade_count = n
                    await self.optimize_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log.exception("optimizer error: %s", exc)


# --- standalone helper: python -m agents.optimization_agent --once -------
async def _run_once() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    from core.journal import setup_logging
    setup_logging(os.getenv("LOG_PATH", "trading.log"))

    bus = AsyncMessageBus()
    db = Database(os.getenv("DB_PATH", "trading.db"))
    await db.connect()
    config = ConfigManager(os.getenv("CONFIG_PATH", "config.json"))
    await config.load()
    journal = Journal(db)

    agent = OptimizationAgent(bus, db, journal, config)
    version = await agent.optimize_once()
    print(f"Optimization complete -> config version {version}")
    await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimization & ML Retraining Agent")
    parser.add_argument("--once", action="store_true",
                       help="τρέξε έναν γύρο optimization και έξοδος")
    args = parser.parse_args()
    if args.once:
        asyncio.run(_run_once())
    else:
        print("Use --once to run a single optimization round, "
              "or start the full bot via main.py")
