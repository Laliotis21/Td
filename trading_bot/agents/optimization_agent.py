"""
OptimizationAgent — ο μηχανισμός Συνεχούς Επανεκπαίδευσης & Αυτο-Βελτίωσης.

Τρέχει περιοδικά (κάθε OPTIMIZE_EVERY_HOURS) ή μετά από OPTIMIZE_EVERY_TRADES
κλεισμένα trades. Σε κάθε γύρο:

  1. DATA INGESTION   : closed trades (SQLite) + ΠΡΑΓΜΑΤΙΚΑ OHLCV (public
                        Coinbase/CryptoCompare, στο live timeframe) μέσω to_thread.
  2. EVALUATION       : υπολογίζει Sharpe / Win Rate / Profit Factor / Max DD
                        ανά στρατηγική (strategies/metrics.py).
  3. QUANT OPTIMIZATION: walk-forward (train/test split) + fee-aware objective
                        (net return μετά fees) πάνω στη ΣΩΣΤΗ στρατηγική (config
                        mode), με scipy differential_evolution + Nelder-Mead. Τα
                        νέα params γίνονται deploy ΜΟΝΟ αν περάσουν out-of-sample.
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
from datetime import datetime, timedelta, timezone

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
    async def _load_ohlcv(self, symbol: str | None = None) -> np.ndarray:
        """
        Πραγματικά OHLCV από public πηγή (Coinbase/CryptoCompare — δεν είναι
        geo-blocked, σε αντίθεση με το Binance που δίνει 451 εδώ), στο ΙΔΙΟ
        timeframe που κάνει trade ο bot (LIVE_INTERVAL). Fallback σε synthetic
        μόνο αν αποτύχει το δίκτυο, ώστε ο optimizer να μη μένει ποτέ.
        """
        symbol = symbol or os.getenv("OPTIMIZE_SYMBOL", "BTC-USD")
        interval = os.getenv("LIVE_INTERVAL", "1d")
        lookback_days = {"1m": 5, "5m": 20, "15m": 40, "1h": 120,
                         "6h": 400, "1d": 900}.get(interval, 365)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        try:
            from strategies.data_feed import fetch
            arr, source = await asyncio.to_thread(
                fetch, symbol, start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"), interval)
            if arr.shape[0] >= 60:
                self.log.info("optimizer ingested REAL data: %s", source)
                return arr
            self.log.warning("real OHLCV too short (%d bars); synthetic fallback",
                            arr.shape[0])
        except Exception as exc:  # noqa: BLE001
            self.log.warning("real OHLCV fetch failed (%s); synthetic fallback", exc)
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
    def _optimize_walk_forward(self, ohlcv: np.ndarray, mode: str,
                              base_strat: dict, atr_period: int, equity: float,
                              risk: float, fee: float, lev: float,
                              spread: float = 0.0, min_fee: float = 0.0,
                              trail: float = 0.0, adx_thr: float = 0.0
                              ) -> tuple[dict, dict, dict]:
        """
        Walk-forward + fee-aware optimization της **τρέχουσας** στρατηγικής.
        Optimizeάρει σε TRAIN (πρώτο 70%) και επικυρώνει σε αόρατο TEST (30%),
        ώστε να μη γίνεται overfit. Λαμβάνει υπόψη spread+min_fee (κόστη) ΚΑΙ το
        exit mode (trail/adx_thr) — ίδια λογική με το live. Blocking (to_thread).
        Επιστρέφει (params, train_summary, test_summary).
        """
        n = int(ohlcv.shape[0])
        cut = max(int(n * 0.7), 1)
        train, test = ohlcv[:cut], ohlcv[cut:]
        bounds = objective.param_bounds(mode)
        args = (train, mode, base_strat, atr_period, equity, risk, fee, lev,
                spread, min_fee, trail, adx_thr)
        # global search (GA-like) πάνω στο fee-aware net-return objective
        result = differential_evolution(
            objective.net_cost, bounds=bounds, args=args,
            maxiter=40, popsize=12, tol=1e-3, seed=42, polish=False,
        )
        # local refine γύρω από το global optimum
        refined = minimize(
            objective.net_cost, result.x, args=args, method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-2, "fatol": 1e-3},
        )
        best_x = refined.x if refined.fun <= result.fun else result.x
        params = objective.decode_mode(np.asarray(best_x), mode)
        train_sum = objective.simulate_summary(
            best_x, train, mode, base_strat, atr_period, equity, risk, fee, lev,
            spread, min_fee, trail, adx_thr)
        test_sum = objective.simulate_summary(
            best_x, test, mode, base_strat, atr_period, equity, risk, fee, lev,
            spread, min_fee, trail, adx_thr)
        return params, train_sum, test_sum

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
        await self.record("retrain.start", "beginning walk-forward optimization round")
        cfg = await self.config.get()
        strat_cfg = cfg.get("strategy", {})
        risk_cfg = cfg.get("risk", {})
        mode = strat_cfg.get("mode", "regime")
        atr_period = int(strat_cfg.get("atr_period", 14))
        equity = float(risk_cfg.get("account_equity", 10000.0))
        risk = float(risk_cfg.get("risk_per_trade", 0.01))
        lev = float(risk_cfg.get("max_leverage", 1.0))
        costs = cfg.get("costs", {})
        fee = float(costs.get("fee_rate", os.getenv("OPTIMIZE_FEE", "0.0004")))
        spread = float(costs.get("spread_bps", 0.0))
        min_fee = float(costs.get("min_fee", 0.0))
        exit_cfg = cfg.get("exit", {})
        emode = exit_cfg.get("mode", "bracket")
        trail = float(exit_cfg.get("trail_atr", 0.0)) if emode in ("trailing", "adaptive") else 0.0
        adx_thr = float(exit_cfg.get("adx_threshold", 0.0)) if emode == "adaptive" else 0.0

        # 1. ingest πραγματικά δεδομένα (live timeframe)
        ohlcv = await self._load_ohlcv()
        # 2. evaluate live performance (για το journal/audit)
        live_metrics = await self._evaluate_live()
        # 3. walk-forward + fee-aware optimization της ΣΩΣΤΗΣ στρατηγικής (-> thread)
        params, train_sum, test_sum = await asyncio.to_thread(
            self._optimize_walk_forward, ohlcv, mode, strat_cfg, atr_period,
            equity, risk, fee, lev, spread, min_fee, trail, adx_thr)
        # 4. prompt tuning
        new_prompt, lessons = await self._tune_prompt(
            cfg.get("llm", {}).get("system_prompt", ""))

        # 5. OOS gate: ΜΗΝ κάνεις deploy params που δεν γενικεύουν (anti-overfit)
        if test_sum["n_trades"] < 5 or test_sum["return_pct"] <= 0.0:
            await self.record(
                "retrain.rejected",
                f"OOS μη κερδοφόρο (test {test_sum['return_pct']:+.2f}% / "
                f"{int(test_sum['n_trades'])} trades, train "
                f"{train_sum['return_pct']:+.2f}%) — κρατώ v{self.config.version}",
                {"mode": mode, "params": params, "train": train_sum, "test": test_sum},
            )
            return self.config.version

        # 6. deploy: γράψε τα params στα ΣΩΣΤΑ κλειδιά της τρέχουσας στρατηγικής
        new_cfg = dict(cfg)
        new_cfg["strategy"] = {**strat_cfg,
                               **{k: v for k, v in params.items() if k != "rr_ratio"}}
        new_cfg["risk"] = {**risk_cfg, "rr_ratio": params["rr_ratio"]}
        new_cfg["llm"] = {
            **cfg.get("llm", {}),
            "system_prompt": new_prompt,
            "few_shot_lessons": lessons,
        }

        # 7. deploy + hot-reload
        version = await self.config.save(new_cfg)
        await self.db.insert_params(
            version, params, test_sum["return_pct"],
            {"mode": mode, "train": train_sum, "test": test_sum, "live": live_metrics})
        await self.record(
            "retrain.deployed",
            f"v{version} mode={mode} OOS={test_sum['return_pct']:+.2f}% "
            f"(train {train_sum['return_pct']:+.2f}%) params={params}",
            {"train": train_sum, "test": test_sum, "lessons": len(lessons)},
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
