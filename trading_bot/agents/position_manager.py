"""
PositionManagerAgent — κλείνει ανοιχτές θέσεις και καταγράφει realized PnL, με
**adaptive exit**: σε τάση «αφήνει τα κέρδη να τρέξουν» (trailing stop), σε πλάγια
αγορά παίρνει γρήγορο κέρδος (fixed take-profit).

  ExecutionAgent ──[result: open]──► PositionManager
       │  (στο άνοιγμα: τραβά candles -> ATR + ADX -> αποφασίζει exit mode)
       │  poll τιμή ανά PM_POLL:
       │     • trailing:  stop ακολουθεί την τιμή· κλείνει όταν γυρίσει
       │     • bracket :  κλείνει σε σταθερό SL/TP
       ▼
  close_trade(pnl) ──[position.closed]──► Orchestrator + Journal/SQLite

Exit mode από το config `exit`:
  "bracket"  -> πάντα σταθερό SL/TP
  "trailing" -> πάντα trailing (ride όλα)
  "adaptive" -> ADX>=threshold στην είσοδο ? trailing : bracket  («διαλέγει μόνος»)

- PAPER (dry-run): τιμή από public Coinbase ticker· κλείσιμο υπολογιστικά.
- LIVE futures: τιμή/κλείσιμο μέσω BinanceFutures (testnet).
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

from core.bus import AsyncMessageBus
from core.config_manager import ConfigManager
from core.database import Database
from core.journal import Journal
from core.messages import (
    TOPIC_POSITION_CLOSED,
    TOPIC_RESULT,
    ExecutionResult,
    PositionClosed,
)

from .base import BaseAgent


class PositionManagerAgent(BaseAgent):
    name = "positions"

    def __init__(self, bus: AsyncMessageBus, db: Database, journal: Journal,
                config: ConfigManager | None = None,
                poll_interval: float | None = None) -> None:
        super().__init__(bus, db, journal)
        self.config = config
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.market = os.getenv("BINANCE_MARKET", "spot").lower()
        self.fee_rate = float(os.getenv("POSITION_FEE", "0.0004"))
        self.poll = poll_interval if poll_interval is not None \
            else float(os.getenv("PM_POLL", "10"))
        self.interval = os.getenv("LIVE_INTERVAL", "1d")
        # exit mode (διαβάζεται στο run() από το config)
        self.exit_mode = "bracket"
        self.trail_atr = 0.0
        self.adx_threshold = 0.0
        self._open: dict[int, dict] = {}     # trade_id -> position dict
        self._realized = 0.0
        self._client = self._init_live_client()

    def _init_live_client(self):
        if self.dry_run or self.market != "futures":
            return None
        key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
        if not (key and secret):
            return None
        from core.binance_futures import BinanceFutures
        return BinanceFutures(
            key, secret, testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true")

    # --- symbol / price ------------------------------------------------
    def _coinbase_symbol(self, ticker: str) -> str:
        """'BTCUSDT'/'BTC/USD' -> 'BTC-USD' (Coinbase product)."""
        t = ticker.upper().replace("-", "").replace("/", "")
        for q in ("USDT", "USDC", "USD"):
            if t.endswith(q):
                return f"{t[:-len(q)]}-USD"
        return ticker

    def _price(self, ticker: str) -> float | None:
        try:
            if self._client is not None:
                return self._client.price(ticker.upper().replace("-", ""))
            from strategies.data_feed import spot_price
            return spot_price(self._coinbase_symbol(ticker))
        except Exception as exc:  # noqa: BLE001
            self.log.warning("price fetch failed for %s: %s", ticker, exc)
            return None

    # --- exit-mode decision στο άνοιγμα -------------------------------
    def _decide_exit(self, ticker: str) -> tuple[bool, float]:
        """
        Επιστρέφει (use_trail, trail_dist). Τραβά πρόσφατα candles, υπολογίζει ATR
        (απόσταση trailing) και, σε adaptive, ADX (τάση -> trail, πλάγια -> bracket).
        Σε αποτυχία/bracket -> (False, 0.0).
        """
        if self.exit_mode == "bracket" or self.trail_atr <= 0.0:
            return False, 0.0
        try:
            import numpy as np
            from strategies.backtest import _atr
            from strategies.data_feed import fetch
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=120)
            ohlcv, _ = fetch(self._coinbase_symbol(ticker),
                            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                            self.interval)
            if ohlcv.shape[0] < 30:
                return False, 0.0
            atr = _atr(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], 14)
            trail_dist = self.trail_atr * float(atr[-1])
            if self.exit_mode == "adaptive":
                from strategies.signals import adx as _adx
                a = _adx(ohlcv, 14)
                trending = bool(not np.isnan(a[-1]) and a[-1] >= self.adx_threshold)
                return (trending, trail_dist if trending else 0.0)
            return True, trail_dist                       # pure trailing
        except Exception as exc:  # noqa: BLE001
            self.log.warning("exit-mode decision failed for %s: %s -> bracket",
                            ticker, exc)
            return False, 0.0

    # --- lifecycle -----------------------------------------------------
    async def run(self) -> None:
        if self.config is not None:
            ex = (await self.config.get()).get("exit", {})
            self.exit_mode = ex.get("mode", "bracket")
            self.trail_atr = float(ex.get("trail_atr", 0.0))
            self.adx_threshold = float(ex.get("adx_threshold", 0.0))
        results_q = self.bus.subscribe(TOPIC_RESULT)
        self.log.info("position manager ready (dry_run=%s, market=%s, poll=%ss, "
                      "exit=%s trail=%sxATR ADX>=%s)", self.dry_run, self.market,
                      self.poll, self.exit_mode, self.trail_atr, self.adx_threshold)
        await asyncio.gather(self._intake(results_q), self._monitor())

    async def _intake(self, q: asyncio.Queue) -> None:
        """Καταγράφει νέες ανοιχτές θέσεις + αποφασίζει exit mode."""
        while True:
            res: ExecutionResult = await q.get()
            if res.status in {"filled", "simulated"} and res.trade_id > 0:
                use_trail, trail_dist = await asyncio.to_thread(
                    self._decide_exit, res.ticker)
                self._open[res.trade_id] = {
                    "ticker": res.ticker, "side": res.side, "qty": res.qty,
                    "entry": res.entry, "stop": res.stop_loss, "tp": res.take_profit,
                    "dry_run": res.dry_run, "trail": use_trail,
                    "trail_dist": trail_dist, "hw": res.entry, "lw": res.entry,
                }
                self.log.info("tracking #%d %s %s entry=%.2f exit=%s",
                              res.trade_id, res.side, res.ticker, res.entry,
                              f"TRAILING {trail_dist:.2f}" if use_trail
                              else f"bracket SL={res.stop_loss:.2f}/TP={res.take_profit:.2f}")

    async def _monitor(self) -> None:
        """Poll loop: trailing update + έλεγχος εξόδου ανά θέση."""
        while True:
            await asyncio.sleep(self.poll)
            for tid in list(self._open):
                pos = self._open.get(tid)
                if pos is None:
                    continue
                price = await asyncio.to_thread(self._price, pos["ticker"])
                if price is None:
                    continue
                if pos["trail"]:                       # τράβα το stop υπέρ μας
                    if pos["side"] in {"buy", "yes"}:
                        pos["hw"] = max(pos["hw"], price)
                        pos["stop"] = max(pos["stop"], pos["hw"] - pos["trail_dist"])
                    else:
                        pos["lw"] = min(pos["lw"], price)
                        pos["stop"] = min(pos["stop"], pos["lw"] + pos["trail_dist"])
                hit = self._check(pos, price)
                if hit is not None:
                    await self._close(tid, pos, *hit)

    @staticmethod
    def _check(pos: dict, price: float) -> tuple[float, str] | None:
        """(exit_price, reason) αν χτυπήθηκε stop/target, αλλιώς None."""
        long = pos["side"] in {"buy", "yes"}
        stop_reason = "trail_stop" if pos["trail"] else "stop_loss"
        if long:
            if price <= pos["stop"]:
                return pos["stop"], stop_reason
            if not pos["trail"] and price >= pos["tp"]:
                return pos["tp"], "take_profit"
        else:
            if price >= pos["stop"]:
                return pos["stop"], stop_reason
            if not pos["trail"] and price <= pos["tp"]:
                return pos["tp"], "take_profit"
        return None

    async def _close(self, tid: int, pos: dict, exit_price: float,
                    reason: str) -> None:
        long = pos["side"] in {"buy", "yes"}
        qty, entry = pos["qty"], pos["entry"]
        gross = (exit_price - entry) * qty if long else (entry - exit_price) * qty
        fee = self.fee_rate * qty * (entry + exit_price)
        pnl = gross - fee
        if self._client is not None and not pos["dry_run"]:
            sym = pos["ticker"].upper().replace("-", "")
            try:
                await asyncio.to_thread(self._client.close_position, sym)
                await asyncio.to_thread(self._client.cancel_all, sym)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("live close failed #%d: %s", tid, exc)
        await self.db.close_trade(tid, exit_price, pnl)
        self._realized += pnl
        self._open.pop(tid, None)
        await self.record(
            "position.closed",
            f"{reason} {pos['side']} {pos['ticker']} entry={entry:.2f} "
            f"exit={exit_price:.2f} PnL={pnl:+.2f} (realized total {self._realized:+.2f})",
            {"trade_id": tid, "pnl": pnl, "reason": reason},
        )
        self.bus.publish(TOPIC_POSITION_CLOSED, PositionClosed(
            trade_id=tid, ticker=pos["ticker"], side=pos["side"], qty=qty,
            entry=entry, exit=exit_price, pnl=pnl, reason=reason,
            dry_run=pos["dry_run"]))
