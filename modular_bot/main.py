"""
main.py — demo entrypoint. Δείχνει ΠΩΣ εναλλάσσεις Crypto/Stocks: μόνο μέσω της
μεταβλητής BOT_MODE. Το ΙΔΙΟ risk management τρέχει και στα δύο.

    BOT_MODE=crypto python main.py      # CCXT (Binance/Bybit testnet)
    BOT_MODE=stocks python main.py      # ib_insync (IB paper, θέλει TWS/Gateway up)
"""
from __future__ import annotations

import logging
import os

from bot import build_bot
from core.models import TradeSetup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-8s | %(message)s",
)


def make_setup(mode: str) -> TradeSetup:
    """Ίδια στρατηγική, διαφορετικό asset — και τα δύο με R:R = 1:3."""
    if mode == "crypto":
        return TradeSetup(symbol="BTC/USDT", side="buy",
                         entry=60000, stop_loss=58000, take_profit=66000)
    return TradeSetup(symbol="AAPL", side="buy",
                     entry=100, stop_loss=96, take_profit=112)


def main() -> None:
    mode = os.getenv("BOT_MODE", "crypto").lower()

    if mode == "crypto":
        bot = build_bot("crypto", risk_per_trade=0.01, min_rr=3.0,
                       exchange_id=os.getenv("EXCHANGE_ID", "binance"), testnet=True)
    else:
        bot = build_bot("stocks", risk_per_trade=0.01, min_rr=3.0,
                       port=int(os.getenv("IB_PORT", "7497")))

    setup = make_setup(mode)
    try:
        bot.start()
        result = bot.execute_signal(setup)
        if result.ok:
            print(f"✓ Bracket order [{result.broker}] {result.symbol} "
                  f"qty={result.quantity} | entry#{result.entry_id} "
                  f"SL#{result.stop_id} TP#{result.take_id}")
        else:
            print(f"✗ Δεν εκτελέστηκε [{result.broker}]: {result.error}")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
