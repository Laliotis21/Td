"""
binance_check — ασφαλής (read-only) έλεγχος του Binance testnet key.

ΔΕΝ στέλνει καμία εντολή. Απλώς:
  1) Διαβάζει BINANCE_API_KEY/SECRET από το .env
  2) Δοκιμάζει FUTURES testnet (testnet.binancefuture.com) -> balance
  3) Δοκιμάζει SPOT testnet (testnet.binance.vision) -> balance
και αναφέρει ποιο δουλεύει + το υπόλοιπο, ώστε να ξέρεις τι key έχεις.

ΣΗΜΕΙΩΣΗ: από cloud περιβάλλον το SPOT testnet μπορεί να επιστρέφει 451
(geo-block)· από το δικό σου μηχάνημα συνήθως δουλεύει.

Χρήση:
  cp .env.example .env   # βάλε BINANCE_API_KEY / BINANCE_API_SECRET
  python binance_check.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    if not key or not secret:
        print("❌ Λείπουν BINANCE_API_KEY / BINANCE_API_SECRET στο .env")
        return
    try:
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
    except ImportError:
        print("❌ python-binance δεν είναι εγκατεστημένο: pip install python-binance")
        return

    client = Client(key, secret, testnet=True)
    print(f"key: ...{key[-6:]}  (testnet)\n")

    # --- FUTURES testnet (testnet.binancefuture.com) ---
    print("FUTURES testnet (testnet.binancefuture.com):")
    try:
        acct = client.futures_account()
        bal = float(acct["totalWalletBalance"])
        avail = float(acct["availableBalance"])
        print(f"  ✅ AUTH OK — wallet {bal:.2f} USDT, available {avail:.2f} USDT")
        print("  -> Το key σου είναι FUTURES testnet. Μπορεί να tradeαρει ΚΑΙ από εδώ.")
    except BinanceAPIException as e:
        print(f"  ❌ {e.status_code} {e.message}")
    except Exception as e:  # noqa: BLE001 (network/451 κ.λπ.)
        print(f"  ❌ {type(e).__name__}: {e}")

    # --- SPOT testnet (testnet.binance.vision) ---
    print("\nSPOT testnet (testnet.binance.vision):")
    try:
        acct = client.get_account()
        usdt = next((b for b in acct["balances"] if b["asset"] == "USDT"), None)
        free = float(usdt["free"]) if usdt else 0.0
        print(f"  ✅ AUTH OK — {free:.2f} USDT free")
        print("  -> Το key σου είναι SPOT testnet.")
    except BinanceAPIException as e:
        print(f"  ❌ {e.status_code} {e.message}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ {type(e).__name__}: {e}  "
              "(451 από cloud = geo-block· δοκίμασε από το μηχάνημά σου)")

    print("\nΕπόμενο: αν AUTH OK -> στο .env βάλε DRY_RUN=false, BINANCE_TESTNET=true "
          "και τρέξε το bot.")


if __name__ == "__main__":
    main()
