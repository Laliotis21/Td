"""
alpaca_check — ασφαλής (read-only) έλεγχος Alpaca paper keys. Δεν στέλνει εντολές.

Χρήση:
  στο .env βάλε ALPACA_API_KEY / ALPACA_API_SECRET, μετά:
  python alpaca_check.py
"""
from __future__ import annotations

import os


def main() -> None:
    # load .env χειροκίνητα
    if os.path.exists(".env"):
        for line in open(".env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        print("❌ Λείπουν ALPACA_API_KEY / ALPACA_API_SECRET στο .env")
        return
    from core.alpaca import Alpaca, AlpacaError
    api = Alpaca(key, sec, paper=True)
    try:
        a = api.account()
        print("✅ AUTH OK — Alpaca PAPER")
        print(f"   Equity        : ${float(a['equity']):,.2f}")
        print(f"   Cash          : ${float(a['cash']):,.2f}")
        print(f"   Buying power  : ${float(a['buying_power']):,.2f}")
        print(f"   Market open   : {'ΝΑΙ' if api.is_open() else 'ΟΧΙ (εκτός ωραρίου)'}")
    except AlpacaError as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    main()
