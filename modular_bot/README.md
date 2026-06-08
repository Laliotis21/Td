# Modular Trading Bot — Crypto + Stocks, ίδιο Risk Management

Σπονδυλωτό bot: **μία** κοινή λογική διαχείρισης ρίσκου, **δύο** ανταλλάξιμα
execution modules. Αλλάζεις asset class (crypto ↔ μετοχές) χωρίς να αγγίξεις
τη βασική λογική.

## Αρχιτεκτονική

```
modular_bot/
├── core/                     # ΚΟΙΝΟ, broker-agnostic
│   ├── models.py             #   TradeSetup, SizedOrder, OrderResult (κοινό λεξιλόγιο)
│   └── risk.py               #   RiskManager: 1% sizing + έλεγχος R:R≥1:3
├── execution/
│   ├── base.py               #   ExecutionModule (abstract interface — το «συμβόλαιο»)
│   ├── crypto_ccxt.py        #   CryptoExecution  (CCXT, Binance/Bybit testnet)
│   └── stocks_ib.py          #   StocksExecution  (ib_insync, IB paper)
├── bot.py                    # TradingBot (ενορχηστρωτής) + build_bot() factory
└── main.py                   # demo + εναλλαγή mode
```

**Η ροή (κοινή για όλα):**
```
TradeSetup ─► RiskManager (1% + R:R) ─► SizedOrder ─► ExecutionModule.place_bracket_order
```
Ο `TradingBot` μιλά μόνο μέσω της `ExecutionModule` διεπαφής — δεν ξέρει αν από
κάτω είναι CCXT ή ib_insync. Εκεί κρύβεται η σπονδυλωτότητα.

## Εγκατάσταση
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Πώς εναλλάσσεις Crypto ↔ Μετοχές

Μία μεταβλητή — τίποτα άλλο:

```bash
# CRYPTO (CCXT, Binance/Bybit testnet)
export CRYPTO_API_KEY=...   CRYPTO_API_SECRET=...
BOT_MODE=crypto python main.py

# STOCKS (Interactive Brokers paper· πρώτα άνοιξε TWS/IB Gateway σε Paper mode)
BOT_MODE=stocks python main.py
```

Προγραμματιστικά, το ίδιο πράγμα γίνεται με το factory:
```python
from bot import build_bot
from core.models import TradeSetup

bot = build_bot("crypto", testnet=True)          # ή "stocks", port=7497
# bot = build_bot("stocks", port=7497)

setup = TradeSetup("BTC/USDT", "buy", entry=60000, stop_loss=58000, take_profit=66000)
bot.start()
print(bot.execute_signal(setup))   # ίδια κλήση, ανεξάρτητα asset
bot.stop()
```

## Προαπαιτούμενα ανά mode
| Mode | Βιβλιοθήκη | Σύνδεση | Σημείωση |
|------|-----------|---------|----------|
| `crypto` | `ccxt` | API key/secret (testnet) | `set_sandbox_mode(True)` → testnet endpoints |
| `stocks` | `ib_insync` | TWS/IB Gateway **paper** στο :7497 | ποσότητα → ακέραιες μετοχές |

## Error handling
- **Risk gate:** R:R < 1:3 ή κακή γεωμετρία → `RiskRejected` (καμία εντολή δεν στέλνεται).
- **Crypto:** `InsufficientFunds` (ρευστότητα/υπόλοιπο), `NetworkError`, `BaseError`.
- **Stocks:** αποτυχία σύνδεσης (TWS κλειστό), `qty < 1` μετοχή, σφάλματα placement.
- Όλα επιστρέφουν `OrderResult(ok=False, error=...)` αντί να σκάνε το πρόγραμμα.

## Επέκταση
Νέο asset class (π.χ. forex); Φτιάχνεις μια κλάση που κληρονομεί `ExecutionModule`,
υλοποιεί `connect / get_capital / place_bracket_order / disconnect`, και την προσθέτεις
στο `build_bot()`. Το core risk μένει αναλλοίωτο.
