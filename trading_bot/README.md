# Multi-Agent Trading Bot (TradingView · Binance · Polymarket)

Σύστημα ανεξάρτητων, ασύγχρονων (asyncio) agents για algorithmic trading, με
ενσωματωμένο μηχανισμό **Συνεχούς Επανεκπαίδευσης & Αυτο-Βελτίωσης**
(Optimization & ML Retraining loop).

## Agents

| Agent | Ρόλος |
|-------|-------|
| **TradingViewAgent** | FastAPI webhook server· δέχεται/επικυρώνει alerts (HMAC + schema) και τα προωθεί ως `Signal`. |
| **PolymarketAgent** | Παρακολουθεί markets μέσω CLOB· αναλύει πιθανότητες/arbitrage και αποφασίζει με **Claude** (`claude-opus-4-8`, adaptive thinking, structured output). |
| **ExecutionAgent** | Συνδέεται με Binance (`python-binance`)· position sizing 1% risk, αυτόματο SL/TP, έλεγχος margin, retry/backoff σε rate limits. |
| **Orchestrator** | Κεντρικός εγκέφαλος· risk gate, σύνθεση εντολών, **hot-reload** παραμέτρων. |
| **OptimizationAgent** | Διαβάζει το journal + OHLCV, υπολογίζει metrics (Sharpe/Win Rate/Profit Factor/Max DD), τρέχει `scipy.optimize`, κάνει prompt-tuning στον LLM και deployαρίζει νέο `config.json` με hot-reload. |

## Αρχιτεκτονική επικοινωνίας

Όλοι οι agents επικοινωνούν μέσω ενός in-process **async message bus** (pub/sub):

```
TradingView ─┐
             ├─► [signal] ─► Orchestrator ─► [order] ─► Execution ─► [result]
Polymarket ──┘                   ▲                                       │
                                 │                                       ▼
            Optimization ─► [config.reload] (hot-reload)            Journal/DB
```

## Εγκατάσταση & εκτέλεση

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # συμπλήρωσε keys· DRY_RUN=true για ασφάλεια
python main.py                # ξεκινά όλους τους agents
```

### Δοκιμή webhook (dry-run)

```bash
# υπόγραψε το body με το TV_WEBHOOK_SECRET (HMAC-SHA256)
BODY='{"ticker":"BTCUSDT","action":"buy","price":65000}'
SIG=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$TV_WEBHOOK_SECRET" | awk '{print $2}')
curl -X POST localhost:8000/webhook -H "X-Signature: $SIG" -d "$BODY"
```

Θα δεις στο `trading.log` το σήμα, την έγκριση του Orchestrator (με position size
1% & SL/TP) και το προσομοιωμένο order.

### Ένας γύρος optimization (standalone)

```bash
python -m agents.optimization_agent --once
```

Τρέχει `scipy.optimize`, ενημερώνει το `config.json` (version+1), γράφει στο
`params_history` και (όταν τρέχει το bot) στέλνει σήμα hot-reload.

## Ασφάλεια / Risk

- **DRY_RUN=true** by default → καμία πραγματική εντολή· όλα καταγράφονται.
- Position sizing: `qty = (equity × risk_per_trade) / stop_distance` (1% risk).
- Αυστηρά SL/TP, έλεγχος margin, όρια max open positions & daily drawdown.
- API keys/secrets **μόνο** μέσω `.env` (ποτέ στον κώδικα).

## Hot-reload χωρίς χαμένα webhooks

Το config κρατιέται ως immutable dict πίσω από `asyncio.Lock`. Ο Optimization
agent γράφει νέο `config.json` ατομικά (temp file + `os.replace`) και στέλνει
`config.reload`· ο Orchestrator κάνει atomic swap του reference. Τα εισερχόμενα
webhooks απλώς μπαίνουν στο bus (μη-μπλοκάρον), οπότε **κανένα δεν χάνεται** κατά
την επαναφόρτωση.

## Δομή

```
trading_bot/
├── main.py                     # entrypoint
├── config.json                 # live, hot-reloadable παράμετροι
├── core/                       # bus, messages, config_manager, database, journal
├── agents/                     # 5 agents
└── strategies/                 # metrics, backtest, objective (scipy)
```
