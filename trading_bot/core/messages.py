"""
Τυποποιημένα μηνύματα που ανταλλάσσουν οι agents μέσω του message bus.

Όλη η μεταξύ-agents επικοινωνία γίνεται με αυτά τα immutable dataclasses ώστε
κάθε agent να ξέρει ακριβώς τι σχήμα δεδομένων λαμβάνει/στέλνει.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

# --- Topics στο bus (κανάλια pub/sub) ---------------------------------
TOPIC_SIGNAL = "signal"          # TradingView/Polymarket -> Orchestrator
TOPIC_ORDER = "order"            # Orchestrator -> Execution
TOPIC_RESULT = "result"          # Execution -> Journal/loop
TOPIC_CONFIG_RELOAD = "config.reload"  # Optimization -> Orchestrator/Polymarket

Action = Literal["buy", "sell", "close", "yes", "no"]


def _now() -> float:
    return time.time()


def _id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Signal:
    """Ένα trading σήμα από έναν source agent (TradingView ή Polymarket)."""
    source: str                     # "tradingview" | "polymarket"
    ticker: str                     # π.χ. "BTCUSDT" ή market id
    action: Action
    price: float
    confidence: float = 1.0         # 0..1 (ο Polymarket agent βάζει το edge του)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_id)
    ts: float = field(default_factory=_now)


@dataclass(frozen=True)
class OrderIntent:
    """Εντολή που εγκρίθηκε από τον Orchestrator και πάει στο Execution."""
    signal_id: str
    ticker: str
    side: Action
    qty: float
    entry: float
    stop_loss: float
    take_profit: float
    strategy: str = "default"
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_id)
    ts: float = field(default_factory=_now)


@dataclass(frozen=True)
class ExecutionResult:
    """Αποτέλεσμα εκτέλεσης (πραγματικό ή προσομοιωμένο)."""
    order_id: str
    ticker: str
    side: Action
    qty: float
    entry: float
    stop_loss: float
    take_profit: float
    status: str                     # "filled" | "rejected" | "simulated" | "error"
    dry_run: bool
    reason: str = ""
    ts: float = field(default_factory=_now)


@dataclass(frozen=True)
class ConfigReload:
    """Ειδοποίηση ότι νέο config.json deployαρίστηκε από τον Optimization agent."""
    version: int
    ts: float = field(default_factory=_now)
