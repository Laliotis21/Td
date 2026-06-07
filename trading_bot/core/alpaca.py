"""
alpaca — minimal client για Alpaca **paper trading** (US stocks/ETFs, δωρεάν).

Πλεονέκτημα έναντι Binance testnet: native **bracket orders** — entry + take-profit
+ stop-loss σε μία εντολή, διαχειρισμένα από το exchange (όχι bot-managed).

Auth: headers APCA-API-KEY-ID / APCA-API-SECRET-KEY (paper keys από alpaca.markets).
Σημ.: οι εντολές μετοχών εκτελούνται μόνο σε **ώρες αγοράς** (market orders εκτός
ωραρίου μπαίνουν σε ουρά για το άνοιγμα).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

_PAPER = "https://paper-api.alpaca.markets"
_DATA = "https://data.alpaca.markets"


class AlpacaError(Exception):
    pass


class Alpaca:
    def __init__(self, key: str, secret: str, paper: bool = True) -> None:
        self.base = _PAPER if paper else "https://api.alpaca.markets"
        self._h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                  "Content-Type": "application/json"}

    def _req(self, method: str, url: str, body: dict | None = None) -> dict | list:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._h, method=method)
        try:
            return json.loads(urllib.request.urlopen(req, timeout=15).read())
        except urllib.error.HTTPError as e:
            raise AlpacaError(f"{e.code}: {e.read().decode()[:200]}")

    # --- account / market ---------------------------------------------
    def account(self) -> dict:
        return self._req("GET", f"{self.base}/v2/account")

    def is_open(self) -> bool:
        return bool(self._req("GET", f"{self.base}/v2/clock")["is_open"])

    def price(self, symbol: str) -> float:
        r = self._req("GET", f"{_DATA}/v2/stocks/{symbol}/trades/latest")
        return float(r["trade"]["p"])

    # --- positions -----------------------------------------------------
    def positions(self) -> list:
        return self._req("GET", f"{self.base}/v2/positions")

    def close_position(self, symbol: str) -> dict:
        return self._req("DELETE", f"{self.base}/v2/positions/{symbol}")

    def cancel_all(self) -> None:
        self._req("DELETE", f"{self.base}/v2/orders")

    # --- orders --------------------------------------------------------
    def bracket_order(self, symbol: str, qty: float, side: str,
                     take_profit: float, stop_loss: float,
                     otype: str = "market") -> dict:
        """
        Entry + take-profit + stop-loss σε μία εντολή (order_class=bracket).
        side: 'buy' | 'sell'. Το exchange διαχειρίζεται SL/TP.
        """
        body = {
            "symbol": symbol, "qty": str(qty), "side": side, "type": otype,
            "time_in_force": "gtc", "order_class": "bracket",
            "take_profit": {"limit_price": round(take_profit, 2)},
            "stop_loss": {"stop_price": round(stop_loss, 2)},
        }
        return self._req("POST", f"{self.base}/v2/orders", body)
