"""
binance_futures — minimal signed REST client για το Binance USD-M Futures testnet.

Γιατί όχι python-binance: ο constructor του κάνει ping στο SPOT endpoint
(testnet.binance.vision) που είναι geo-blocked (451) σε πολλά clouds, οπότε
αποτυγχάνει πριν καν φτάσει στο futures. Αυτός ο client χτυπά ΜΟΝΟ το
testnet.binancefuture.com (προσβάσιμο) με δικά μας signed requests.

Χειρίζεται και το testnet quirk -4120: STOP/STOP_MARKET εντολές απορρίπτονται
("use Algo Order API"), οπότε το take-profit μπαίνει ως LIMIT (reduceOnly) και το
stop-loss επιστρέφεται ως «bot-managed» (ο LiveTrader το κλείνει client-side).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse
import urllib.request

_LIVE = "https://fapi.binance.com"
_TESTNET = "https://testnet.binancefuture.com"


class BinanceFuturesError(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(f"[{code}] {msg}")
        self.code = code
        self.msg = msg


class BinanceFutures:
    """Σύγχρονος client· τύλιξέ τον σε asyncio.to_thread από async κώδικα."""

    STOP_UNSUPPORTED = -4120

    def __init__(self, key: str, secret: str, testnet: bool = True) -> None:
        self.key = key
        self.secret = secret.encode()
        self.base = _TESTNET if testnet else _LIVE

    # --- low-level -----------------------------------------------------
    def _server_time(self) -> int:
        return int(json.loads(self._raw_get("/fapi/v1/time"))["serverTime"])

    def _raw_get(self, path: str) -> bytes:
        req = urllib.request.Request(self.base + path,
                                    headers={"User-Agent": "tradingbot/1.0"})
        return urllib.request.urlopen(req, timeout=10).read()

    def _request(self, method: str, path: str, params: dict | None = None,
                signed: bool = False) -> dict:
        params = dict(params or {})
        if signed:
            params["timestamp"] = self._server_time()
            params["recvWindow"] = 5000
            q = urllib.parse.urlencode(params)
            sig = hmac.new(self.secret, q.encode(), hashlib.sha256).hexdigest()
            url = f"{self.base}{path}?{q}&signature={sig}"
        else:
            url = f"{self.base}{path}?{urllib.parse.urlencode(params)}" if params \
                else self.base + path
        req = urllib.request.Request(url, method=method,
                                    headers={"X-MBX-APIKEY": self.key})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=10).read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                j = json.loads(body)
                raise BinanceFuturesError(int(j.get("code", 0)), j.get("msg", body))
            except (ValueError, KeyError):
                raise BinanceFuturesError(e.code, body[:200])

    # --- public --------------------------------------------------------
    def price(self, symbol: str) -> float:
        return float(self._request("GET", "/fapi/v1/ticker/price",
                                   {"symbol": symbol})["price"])

    # --- account -------------------------------------------------------
    def available_balance(self, asset: str = "USDT") -> float:
        acct = self._request("GET", "/fapi/v2/account", signed=True)
        return float(acct["availableBalance"])

    def position(self, symbol: str) -> dict | None:
        rows = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol},
                            signed=True)
        for r in rows:
            if float(r["positionAmt"]) != 0:
                return r
        return None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._request("POST", "/fapi/v1/leverage",
                     {"symbol": symbol, "leverage": leverage}, signed=True)

    # --- orders --------------------------------------------------------
    def market_order(self, symbol: str, side: str, qty: float) -> dict:
        return self._request("POST", "/fapi/v1/order",
                            {"symbol": symbol, "side": side, "type": "MARKET",
                             "quantity": qty}, signed=True)

    def limit_reduce(self, symbol: str, side: str, qty: float, price: float) -> dict:
        """LIMIT reduce-only (για take-profit). Δουλεύει στο testnet."""
        return self._request("POST", "/fapi/v1/order",
                            {"symbol": symbol, "side": side, "type": "LIMIT",
                             "price": price, "quantity": qty, "reduceOnly": "true",
                             "timeInForce": "GTC"}, signed=True)

    def stop_market(self, symbol: str, side: str, qty: float,
                   stop_price: float) -> dict:
        """STOP_MARKET reduce-only. Μπορεί να ρίξει -4120 σε αυτό το testnet."""
        return self._request("POST", "/fapi/v1/order",
                            {"symbol": symbol, "side": side, "type": "STOP_MARKET",
                             "stopPrice": stop_price, "quantity": qty,
                             "reduceOnly": "true"}, signed=True)

    def cancel_all(self, symbol: str) -> None:
        self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol},
                     signed=True)

    def close_position(self, symbol: str) -> dict | None:
        """Market-close τυχόν ανοιχτής θέσης (reduce-only)."""
        pos = self.position(symbol)
        if pos is None:
            return None
        amt = float(pos["positionAmt"])
        side = "SELL" if amt > 0 else "BUY"
        return self._request("POST", "/fapi/v1/order",
                            {"symbol": symbol, "side": side, "type": "MARKET",
                             "quantity": abs(amt), "reduceOnly": "true"}, signed=True)
