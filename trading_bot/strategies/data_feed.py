"""
data_feed — λήψη πραγματικών OHLCV από **public** (no-key) πηγές που δεν είναι
geo-blocked εδώ. Το Binance επιστρέφει HTTP 451, οπότε χρησιμοποιούμε Coinbase
(πρωτεύον) με CryptoCompare ως fallback.

Επιστρέφει πάντα array (N,5) = [open, high, low, close, volume], αύξουσα χρονικά.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

_UA = {"User-Agent": "Mozilla/5.0"}
_GRAN = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}


def _get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def _to_epoch(d: str) -> int:
    """'2026-05-24' ή '2026-05-24T13:00:00Z' -> epoch seconds (UTC)."""
    d = d.strip()
    fmt = "%Y-%m-%dT%H:%M:%S" if "T" in d else "%Y-%m-%d"
    dt = datetime.strptime(d.replace("Z", ""), fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_coinbase(product: str, start: str, end: str,
                  interval: str = "1h") -> np.ndarray:
    """
    Coinbase Exchange public candles. Μέγιστο 300 candles ανά κλήση, οπότε
    σπάμε το διάστημα σε σελίδες. Candle = [time, low, high, open, close, volume].
    Κρατάμε το timestamp και ταξινομούμε αύξουσα στο τέλος (Coinbase γυρνά DESC).
    """
    gran = _GRAN[interval]
    t0, t1 = _to_epoch(start), _to_epoch(end)
    page = 300 * gran
    rows: list[tuple[float, float, float, float, float, float]] = []
    cur = t0
    while cur < t1:
        s = datetime.fromtimestamp(cur, timezone.utc).isoformat()
        e = datetime.fromtimestamp(min(cur + page, t1), timezone.utc).isoformat()
        url = (f"https://api.exchange.coinbase.com/products/{product}/candles"
               f"?granularity={gran}&start={s}&end={e}")
        raw = json.loads(_get(url))
        for c in raw:  # [time, low, high, open, close, volume]
            rows.append((c[0], c[3], c[2], c[1], c[4], c[5]))  # ts,open,high,low,close,vol
        cur += page
        time.sleep(0.25)  # ευγενικό rate-limit
    if not rows:
        return np.empty((0, 5))
    rows.sort(key=lambda r: r[0])              # αύξουσα χρονικά, dedup-friendly
    seen: set[float] = set()
    ohlcv: list[list[float]] = []
    for ts, o, h, low, c, v in rows:
        if ts in seen:
            continue
        seen.add(ts)
        ohlcv.append([o, h, low, c, v])
    return np.array(ohlcv, dtype=float)


def fetch_yahoo(symbol: str, interval: str = "1d",
               range_: str = "6mo") -> np.ndarray:
    """
    Yahoo Finance public chart API — καλύπτει **stocks, ETFs ΚΑΙ crypto** με ένα
    endpoint (π.χ. AAPL, SPY, GLD, BTC-USD). Επιστρέφει (N,5) [O,H,L,C,V].
    Φιλτράρει bars με ελλείποντα δεδομένα (None).
    """
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval={interval}&range={range_}")
    d = json.loads(_get(url))
    r = d["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    rows: list[list[float]] = []
    for o, h, lo, c, v in zip(q["open"], q["high"], q["low"], q["close"],
                              q["volume"]):
        if None in (o, h, lo, c):
            continue
        rows.append([o, h, lo, c, v or 0.0])
    return np.array(rows, dtype=float)


def fetch_cryptocompare(fsym: str, tsym: str, end: str,
                       interval: str = "1h", limit: int = 72) -> np.ndarray:
    """Fallback: CryptoCompare histohour/histoday. Επιστρέφει τα τελευταία `limit`
    candles μέχρι την ημερομηνία `end`."""
    path = "histohour" if interval == "1h" else "histoday"
    to_ts = _to_epoch(end)
    url = (f"https://min-api.cryptocompare.com/data/v2/{path}"
           f"?fsym={fsym}&tsym={tsym}&limit={limit}&toTs={to_ts}")
    data = json.loads(_get(url))["Data"]["Data"]
    rows = [[d["open"], d["high"], d["low"], d["close"], d["volumefrom"]]
            for d in data]
    return np.array(rows, dtype=float)


def fetch(symbol: str = "BTC-USD", start: str = "", end: str = "",
         interval: str = "1h") -> tuple[np.ndarray, str]:
    """
    Προσπαθεί Coinbase πρώτα, μετά CryptoCompare. Επιστρέφει (ohlcv, source_label).
    `symbol` μορφή Coinbase 'BTC-USD'· μετατρέπεται αυτόματα για CryptoCompare.
    """
    try:
        arr = fetch_coinbase(symbol, start, end, interval)
        if arr.shape[0] > 0:
            return arr, f"Coinbase {symbol} {interval} {start}->{end} ({arr.shape[0]} bars)"
    except Exception as exc:  # noqa: BLE001
        print(f"[data_feed] Coinbase failed: {exc}; trying CryptoCompare")
    fsym, tsym = (symbol.split("-") + ["USD"])[:2]
    # εκτίμηση πλήθους candles από το διάστημα
    span = max(1, (_to_epoch(end) - _to_epoch(start)) // _GRAN[interval]) if start else 72
    arr = fetch_cryptocompare(fsym, tsym, end or
                              datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                              interval, int(span))
    return arr, f"CryptoCompare {fsym}/{tsym} {interval} (~{arr.shape[0]} bars)"
