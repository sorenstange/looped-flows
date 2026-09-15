"""Download, cache and clean Binance OHLCV klines."""

import json
import time
from pathlib import Path

import pandas as pd
from binance.client import Client
from binance.enums import HistoricalKlinesType

from src.config import DataConfig

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                 "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
BAR_COLUMNS = PRICE_COLUMNS + ["volume", "quote_volume", "trades", "taker_buy_volume"]

_KLINE_TYPES = {"futures": HistoricalKlinesType.FUTURES, "spot": HistoricalKlinesType.SPOT}
_INTERVAL_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(ping=False)  # public market-data endpoints need no API key
    return _client


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    count, unit = interval[:-1], interval[-1:]
    if unit not in _INTERVAL_UNITS or not count.isdigit() or int(count) == 0:
        raise ValueError(f"unsupported interval {interval!r}; expected <n>m, <n>h, <n>d or <n>w")
    return pd.Timedelta(**{_INTERVAL_UNITS[unit]: int(count)})


def to_utc(timestamp: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def fetch_ohlcv(symbol: str, interval: str, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None,
                market: str = "futures") -> pd.DataFrame:
    """Fetch closed klines with open time in [start, end) from Binance, indexed by open time (UTC)."""
    start, end = to_utc(start), (to_utc(end) if end is not None else None)
    raw = _get_client().get_historical_klines(
        symbol,
        interval,
        start_str=int(start.timestamp() * 1000),
        end_str=int(end.timestamp() * 1000) if end is not None else None,
        klines_type=_KLINE_TYPES[market],
    )
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    df = df[df["close_time"].astype("int64") < time.time() * 1000]  # drop the still-open bar
    df.index = pd.DatetimeIndex(pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True), name="open_time")
    df = df[BAR_COLUMNS].astype("float64")
    return df[df.index < end] if end is not None else df


def cache_path(cfg: DataConfig) -> Path:
    return Path(cfg.cache_dir) / f"{cfg.market}_{cfg.symbol}_{cfg.interval}.parquet"


def load_ohlcv(cfg: DataConfig) -> pd.DataFrame:
    """Raw klines for the configured range, served from a parquet cache that is topped up from Binance on demand.

    The cache keeps everything ever fetched for (market, symbol, interval); a sidecar JSON remembers the earliest
    requested start, so a start before the listing date does not trigger a refetch on every call.
    """
    path = cache_path(cfg)
    meta_path = path.with_suffix(".json")
    start = to_utc(cfg.start)
    end = to_utc(cfg.end) if cfg.end is not None else None
    step = interval_to_timedelta(cfg.interval)

    cached = pd.read_parquet(path) if path.exists() else None
    fetched_start = to_utc(json.loads(meta_path.read_text())["fetched_start"]) if meta_path.exists() else None

    if cfg.update:
        parts = [] if cached is None else [cached]
        if cached is None or fetched_start is None:
            parts.append(fetch_ohlcv(cfg.symbol, cfg.interval, start, end, cfg.market))
            fetched_start = start
        else:
            if start < fetched_start:
                parts.append(fetch_ohlcv(cfg.symbol, cfg.interval, start, fetched_start, cfg.market))
                fetched_start = start
            next_bar = cached.index[-1] + step
            if end is None or end > next_bar:
                parts.append(fetch_ohlcv(cfg.symbol, cfg.interval, next_bar, end, cfg.market))
        non_empty = [part for part in parts if not part.empty]
        merged = pd.concat(non_empty) if non_empty else parts[0]
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path)
        meta_path.write_text(json.dumps({"fetched_start": fetched_start.isoformat()}))
        cached = merged
    elif cached is None:
        raise FileNotFoundError(f"no cached data at {path} and data.update=false")

    mask = cached.index >= start
    if end is not None:
        mask &= cached.index < end
    bars = cached[mask]
    if bars.empty:
        raise ValueError(f"no bars for {cfg.symbol} {cfg.interval} in [{cfg.start}, {cfg.end})")
    return bars


def clean_ohlcv(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Put bars on a regular time grid and flag unusable ones in a boolean `valid` column.

    Missing bars are filled flat at the previous close with zero volume. Filled and malformed bars are marked
    invalid so that any sample window touching them can be skipped.
    """
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    grid = pd.date_range(bars.index[0], bars.index[-1], freq=interval_to_timedelta(interval), name="open_time")
    out = bars.reindex(grid)
    missing = out["close"].isna()
    out["close"] = out["close"].ffill()
    for col in ("open", "high", "low"):
        out[col] = out[col].fillna(out["close"])
    for col in BAR_COLUMNS[len(PRICE_COLUMNS):]:
        out[col] = out[col].fillna(0.0)
    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    malformed = (out[PRICE_COLUMNS] <= 0).any(axis=1) | (out["high"] < body_high) | (out["low"] > body_low) \
        | (out["volume"] < 0)
    out["valid"] = ~(missing | malformed)
    return out
