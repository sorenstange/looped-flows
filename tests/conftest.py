import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 2000, start: str = "2024-01-01", interval: str = "1h", seed: int = 0) -> pd.DataFrame:
    """Synthetic random-walk klines in the layout returned by `src.data.fetch_ohlcv`."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.lognormal(5, 1, n)
    index = pd.date_range(start, periods=n, freq=interval, tz="UTC", name="open_time")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "quote_volume": volume * close, "trades": rng.integers(100, 1000, n).astype(float),
        "taker_buy_volume": volume * rng.uniform(0.3, 0.7, n),
    }, index=index)


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture(autouse=True)
def no_wandb(monkeypatch):
    """Tests never log to Weights & Biases."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
