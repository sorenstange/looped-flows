"""Sample windows, chronological splits and PyTorch datasets.

A sample is identified by its anchor bar T: the context is bars T-lookback+1..T, the trajectory covers bars T+1..T+H.
An anchor is usable only if every bar in T-lookback..T+H is valid (bar T-lookback provides the first context return).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from src.config import Config, SplitConfig
from src.data import clean_ohlcv, load_ohlcv, to_utc
from src.features import FEATURE_NAMES, bar_features, standardize_windows
from src.oracle import bar_returns, make_levels, oracle_trajectory

SPLITS = ("train", "val", "test")


@dataclass
class MarketData:
    times: pd.DatetimeIndex  # (T,) bar open times
    features: np.ndarray  # (T, F) float32, see src.features.FEATURE_NAMES
    returns: np.ndarray  # (T,) float64, return of bar t from close t-1 to close t
    valid: np.ndarray  # (T,) bool

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, return_type: str) -> "MarketData":
        """Build from the output of `clean_ohlcv`."""
        features = bar_features(bars)
        returns = bar_returns(bars["close"].to_numpy(), return_type)
        valid = bars["valid"].to_numpy() & np.isfinite(features[:, 1:]).all(axis=1)
        return cls(bars.index, features, returns, valid)

    @classmethod
    def load(cls, cfg: Config) -> "MarketData":
        return cls.from_bars(clean_ohlcv(load_ohlcv(cfg.data), cfg.data.interval), cfg.oracle.return_type)


def split_ranges(times: pd.DatetimeIndex, splits: SplitConfig, lookback: int, horizon: int) -> dict[str, tuple[int, int]]:
    """Half-open bar index ranges per split; val and test start `purge` bars after their boundary date."""
    purge = splits.purge if splits.purge is not None else lookback + horizon
    val_i = int(times.searchsorted(to_utc(splits.val_start)))
    test_i = int(times.searchsorted(to_utc(splits.test_start)))
    return {"train": (0, val_i), "val": (val_i + purge, test_i), "test": (test_i + purge, len(times))}


def valid_anchors(valid: np.ndarray, lookback: int, horizon: int, lo: int, hi: int, stride: int = 1) -> np.ndarray:
    """Anchors T whose bars T-lookback..T+horizon all lie in [lo, hi) and are valid."""
    invalid_before = np.concatenate([[0], np.cumsum(~valid)])
    anchors = np.arange(lo + lookback, hi - horizon)
    clean = invalid_before[anchors + horizon + 1] == invalid_before[anchors - lookback]
    return anchors[clean][::stride]


class WindowDataset(Dataset):
    """(context, current position, oracle trajectory) samples for a fixed set of anchors.

    Index with a list of positions to fetch a whole batch in one vectorized call (see `make_loader`); an int index
    returns a single unbatched sample. The current position a_0 is drawn deterministically from
    (seed, epoch, anchor), so batches are reproducible regardless of worker count; call `set_epoch` to redraw.
    """

    def __init__(self, market: MarketData, anchors: np.ndarray, cfg: Config, seed: int = 0):
        self.market = market
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.lookback = cfg.features.lookback
        self.horizon = cfg.oracle.horizon
        self.cost = cfg.oracle.cost
        self.max_step = cfg.oracle.max_step
        self.levels = make_levels(cfg.oracle.num_levels)
        self.initial_position = cfg.oracle.initial_position
        self.standardize = [FEATURE_NAMES.index(name) for name in cfg.features.standardize]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int | list[int]) -> dict[str, torch.Tensor]:
        batched = not np.isscalar(index)
        anchors = self.anchors[np.atleast_1d(np.asarray(index, dtype=np.int64))]
        context_bars = anchors[:, None] + np.arange(1 - self.lookback, 1)
        future_bars = anchors[:, None] + np.arange(1, self.horizon + 1)

        context = standardize_windows(torch.from_numpy(self.market.features[context_bars]), self.standardize)
        returns = torch.from_numpy(self.market.returns[future_bars])
        a0 = self.initial_positions(anchors)
        target, pnl = oracle_trajectory(returns, a0, self.levels, self.cost, self.max_step)

        sample = {
            "context": context,  # (B, lookback, F) float32
            "position": a0.float(),  # (B,) a_0
            "target": target,  # (B, H) long, indices into make_levels(oracle.num_levels)
            "returns": returns.float(),  # (B, H) future bar returns
            "oracle_pnl": pnl.float(),  # (B,)
            "anchor": torch.from_numpy(anchors),  # (B,) bar index of T
        }
        return sample if batched else {key: value[0] for key, value in sample.items()}

    def initial_positions(self, anchors: np.ndarray) -> torch.Tensor:
        u = torch.from_numpy(_hash_uniform(self.seed, self.epoch, anchors))
        if self.initial_position == "uniform":
            return 2 * u - 1
        if self.initial_position == "zero":
            return torch.zeros_like(u)
        return self.levels[(u * len(self.levels)).long()]


def build_datasets(cfg: Config, market: MarketData | None = None) -> dict[str, WindowDataset]:
    market = market if market is not None else MarketData.load(cfg)
    lookback, horizon = cfg.features.lookback, cfg.oracle.horizon
    datasets = {}
    for name, (lo, hi) in split_ranges(market.times, cfg.splits, lookback, horizon).items():
        stride = cfg.splits.train_stride if name == "train" else 1
        anchors = valid_anchors(market.valid, lookback, horizon, lo, hi, stride)
        if len(anchors) == 0:
            raise ValueError(f"split {name!r} has no usable samples; check data range, split dates, lookback, horizon")
        datasets[name] = WindowDataset(market, anchors, cfg, seed=cfg.loader.seed)
    return datasets


def make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0,
                drop_last: bool = False) -> DataLoader:
    """DataLoader that hands whole index batches to the dataset instead of collating single samples."""
    base = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(base, batch_size, drop_last), batch_size=None,
                      num_workers=num_workers)


def _hash_uniform(*keys: int | np.ndarray) -> np.ndarray:
    """Counter-based uniform [0, 1) draws via chained splitmix64, one per broadcast element of `keys`."""
    h = np.zeros(np.broadcast(*[np.asarray(k) for k in keys]).shape, dtype=np.uint64)
    for key in keys:
        h = _splitmix64(h ^ np.asarray(key).astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) / 2.0**53


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))
