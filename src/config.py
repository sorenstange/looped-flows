"""Typed project configuration backed by YAML files.

The dataclasses below are the schema and hold the defaults. YAML files in `configs/` override them, and may inherit
from another YAML file via a top-level `extends: <relative path>` key. Command-line `key=value` dotlist overrides are
applied last. Every run should save its fully resolved config (`save_config`) next to its outputs.

    cfg = load_config("configs/smoke.yaml", ["oracle.cost=0.001"])
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"  # Binance kline interval: <n>m, <n>h, <n>d or <n>w
    market: str = "futures"  # futures (USDT-M perpetuals) | spot
    start: str = "2019-10-01"  # UTC; skips the placeholder bars right after the BTCUSDT perp listing
    end: Optional[str] = None  # UTC, exclusive; null = up to the last closed bar
    cache_dir: str = "data/raw"
    update: bool = True  # fetch missing bars from Binance; false = use the cache only


@dataclass
class FeatureConfig:
    lookback: int = 256  # context bars per sample
    standardize: List[str] = field(default_factory=lambda: ["log_volume"])  # z-scored over each window


@dataclass
class OracleConfig:
    horizon: int = 64  # trajectory length H in bars
    cost: float = 0.0005  # proportional cost per unit of turnover
    num_levels: int = 21  # allocation levels, evenly spaced over [-1, 1] (21 -> steps of 0.1)
    max_step: Optional[float] = 0.1  # max |p_k - p_{k-1}| per bar, including the move from a_0; null = unlimited
    return_type: str = "simple"  # simple | log
    initial_position: str = "uniform"  # distribution of a_0: uniform in [-1, 1] | zero | levels


@dataclass
class SplitConfig:
    val_start: str = "2024-01-01"  # UTC
    test_start: str = "2025-01-01"  # UTC
    purge: Optional[int] = None  # bars skipped at the start of val/test; null = lookback + horizon
    train_stride: int = 1  # use every n-th train anchor (val/test always use every anchor)


@dataclass
class LoaderConfig:
    batch_size: int = 256
    num_workers: int = 0
    seed: int = 0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    output_dir: str = "outputs"


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> Config:
    """Merge schema defaults <- YAML inheritance chain <- dotlist overrides, then validate."""
    layers = _load_yaml_chain(Path(path)) if path is not None else []
    merged = OmegaConf.merge(OmegaConf.structured(Config), *layers, OmegaConf.from_dotlist(list(overrides)))
    cfg: Config = OmegaConf.to_object(merged)
    validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), path)


def parse_args(description: str | None = None, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Standard CLI for scripts: `--config file.yaml` followed by any number of `key=value` overrides."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. oracle.cost=0.001")
    return parser.parse_args(argv)


def _load_yaml_chain(path: Path, seen: tuple[Path, ...] = ()) -> list:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular `extends` in config: {path}")
    layer = OmegaConf.load(path)
    base = layer.pop("extends", None)
    chain = _load_yaml_chain(path.parent / base, seen + (path,)) if base else []
    return chain + [layer]


def validate(cfg: Config) -> None:
    from src.data import interval_to_timedelta
    from src.features import FEATURE_NAMES

    def check(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(f"invalid config: {message}")

    interval_to_timedelta(cfg.data.interval)
    check(cfg.data.market in ("futures", "spot"), f"data.market={cfg.data.market!r}")
    check(cfg.features.lookback > 1, "features.lookback must be > 1")
    unknown = set(cfg.features.standardize) - set(FEATURE_NAMES)
    check(not unknown, f"features.standardize has unknown features {sorted(unknown)}; known: {FEATURE_NAMES}")
    check(cfg.oracle.horizon > 0, "oracle.horizon must be > 0")
    check(cfg.oracle.cost >= 0, "oracle.cost must be >= 0")
    check(cfg.oracle.num_levels >= 2, "oracle.num_levels must be >= 2")
    spacing = 2 / (cfg.oracle.num_levels - 1)
    check(cfg.oracle.max_step is None or cfg.oracle.max_step >= spacing - 1e-9,
          f"oracle.max_step must be >= the level spacing {spacing:g}, otherwise positions cannot change")
    check(cfg.oracle.return_type in ("simple", "log"), f"oracle.return_type={cfg.oracle.return_type!r}")
    check(cfg.oracle.initial_position in ("uniform", "zero", "levels"),
          f"oracle.initial_position={cfg.oracle.initial_position!r}")
    check(pd.Timestamp(cfg.splits.val_start) < pd.Timestamp(cfg.splits.test_start), "splits.val_start >= test_start")
    check(cfg.splits.purge is None or cfg.splits.purge >= 0, "splits.purge must be >= 0")
    check(cfg.splits.train_stride >= 1, "splits.train_stride must be >= 1")
    check(cfg.loader.batch_size >= 1, "loader.batch_size must be >= 1")
