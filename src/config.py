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
    interval: str = "5m"  # Binance kline interval: <n>m, <n>h, <n>d or <n>w
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
class BaselineConfig:
    ma_fast: int = 288  # bars in the fast moving average (1 day at 5m)
    ma_slow: int = 2016  # bars in the slow moving average (1 week at 5m)
    momentum_window: int = 2016  # bars over which the momentum return is measured (1 week at 5m)


@dataclass
class BacktestConfig:
    split: str = "val"  # train | val | test (test only for the final evaluation)
    cost: Optional[float] = None  # cost per unit of turnover; null = oracle.cost
    enforce_max_step: bool = False  # clip every executed move of every policy to oracle.max_step
    clipped_baselines: bool = True  # also run the rule-based baselines with the step limit, as "<name>_clip"
    initial_position: float = 0.0
    policies: List[str] = field(default_factory=lambda: [
        "flat", "buy_hold", "random", "ma_crossover", "momentum", "oracle"])
    seed: int = 0  # for the random policy
    baselines: BaselineConfig = field(default_factory=BaselineConfig)


@dataclass
class InferenceConfig:
    """How a trained model's sampled trajectories become the traded allocation (see src.policies.readout_allocation)."""
    samples: int = 16  # K trajectories sampled per decision
    aggregation: str = "mean"  # mean over the K samples | best_q (the sample with the highest confidence score)
    readout: str = "step"  # step: expected level at readout_step | prefix_mean: mean over steps 1..readout_step
    readout_step: int = 1  # 1-based trajectory step; 1 = next bar's planned position


@dataclass
class OracleSweepConfig:
    """Grid for scripts/sweep_oracle.py: receding-horizon oracle backtests over these oracle settings."""
    max_steps: List[Optional[float]] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, None])
    horizons: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    costs: List[float] = field(default_factory=lambda: [0.0005, 0.001, 0.002])  # oracle planning costs


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    oracle_sweep: OracleSweepConfig = field(default_factory=OracleSweepConfig)
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
    from src.policies import POLICY_NAMES

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
    check(cfg.backtest.split in ("train", "val", "test"), f"backtest.split={cfg.backtest.split!r}")
    check(cfg.backtest.cost is None or cfg.backtest.cost >= 0, "backtest.cost must be >= 0")
    check(-1 <= cfg.backtest.initial_position <= 1, "backtest.initial_position must lie in [-1, 1]")
    unknown = set(cfg.backtest.policies) - set(POLICY_NAMES)
    check(not unknown, f"backtest.policies has unknown policies {sorted(unknown)}; known: {POLICY_NAMES}")
    baselines = cfg.backtest.baselines
    check(0 < baselines.ma_fast < baselines.ma_slow, "backtest.baselines needs 0 < ma_fast < ma_slow")
    check(baselines.momentum_window > 0, "backtest.baselines.momentum_window must be > 0")
    inference = cfg.inference
    check(inference.samples >= 1, "inference.samples must be >= 1")
    check(inference.aggregation in ("mean", "best_q"), f"inference.aggregation={inference.aggregation!r}")
    check(inference.readout in ("step", "prefix_mean"), f"inference.readout={inference.readout!r}")
    check(1 <= inference.readout_step <= cfg.oracle.horizon, "inference.readout_step must lie in 1..oracle.horizon")
    sweep = cfg.oracle_sweep
    check(all(step is None or step >= spacing - 1e-9 for step in sweep.max_steps),
          f"oracle_sweep.max_steps must be null or >= the level spacing {spacing:g}")
    check(len(sweep.horizons) > 0 and all(h > 0 for h in sweep.horizons), "oracle_sweep.horizons must be > 0")
    check(len(sweep.costs) > 0 and all(c >= 0 for c in sweep.costs), "oracle_sweep.costs must be >= 0")
