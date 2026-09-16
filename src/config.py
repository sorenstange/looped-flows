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
    scaling: str = "rolling"  # rolling: trailing-window scaling | window: log volume z-scored per sample window
    rolling_window: int = 2016  # bars of trailing statistics (1 week at 5m); also used by log_vol_to_cost
    clip: Optional[float] = 10.0  # clip context features to [-clip, clip]; null = no clipping


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
    seed: int = 0  # for the random policy and the model's sampling noise
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    checkpoint: Optional[str] = None  # model checkpoint to backtest alongside the policies
    device: str = "auto"  # for the model: auto | cpu | cuda
    chains: int = 256  # model backtest: parallel chains over the decision bars (1 = fully sequential)
    burn_in: int = 64  # bars at the start of every chain but the first, run for their a_0 and then discarded
    inference_samples: Optional[int] = 4  # K for model backtests (cheap dev default); null = inference.samples
    inference_flow_steps: Optional[int] = 16  # n for model backtests (cheap dev default); null = inference.flow_steps


@dataclass
class ModelConfig:
    """Denoiser backbone (paper Appendix A / TRM): shared transformer F applied in a two-state recurrence."""
    width: int = 512  # hidden size d
    heads: int = 8
    layers: int = 2  # transformer blocks in the shared network F
    mlp_width: int = 1536  # SwiGLU intermediate size
    cycles: int = 3  # recurrence cycles per denoiser call; only the last one is backpropagated
    inner_steps: int = 4  # ℓ updates per cycle (m in the paper)
    patch_size: int = 1  # context bars per context token; features.lookback must be divisible by it
    rope_base: float = 10000.0
    norm_eps: float = 1e-5


@dataclass
class FlowConfig:
    """Looped-flow training (paper Algorithm 1)."""
    steps: int = 16  # k: denoising steps per training rollout (each is one optimizer step)
    time_sampler: str = "sorted"  # sorted: k+1 uniform draws, sorted | random_start: t_0 ~ U[0,1], rest ~ U[t_0, 1]
    noise_scale: Optional[float] = None  # σ of the noise x_0 ~ N(0, σ² I); null = 1 / sqrt(num_levels)
    share_noise: bool = True  # one noise sample per rollout (paper); false = fresh noise each step (ablation)
    pseudotargets: bool = False  # interpolate towards the previous prediction instead of the target (App. D)
    pseudotarget_ramp_steps: int = 20_000  # pseudotarget probability rises linearly from 0 to 1 over these steps


@dataclass
class WandbConfig:
    enabled: bool = True  # the API key is read from WANDB_API_KEY (environment or .env); WANDB_MODE=disabled wins
    project: str = "looped-flows"
    entity: Optional[str] = None  # null = the key's default entity
    mode: str = "online"  # online | offline | disabled
    tags: List[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    method: str = "direct"  # direct: one call on an empty trajectory (baseline) | looped_flow: Algorithm 1
    max_steps: int = 100_000
    optimizer: str = "adam_atan2"  # adam_atan2 (paper) | adamw
    lr: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    lr_schedule: str = "constant"  # constant | cosine (decays to min_lr_ratio * lr at max_steps)
    min_lr_ratio: float = 0.1
    grad_clip: Optional[float] = 1.0  # max gradient norm; null = no clipping
    ema_decay: Optional[float] = 0.999  # parameter EMA used for evaluation and checkpoints; null = no EMA
    loss: str = "stablemax"  # stablemax (paper) | softmax cross-entropy
    device: str = "auto"  # auto | cpu | cuda
    precision: str = "bf16"  # bf16 (autocast on CUDA only) | fp32
    log_every: int = 100
    eval_every: int = 2000
    eval_batches: int = 20  # validation batches, evenly spaced over the val split
    sample_eval_batches: int = 4  # batches for metrics of sampled trajectories (inference settings); 0 = off
    checkpoint_every: int = 10_000
    overfit_samples: Optional[int] = None  # train on the first N train samples with fixed a_0 (sanity check)
    seed: int = 0


@dataclass
class ActConfig:
    """Confidence head q (the paper's ACT head): halting during training, optional use at inference."""
    tolerance: float = 0.1  # q target: mean |rounded predicted level - oracle level| <= tolerance
    steps: Optional[int] = None  # compare the first m trajectory steps; null = all H
    loss_weight: float = 0.5  # weight λ of the BCE loss (paper: 0.5)
    halting: bool = True  # looped flow: a sample stops contributing to later steps once q > 1/2
    exploration_prob: float = 0.1  # per sample, force a random minimum of 2..k steps before halting (paper: 0.1)


@dataclass
class InferenceConfig:
    """How a trained model's sampled trajectories become the traded allocation (see src.policies.readout_allocation)."""
    samples: int = 16  # K trajectories sampled per decision
    flow_steps: int = 32  # n: integration steps of the sampler (uniform grid on [0, 1])
    gamma: float = 5.0  # γ: stochasticity of the sampler; 0 = deterministic Euler integration
    aggregation: str = "mean"  # mean over the K samples | best_q (highest confidence score) | q_weighted
    readout: str = "step"  # step: expected level at readout_step | prefix_mean: mean over steps 1..readout_step
    readout_step: int = 1  # 1-based trajectory step; 1 = next bar's planned position


@dataclass
class InspectConfig:
    """Qualitative inspection plots for a trained checkpoint (scripts/inspect_model.py)."""
    checkpoint: Optional[str] = None  # checkpoint to inspect; null = backtest.checkpoint
    split: str = "val"  # train | val | test (test only for the final evaluation)
    decisions: int = 2048  # decisions, evenly spaced over the split, behind the aggregate plots
    examples: int = 6  # individual decisions drawn as trajectory panels and probability heatmaps
    timeline_bars: int = 2016  # contiguous bars in the position-vs-price timeline (1 week at 5m)
    timeline_start: Optional[str] = None  # UTC start of that window; null = its largest absolute price move
    scaling_curve: bool = True  # also plot accuracy against sampler steps n and samples K (costs extra sampling)
    scaling_decisions: int = 256  # decisions used for the scaling curve (it re-samples per grid point)
    samples: Optional[int] = None  # K for the main plots; null = inference.samples
    flow_steps: Optional[int] = None  # n for the main plots; null = inference.flow_steps
    device: str = "auto"  # auto | cpu | cuda
    seed: int = 0


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
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    act: ActConfig = field(default_factory=ActConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    inspect: InspectConfig = field(default_factory=InspectConfig)
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


def config_to_dict(cfg: Config) -> dict:
    """Plain-dict form of a config, e.g. for embedding in checkpoints."""
    return OmegaConf.to_container(OmegaConf.structured(cfg))


def config_from_dict(data: dict) -> Config:
    cfg: Config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Config), data))
    validate(cfg)
    return cfg


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
    from src.policies import POLICY_NAMES

    def check(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(f"invalid config: {message}")

    interval_to_timedelta(cfg.data.interval)
    check(cfg.data.market in ("futures", "spot"), f"data.market={cfg.data.market!r}")
    check(cfg.features.lookback > 1, "features.lookback must be > 1")
    check(cfg.features.scaling in ("rolling", "window"), f"features.scaling={cfg.features.scaling!r}")
    check(cfg.features.rolling_window >= 2, "features.rolling_window must be >= 2")
    check(cfg.features.clip is None or cfg.features.clip > 0, "features.clip must be null or > 0")
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
    check(cfg.backtest.device in ("auto", "cpu", "cuda"), f"backtest.device={cfg.backtest.device!r}")
    check(cfg.backtest.chains >= 1 and cfg.backtest.burn_in >= 0, "backtest.chains must be >= 1, burn_in >= 0")
    check(all(v is None or v >= 1 for v in (cfg.backtest.inference_samples, cfg.backtest.inference_flow_steps)),
          "backtest.inference_samples / inference_flow_steps must be null or >= 1")
    check(baselines.momentum_window > 0, "backtest.baselines.momentum_window must be > 0")
    model = cfg.model
    check(min(model.width, model.heads, model.layers, model.mlp_width, model.cycles, model.inner_steps,
              model.patch_size) >= 1, "model sizes and counts must be >= 1")
    check(model.width % model.heads == 0 and (model.width // model.heads) % 2 == 0,
          "model.width must be divisible by model.heads with an even head dimension (RoPE)")
    check(cfg.features.lookback % model.patch_size == 0, "features.lookback must be divisible by model.patch_size")
    train = cfg.train
    check(train.method in ("direct", "looped_flow"), f"train.method={train.method!r}")
    flow = cfg.flow
    check(flow.steps >= 1, "flow.steps must be >= 1")
    check(flow.time_sampler in ("sorted", "random_start"), f"flow.time_sampler={flow.time_sampler!r}")
    check(flow.noise_scale is None or flow.noise_scale > 0, "flow.noise_scale must be null or > 0")
    check(flow.pseudotarget_ramp_steps >= 1, "flow.pseudotarget_ramp_steps must be >= 1")
    check(0 <= cfg.act.exploration_prob <= 1, "act.exploration_prob must lie in [0, 1]")
    check(cfg.wandb.mode in ("online", "offline", "disabled"), f"wandb.mode={cfg.wandb.mode!r}")
    check(train.optimizer in ("adam_atan2", "adamw"), f"train.optimizer={train.optimizer!r}")
    check(train.lr_schedule in ("constant", "cosine"), f"train.lr_schedule={train.lr_schedule!r}")
    check(train.loss in ("stablemax", "softmax"), f"train.loss={train.loss!r}")
    check(train.device in ("auto", "cpu", "cuda"), f"train.device={train.device!r}")
    check(train.precision in ("bf16", "fp32"), f"train.precision={train.precision!r}")
    check(train.max_steps >= 1 and train.lr > 0 and train.warmup_steps >= 0, "train steps / lr out of range")
    check(len(train.betas) == 2 and all(0 <= b < 1 for b in train.betas), "train.betas must be two values in [0, 1)")
    check(train.ema_decay is None or 0 < train.ema_decay < 1, "train.ema_decay must be null or in (0, 1)")
    check(train.grad_clip is None or train.grad_clip > 0, "train.grad_clip must be null or > 0")
    check(min(train.log_every, train.eval_every, train.eval_batches, train.checkpoint_every) >= 1,
          "train logging / eval / checkpoint intervals must be >= 1")
    check(train.sample_eval_batches >= 0, "train.sample_eval_batches must be >= 0")
    check(train.overfit_samples is None or train.overfit_samples >= 1, "train.overfit_samples must be null or >= 1")
    check(cfg.act.tolerance >= 0, "act.tolerance must be >= 0")
    check(cfg.act.steps is None or 1 <= cfg.act.steps <= cfg.oracle.horizon, "act.steps must lie in 1..oracle.horizon")
    check(cfg.act.loss_weight >= 0, "act.loss_weight must be >= 0")
    inference = cfg.inference
    check(inference.samples >= 1, "inference.samples must be >= 1")
    check(inference.flow_steps >= 1 and inference.gamma >= 0, "inference.flow_steps must be >= 1, gamma >= 0")
    check(inference.aggregation in ("mean", "best_q", "q_weighted"), f"inference.aggregation={inference.aggregation!r}")
    check(inference.readout in ("step", "prefix_mean"), f"inference.readout={inference.readout!r}")
    check(1 <= inference.readout_step <= cfg.oracle.horizon, "inference.readout_step must lie in 1..oracle.horizon")
    inspect = cfg.inspect
    check(inspect.split in ("train", "val", "test"), f"inspect.split={inspect.split!r}")
    check(inspect.device in ("auto", "cpu", "cuda"), f"inspect.device={inspect.device!r}")
    check(min(inspect.decisions, inspect.examples, inspect.timeline_bars, inspect.scaling_decisions) >= 1,
          "inspect.decisions / examples / timeline_bars / scaling_decisions must be >= 1")
    check(all(v is None or v >= 1 for v in (inspect.samples, inspect.flow_steps)),
          "inspect.samples / flow_steps must be null or >= 1")
    sweep = cfg.oracle_sweep
    check(all(step is None or step >= spacing - 1e-9 for step in sweep.max_steps),
          f"oracle_sweep.max_steps must be null or >= the level spacing {spacing:g}")
    check(len(sweep.horizons) > 0 and all(h > 0 for h in sweep.horizons), "oracle_sweep.horizons must be > 0")
    check(len(sweep.costs) > 0 and all(c >= 0 for c in sweep.costs), "oracle_sweep.costs must be >= 0")
