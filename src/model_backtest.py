"""Backtesting trained models with the engine's timing, clipping and fee conventions.

At every decision bar the model samples K trajectories (`sampling.predict`), `readout_allocations` turns them into
the traded position, and that position is the next decision's a_0.

Decisions depend on each other only through a_0, so `run_model_backtest` splits the decision bars into `chains`
consecutive chunks that run side by side in one batch. Every chain but the first starts `burn_in` bars before its
chunk, at `initial_position`, and discards those burn-in positions; inside its chunk each bar is decided exactly as in
a sequential run with the chain's own a_0 history. `chains = 1` is a fully sequential run, equal to running
`ModelPolicy` through `backtest.run_backtest`.
"""

import math
from dataclasses import replace

import numpy as np
import torch

from src.backtest import BacktestResult, History, Policy
from src.config import Config
from src.dataset import MarketData, context_windows
from src.oracle import make_levels
from src.policies import readout_allocations
from src.sampling import predict
from src.training import autocast

# Settings that change the model's inputs or outputs; a checkpoint is only valid for data built with the same values.
COMPATIBILITY_FIELDS = ["data.symbol", "data.market", "data.interval", "features.lookback", "features.scaling",
                        "features.rolling_window", "features.clip", "oracle.cost", "oracle.num_levels",
                        "oracle.horizon"]


def check_compatible(model_cfg: Config, cfg: Config) -> None:
    """Raise if the run config builds different model inputs than the checkpoint was trained on (oracle.cost enters
    the `log_vol_to_cost` feature)."""
    def get(config: Config, path: str):
        value = config
        for part in path.split("."):
            value = getattr(value, part)
        return value

    mismatches = [f"{path}: checkpoint {get(model_cfg, path)!r} vs run {get(cfg, path)!r}"
                  for path in COMPATIBILITY_FIELDS if get(model_cfg, path) != get(cfg, path)]
    if mismatches:
        raise ValueError("checkpoint does not match the backtest config:\n  " + "\n  ".join(mismatches))


class ModelDecider:
    """Batched decisions of one model: context windows -> K samples -> readout allocations."""

    def __init__(self, model: torch.nn.Module, model_cfg: Config, cfg: Config, market: MarketData):
        self.model = model
        self.market = market
        # Sampling follows the checkpoint (method, loss, noise); readout and sample counts follow the run config.
        self.cfg = replace(model_cfg, inference=cfg.inference)
        self.samples = cfg.backtest.inference_samples or cfg.inference.samples
        self.flow_steps = cfg.backtest.inference_flow_steps or cfg.inference.flow_steps
        self.device = next(model.parameters()).device
        self.levels = make_levels(model_cfg.oracle.num_levels).to(self.device)
        self.seed = cfg.backtest.seed
        self.reset()

    def reset(self) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    @torch.no_grad()
    def __call__(self, anchors: np.ndarray, positions: np.ndarray) -> np.ndarray:
        context = context_windows(self.market, anchors, self.cfg).to(self.device)
        position = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        with autocast(self.device, self.cfg.train.precision):
            out = predict(self.model, context, position, self.cfg, generator=self.generator, samples=self.samples,
                          flow_steps=self.flow_steps)
        inference = self.cfg.inference
        allocation = readout_allocations(out.probs, self.levels, inference.readout, inference.readout_step,
                                         inference.aggregation, out.q_logit.float().sigmoid())
        return allocation.cpu().numpy()


class ModelPolicy(Policy):
    """Sequential model policy for `backtest.run_backtest` (reference implementation; one decision per call)."""

    def __init__(self, decider: ModelDecider, name: str):
        self.decider = decider
        self.name = name
        self.warmup = decider.cfg.features.lookback

    def reset(self) -> None:
        self.decider.reset()

    def act(self, history: History, position: float) -> float:
        return float(self.decider(np.array([history.t]), np.array([position]))[0])


def run_model_backtest(decider: ModelDecider, start: int, end: int, cost: float, max_step: float | None,
                       initial_position: float, chains: int, burn_in: int, name: str) -> BacktestResult:
    """Decisions at bars start..end-1 (each earning the next bar's return), computed in parallel chains."""
    market, lookback = decider.market, decider.cfg.features.lookback
    if not (lookback <= start < end < len(market.times)):
        raise ValueError(f"need lookback <= start < end < {len(market.times)}, got {start}, {end}")
    decider.reset()
    decisions = end - start
    chains = min(chains, decisions)
    chunk = math.ceil(decisions / chains)
    chunk_starts = start + chunk * np.arange(chains)
    chunk_starts = chunk_starts[chunk_starts < end]
    chain_starts = np.maximum(start, chunk_starts - burn_in)
    chain_starts[0] = start
    chain_ends = np.minimum(chunk_starts + chunk, end)
    lengths = chain_ends - chain_starts

    invalid_before = np.concatenate([[0], np.cumsum(~market.valid)])
    chain_positions = np.full(len(chain_starts), float(initial_position))
    positions = np.empty(decisions)
    for i in range(int(lengths.max())):
        bars = chain_starts + i
        active = i < lengths
        decide = active & (invalid_before[np.minimum(bars, len(market.valid) - 1) + 1]
                           == invalid_before[np.maximum(bars - lookback, 0)])
        if decide.any():
            target = np.clip(decider(bars[decide], chain_positions[decide]).astype(np.float64), -1.0, 1.0)
            if max_step is not None:
                current = chain_positions[decide]
                target = np.clip(target, current - max_step, current + max_step)
            chain_positions[decide] = target
        keep = active & (bars >= chunk_starts)  # burn-in positions are discarded
        positions[bars[keep] - start] = chain_positions[keep]

    previous = np.concatenate([[initial_position], positions[:-1]])
    return BacktestResult(name, market.times[start:end], positions, market.returns[start + 1:end + 1],
                          np.abs(positions - previous), cost)
