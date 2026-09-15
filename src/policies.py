"""Backtest policies: rule-based baselines and the hindsight oracle."""

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from src.backtest import History, Policy
from src.config import Config
from src.dataset import MarketData
from src.oracle import make_levels, oracle_rest_values, oracle_step

POLICY_NAMES = ("flat", "buy_hold", "random", "ma_crossover", "momentum", "oracle")
RULE_BASED = ("buy_hold", "random", "ma_crossover", "momentum")  # baselines also run with the step limit


def readout_allocation(probs: torch.Tensor, levels: torch.Tensor, readout: str = "step", readout_step: int = 1,
                       aggregation: str = "mean", scores: torch.Tensor | None = None) -> float:
    """Traded allocation from K sampled trajectories of a model.

    Args:
        probs: (K, H, L) predicted level probabilities per sample and trajectory step.
        levels: (L,) allocation levels.
        readout: "step" uses the expected level at `readout_step`; "prefix_mean" averages steps 1..readout_step.
        readout_step: 1-based trajectory step.
        aggregation: "mean" averages the per-sample values (the Monte Carlo expected position; uncertainty across
            samples becomes position size); "best_q" takes the sample with the highest confidence score; "q_weighted"
            averages with weights proportional to the scores.
        scores: (K,) non-negative confidence scores, required for "best_q" and "q_weighted".
    """
    expected = probs.to(torch.float64) @ levels.to(torch.float64)  # (K, H)
    if readout == "step":
        per_sample = expected[:, readout_step - 1]
    elif readout == "prefix_mean":
        per_sample = expected[:, :readout_step].mean(dim=1)
    else:
        raise ValueError(f"unknown readout {readout!r}")
    if aggregation == "mean":
        return per_sample.mean().item()
    if aggregation not in ("best_q", "q_weighted"):
        raise ValueError(f"unknown aggregation {aggregation!r}")
    if scores is None:
        raise ValueError(f"aggregation={aggregation!r} needs confidence scores")
    if aggregation == "best_q":
        return per_sample[scores.argmax()].item()
    weights = scores.to(torch.float64).clamp(min=0)
    if weights.sum() <= 0:
        return per_sample.mean().item()
    return (per_sample * weights).sum().item() / weights.sum().item()


class ConstantPolicy(Policy):
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def act(self, history: History, position: float) -> float:
        return self.value


class RandomPolicy(Policy):
    """Uniform random target in [-1, 1] every bar; a sanity check that should lose roughly its fees."""
    name = "random"

    def __init__(self, seed: int):
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, history: History, position: float) -> float:
        return float(self.rng.uniform(-1.0, 1.0))


class MACrossoverPolicy(Policy):
    """Full long while the fast moving average of the close is above the slow one, full short otherwise."""
    name = "ma_crossover"

    def __init__(self, fast: int, slow: int):
        self.fast, self.slow = fast, slow
        self.warmup = slow

    def act(self, history: History, position: float) -> float:
        close = history.close
        return 1.0 if close[-self.fast:].mean() > close[-self.slow:].mean() else -1.0


class MomentumPolicy(Policy):
    """Full long if the close is above the close `window` bars ago, full short otherwise."""
    name = "momentum"

    def __init__(self, window: int):
        self.window = window
        self.warmup = window

    def act(self, history: History, position: float) -> float:
        return 1.0 if history.close[-1] > history.close[-1 - self.window] else -1.0


class OraclePolicy(Policy):
    """Hindsight upper bound, not tradable: at every bar, take the first step of the oracle trajectory over the next
    `horizon` bars (never past `last_bar`), starting from the current position.

    The position-independent part of the DP is computed for all remaining decision bars in one batched pass on the
    first call, so each decision only solves the first step.
    """
    name = "oracle"
    chunk_size = 4096

    def __init__(self, returns: np.ndarray, last_bar: int, horizon: int, levels: torch.Tensor, cost: float,
                 max_step: float | None):
        self.returns = returns
        self.last_bar = last_bar
        self.horizon = horizon
        self.levels = levels.to(torch.float64)
        self.cost = cost
        self.max_step = max_step
        self._start = 0
        self._rest: torch.Tensor | None = None

    def act(self, history: History, position: float) -> float:
        t = history.t
        if self._rest is None or not self._start <= t < self._start + len(self._rest):
            self._precompute(t)
        i = t - self._start
        step = oracle_step(torch.tensor([self.returns[t + 1]]), self._rest[i:i + 1],
                           torch.tensor([position], dtype=torch.float64), self.levels, self.cost, self.max_step)
        return self.levels[step[0]].item()

    def _precompute(self, start: int) -> None:
        """First-step rest values for decision bars start..last_bar-1; windows are zero-padded past last_bar."""
        if not start < self.last_bar:
            raise ValueError(f"oracle cannot decide at bar {start} >= last_bar {self.last_bar}")
        padded = np.concatenate([self.returns[start + 1:self.last_bar + 1], np.zeros(self.horizon)])
        windows = sliding_window_view(padded, self.horizon)[:self.last_bar - start]  # row i: bars start+i+1 ..
        self._rest = torch.cat([
            oracle_rest_values(torch.from_numpy(np.ascontiguousarray(windows[i:i + self.chunk_size])), self.levels,
                               self.cost, self.max_step, first_only=True)
            for i in range(0, len(windows), self.chunk_size)
        ])
        self._start = start


def make_policy(name: str, cfg: Config, market: MarketData, last_bar: int) -> Policy:
    """Build a policy by name. `last_bar` bounds the future the oracle may look at (the end of the backtest)."""
    baselines = cfg.backtest.baselines
    if name == "flat":
        return ConstantPolicy("flat", 0.0)
    if name == "buy_hold":
        return ConstantPolicy("buy_hold", 1.0)
    if name == "random":
        return RandomPolicy(cfg.backtest.seed)
    if name == "ma_crossover":
        return MACrossoverPolicy(baselines.ma_fast, baselines.ma_slow)
    if name == "momentum":
        return MomentumPolicy(baselines.momentum_window)
    if name == "oracle":
        return OraclePolicy(market.returns, last_bar, cfg.oracle.horizon, make_levels(cfg.oracle.num_levels),
                            cfg.oracle.cost, cfg.oracle.max_step)
    raise ValueError(f"unknown policy {name!r}; known: {POLICY_NAMES}")
