"""Hindsight-optimal allocation trajectories (the training targets) and trajectory PnL.

Timing convention, shared by oracle, backtest and baselines: a decision is made at the close of the anchor bar T.
Position `p_k` (k = 1..H) is held over bar T+k and earns that bar's return `R_k` (close T+k-1 -> close T+k). Moving
from `p_{k-1}` to `p_k` costs `cost * |p_k - p_{k-1}|`, with `p_0 = a_0` the position held before the decision.
"""

import numpy as np
import torch


def make_levels(num_levels: int) -> torch.Tensor:
    """Evenly spaced allocation levels over [-1, 1], rounded so that e.g. 0.1 steps are exact decimals."""
    return torch.from_numpy(np.round(np.linspace(-1.0, 1.0, num_levels), 12))


def bar_returns(close: np.ndarray, return_type: str = "simple") -> np.ndarray:
    """Return of each bar from the previous close, float64; element 0 is NaN."""
    close = np.asarray(close, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = close[1:] / close[:-1] - 1 if return_type == "simple" else np.diff(np.log(close))
    return np.concatenate([[np.nan], returns])


def trajectory_pnl(positions: torch.Tensor, returns: torch.Tensor, a0: torch.Tensor, cost: float) -> torch.Tensor:
    """Net PnL per trajectory. positions, returns: (B, H); a0: (B,)."""
    previous = torch.cat([a0[:, None].to(positions.dtype), positions[:, :-1]], dim=1)
    return (positions * returns - cost * (positions - previous).abs()).sum(dim=1)


@torch.no_grad()
def oracle_trajectory(returns: torch.Tensor, a0: torch.Tensor, levels: torch.Tensor, cost: float,
                      max_step: float | None = None, tie_eps: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve max_p sum_k p_k R_k - cost |p_k - p_{k-1}| over p_k in `levels` exactly, by dynamic programming.

    Args:
        returns: (B, H) future bar returns R_1..R_H.
        a0: (B,) current position, any value in [-1, 1].
        levels: (K,) allowed allocation levels.
        max_step: if given, every move |p_k - p_{k-1}| (including |p_1 - a_0|) is at most this large.
    Returns:
        (B, H) indices into `levels` of the optimal path, and (B,) its net PnL (float64).

    Ties are broken toward lower turnover by running the DP with `cost + tie_eps`; this can only change the choice
    between paths whose true PnL differs by less than tie_eps * turnover. The PnL is recomputed with the real cost.
    """
    returns = returns.to(torch.float64)
    a0 = a0.to(torch.float64)
    lv = levels.to(torch.float64)
    batch, horizon = returns.shape
    penalty = cost + tie_eps

    def move_cost(distance: torch.Tensor) -> torch.Tensor:  # forbidden moves cost infinity
        result = penalty * distance
        return result if max_step is None else result.masked_fill(distance > max_step + 1e-9, torch.inf)

    switch = move_cost((lv[:, None] - lv[None, :]).abs())  # (K_prev, K_next)
    value = lv * returns[:, :1] - move_cost((lv - a0[:, None]).abs())  # (B, K): best PnL ending at each level
    if not torch.isfinite(value).any(dim=1).all():
        raise ValueError("some a0 has no level within max_step")
    backpointer = torch.zeros(batch, horizon, len(lv), dtype=torch.long)
    for k in range(1, horizon):
        value, backpointer[:, k] = (value[:, :, None] - switch).max(dim=1)
        value = value + lv * returns[:, k:k + 1]

    path = torch.empty(batch, horizon, dtype=torch.long)
    path[:, -1] = value.argmax(dim=1)
    for k in range(horizon - 1, 0, -1):
        path[:, k - 1] = backpointer[:, k].gather(1, path[:, k:k + 1]).squeeze(1)
    return path, trajectory_pnl(lv[path], returns, a0, cost)


def oracle_stats(path: torch.Tensor, levels: torch.Tensor, a0: torch.Tensor, pnl: torch.Tensor) -> dict:
    """Summary statistics of a batch of oracle trajectories (for sanity-checking targets)."""
    positions = levels.to(torch.float64)[path]
    previous = torch.cat([a0[:, None].to(torch.float64), positions[:, :-1]], dim=1)
    changes = path[:, 1:] != path[:, :-1]
    runs = changes.sum().item() + path.shape[0]  # holding runs, truncated at the horizon edges
    sign = positions.sign()
    direction_runs = (sign[:, 1:] != sign[:, :-1]).sum().item() + path.shape[0]
    return {
        "samples": path.shape[0],
        "level_freq": {f"{level:g}": (path == i).double().mean().item() for i, level in enumerate(levels.tolist())},
        "short_frac": (positions < 0).double().mean().item(),
        "flat_frac": (positions == 0).double().mean().item(),
        "long_frac": (positions > 0).double().mean().item(),
        "mean_abs_position": positions.abs().mean().item(),
        "full_position_frac": (positions.abs() == 1).double().mean().item(),
        "turnover_per_bar": (positions - previous).abs().mean().item(),
        "mean_holding_bars": path.numel() / runs,  # bars at an unchanged level
        "mean_direction_bars": path.numel() / direction_runs,  # bars with unchanged sign (short / flat / long)
        "pnl_mean": pnl.mean().item(),
        "pnl_median": pnl.median().item(),
    }
