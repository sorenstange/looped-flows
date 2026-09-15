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


TIE_EPS = 1e-10  # added to the cost inside the DP so that ties are broken toward lower turnover
STEP_TOL = 1e-9  # float tolerance for the max_step constraint


def _move_cost(distance: torch.Tensor, cost: float, max_step: float | None) -> torch.Tensor:
    """DP cost of moving `distance`; moves larger than `max_step` cost infinity."""
    result = (cost + TIE_EPS) * distance
    return result if max_step is None else result.masked_fill(distance > max_step + STEP_TOL, torch.inf)


@torch.no_grad()
def oracle_rest_values(returns: torch.Tensor, levels: torch.Tensor, cost: float, max_step: float | None = None,
                       first_only: bool = False) -> torch.Tensor:
    """Backward DP over (B, H) return windows.

    rest[:, k, j] is the best PnL of steps k+1..H-1 (0-based) given the position at step k is `levels[j]`, so
    rest[:, H-1] = 0. Returns (B, H, K), or only rest[:, 0] of shape (B, K) if `first_only`.

    The values do not depend on a_0, so they can be computed for many windows at once. Appending zero returns to a
    window does not change them (holding is always feasible and free), so shorter windows can be zero-padded.
    """
    returns = returns.to(torch.float64)
    lv = levels.to(torch.float64)
    batch, horizon = returns.shape
    switch = _move_cost((lv[:, None] - lv[None, :]).abs(), cost, max_step)  # (K_prev, K_next)
    rest = torch.zeros(batch, len(lv), dtype=torch.float64)
    stored = [rest]
    for k in range(horizon - 1, 0, -1):
        rest = ((lv * returns[:, k:k + 1] + rest)[:, None, :] - switch).amax(dim=2)
        if not first_only:
            stored.append(rest)
    return rest if first_only else torch.stack(stored[::-1], dim=1)


@torch.no_grad()
def oracle_step(step_return: torch.Tensor, rest: torch.Tensor, position: torch.Tensor, levels: torch.Tensor,
                cost: float, max_step: float | None = None) -> torch.Tensor:
    """Optimal level index for one step given the step's return (B,), the rest values of that step (B, K) and the
    position before the step (B,)."""
    lv = levels.to(torch.float64)
    move = _move_cost((lv - position.to(torch.float64)[:, None]).abs(), cost, max_step)
    if not torch.isfinite(move).any(dim=1).all():
        raise ValueError("some position has no level within max_step")
    return (lv * step_return.to(torch.float64)[:, None] - move + rest).argmax(dim=1)


@torch.no_grad()
def oracle_trajectory(returns: torch.Tensor, a0: torch.Tensor, levels: torch.Tensor, cost: float,
                      max_step: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve max_p sum_k p_k R_k - cost |p_k - p_{k-1}| over p_k in `levels` exactly, by dynamic programming.

    Args:
        returns: (B, H) future bar returns R_1..R_H.
        a0: (B,) current position, any value in [-1, 1].
        levels: (K,) allowed allocation levels.
        max_step: if given, every move |p_k - p_{k-1}| (including |p_1 - a_0|) is at most this large.
    Returns:
        (B, H) indices into `levels` of the optimal path, and (B,) its net PnL (float64).

    Ties are broken toward lower turnover by running the DP with `cost + TIE_EPS`; this can only change the choice
    between paths whose true PnL differs by less than TIE_EPS * turnover. The PnL is recomputed with the real cost.
    """
    returns = returns.to(torch.float64)
    lv = levels.to(torch.float64)
    rest = oracle_rest_values(returns, lv, cost, max_step)
    path = torch.empty(returns.shape, dtype=torch.long)
    position = a0.to(torch.float64)
    for k in range(returns.shape[1]):
        path[:, k] = oracle_step(returns[:, k], rest[:, k], position, lv, cost, max_step)
        position = lv[path[:, k]]
    return path, trajectory_pnl(lv[path], returns, a0.to(torch.float64), cost)


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
