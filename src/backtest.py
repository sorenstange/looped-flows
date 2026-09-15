"""Bar-by-bar backtest engine and performance metrics.

Timing and costs follow `src.oracle`: the position decided at the close of bar t is held over bar t+1, earns that
bar's return R_{t+1}, and costs `cost * |p_t - p_{t-1}|` when it is set.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.data import interval_to_timedelta
from src.dataset import MarketData, split_ranges


@dataclass(frozen=True)
class History:
    """Market data known at the close of bar `t`: every array is truncated to bars 0..t."""
    t: int
    close: np.ndarray
    features: np.ndarray
    returns: np.ndarray
    valid: np.ndarray

    @classmethod
    def at(cls, market: MarketData, t: int) -> "History":
        end = t + 1
        return cls(t, market.close[:end], market.features[:end], market.returns[:end], market.valid[:end])


class Policy:
    """Maps what is known at the close of bar t, plus the current position, to a desired position in [-1, 1]."""
    name = "policy"
    warmup = 0  # bars of history needed before the first decision

    def reset(self) -> None:
        pass

    def act(self, history: History, position: float) -> float:
        raise NotImplementedError


@dataclass
class BacktestResult:
    policy: str
    times: pd.DatetimeIndex  # open time of each decision bar t
    positions: np.ndarray  # position held over bar t+1
    returns: np.ndarray  # return of bar t+1
    turnover: np.ndarray  # |p_t - p_{t-1}|
    cost: float

    @property
    def gross(self) -> np.ndarray:
        return self.positions * self.returns

    @property
    def fees(self) -> np.ndarray:
        return self.cost * self.turnover

    @property
    def pnl(self) -> np.ndarray:
        return self.gross - self.fees

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"position": self.positions, "return": self.returns, "turnover": self.turnover,
                             "pnl": self.pnl}, index=self.times)


def run_backtest(policy: Policy, market: MarketData, start: int, end: int, cost: float, lookback: int,
                 max_step: float | None = None, initial_position: float = 0.0) -> BacktestResult:
    """Run `policy` with decisions at bars start..end-1; the last decision earns the return of bar `end`.

    The policy is only queried when bars t-lookback..t are all valid; otherwise the position is held. Desired positions
    are clipped to [-1, 1] and, if `max_step` is given, to `position ± max_step`.
    """
    if not (max(lookback, policy.warmup) <= start < end < len(market.times)):
        raise ValueError(f"need max(lookback, warmup) <= start < end < {len(market.times)}, got {start}, {end}")
    invalid_before = np.concatenate([[0], np.cumsum(~market.valid)])
    policy.reset()
    positions = np.empty(end - start)
    position = initial_position
    for i, t in enumerate(range(start, end)):
        if invalid_before[t + 1] == invalid_before[t - lookback]:
            target = min(max(float(policy.act(History.at(market, t), position)), -1.0), 1.0)
            if max_step is not None:
                target = min(max(target, position - max_step), position + max_step)
            position = target
        positions[i] = position
    previous = np.concatenate([[initial_position], positions[:-1]])
    return BacktestResult(policy.name, market.times[start:end], positions, market.returns[start + 1:end + 1],
                          np.abs(positions - previous), cost)


def performance(result: BacktestResult, interval: str) -> dict:
    """Metrics of a backtest. PnL is additive (constant notional), so returns and drawdowns are in units of notional."""
    bars_per_year = pd.Timedelta(days=365.25) / interval_to_timedelta(interval)
    pnl, gross, positions = result.pnl, result.gross, result.positions
    equity = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity
    vol = pnl.std()
    active = positions != 0
    sign = np.sign(positions)
    direction_runs = int((sign[1:] != sign[:-1]).sum()) + 1
    return {
        "bars": len(pnl),
        "years": len(pnl) / bars_per_year,
        "net_pnl": float(pnl.sum()),
        "gross_pnl": float(gross.sum()),
        "fees": float(result.fees.sum()),
        "annual_return": float(pnl.mean() * bars_per_year),
        "annual_vol": float(vol * np.sqrt(bars_per_year)),
        "sharpe": float(pnl.mean() / vol * np.sqrt(bars_per_year)) if vol > 0 else 0.0,
        "max_drawdown": float(drawdown.max()),
        "turnover_per_bar": float(result.turnover.mean()),
        "mean_abs_position": float(np.abs(positions).mean()),
        "long_frac": float((positions > 0).mean()),
        "short_frac": float((positions < 0).mean()),
        "mean_direction_bars": len(positions) / direction_runs,  # bars per run of unchanged sign (short/flat/long)
        "hit_rate": float((gross[active] > 0).mean()) if active.any() else None,
    }


def decision_range(market: MarketData, cfg: Config, warmup: int = 0, horizon: int | None = None) -> tuple[int, int]:
    """Decision bars [start, end) on `cfg.backtest.split`: from the split's first dataset anchor (later if a policy
    needs more warmup) to the second-to-last bar of the split, so the last decision earns the split's last return.
    `horizon` overrides `cfg.oracle.horizon` for the purge gap, to align runs with different horizons."""
    lookback = cfg.features.lookback
    horizon = cfg.oracle.horizon if horizon is None else horizon
    lo, hi = split_ranges(market.times, cfg.splits, lookback, horizon)[cfg.backtest.split]
    return max(lo + lookback, warmup), hi - 1
