import numpy as np
import pandas as pd
import pytest
import torch

from src.backtest import History, Policy, performance, run_backtest
from src.config import load_config
from src.dataset import MarketData
from src.features import FEATURE_NAMES
from src.oracle import make_levels, oracle_trajectory, trajectory_pnl
from src.policies import MACrossoverPolicy, MomentumPolicy, OraclePolicy, make_policy, readout_allocation


def toy_market(returns, valid=None) -> MarketData:
    returns = np.concatenate([[np.nan], np.asarray(returns, dtype=np.float64)])
    close = 100 * np.cumprod(np.nan_to_num(returns) + 1)
    n = len(returns)
    return MarketData(pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"), close,
                      np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32), returns,
                      np.ones(n, dtype=bool) if valid is None else np.asarray(valid))


class ScriptedPolicy(Policy):
    name = "scripted"

    def __init__(self, targets):
        self.targets = list(targets)
        self.calls = []

    def act(self, history, position):
        self.calls.append((history.t, len(history.close), position))
        return self.targets[len(self.calls) - 1]


def test_positions_fees_and_pnl_by_hand():
    market = toy_market([0.0, 0.01, -0.02, 0.03, 0.01])
    policy = ScriptedPolicy([1.0, 2.0, -0.5, 0.0])
    result = run_backtest(policy, market, start=1, end=5, cost=0.001, lookback=1)

    assert result.positions.tolist() == [1.0, 1.0, -0.5, 0.0]  # 2.0 clipped to 1.0
    np.testing.assert_allclose(result.returns, [0.01, -0.02, 0.03, 0.01])  # returns of bars t+1 = 2..5
    np.testing.assert_allclose(result.turnover, [1.0, 0.0, 1.5, 0.5])
    np.testing.assert_allclose(result.pnl, [0.01 - 0.001, -0.02, -0.015 - 0.0015, -0.0005])
    assert [call[:2] for call in policy.calls] == [(1, 2), (2, 3), (3, 4), (4, 5)]  # history ends at bar t
    assert [call[2] for call in policy.calls] == [0.0, 1.0, 1.0, -0.5]


def test_max_step_clips_executed_moves():
    market = toy_market(np.full(6, 0.01))
    result = run_backtest(ScriptedPolicy([1.0, 1.0, -1.0, 0.25, 0.25]), market, 1, 6, cost=0.0, lookback=1,
                          max_step=0.1)
    np.testing.assert_allclose(result.positions, [0.1, 0.2, 0.1, 0.2, 0.25])


def test_invalid_bars_hold_the_position():
    valid = np.ones(10, dtype=bool)
    valid[4] = False
    policy = ScriptedPolicy([1.0, -1.0, 0.5, 0.5, 0.5])
    result = run_backtest(policy, toy_market(np.full(9, 0.01), valid), start=2, end=9, cost=0.0, lookback=1)
    assert [t for t, _, _ in policy.calls] == [2, 3, 6, 7, 8]  # windows t-1..t touching bar 4 are skipped
    np.testing.assert_allclose(result.positions, [1.0, -1.0, -1.0, -1.0, 0.5, 0.5, 0.5])


def test_engine_pnl_matches_trajectory_pnl():
    rng = np.random.default_rng(0)
    market = toy_market(rng.normal(0, 0.01, 50))
    result = run_backtest(ScriptedPolicy(rng.uniform(-1, 1, 40)), market, 5, 45, cost=0.0007, lookback=5,
                          initial_position=0.3)
    expected = trajectory_pnl(torch.from_numpy(result.positions)[None], torch.from_numpy(result.returns)[None],
                              torch.tensor([0.3], dtype=torch.float64), 0.0007)
    assert result.pnl.sum() == pytest.approx(expected.item(), abs=1e-12)


@pytest.mark.parametrize("max_step", [None, 0.1])
def test_receding_oracle_with_full_horizon_follows_the_optimal_path(max_step):
    rng = np.random.default_rng(1)
    market = toy_market(rng.normal(0, 0.01, 40))
    start, end, cost = 3, 40, 0.0005
    levels = make_levels(21)
    policy = OraclePolicy(market.returns, last_bar=end, horizon=100, levels=levels, cost=cost, max_step=max_step)
    result = run_backtest(policy, market, start, end, cost, lookback=3, max_step=max_step)

    future = torch.from_numpy(market.returns[start + 1:end + 1])[None]
    path, best = oracle_trajectory(future, torch.zeros(1), levels, cost, max_step)
    np.testing.assert_allclose(result.positions, levels[path[0]].numpy())
    assert result.pnl.sum() == pytest.approx(best.item(), abs=1e-12)


class NaiveOraclePolicy(Policy):
    """Reference implementation: full DP over the truncated future window at every bar."""
    name = "naive_oracle"

    def __init__(self, returns, last_bar, horizon, levels, cost, max_step):
        self.returns, self.last_bar, self.horizon = returns, last_bar, horizon
        self.levels, self.cost, self.max_step = levels, cost, max_step

    def act(self, history, position):
        t = history.t
        future = torch.from_numpy(self.returns[t + 1:min(t + self.horizon, self.last_bar) + 1])[None]
        path, _ = oracle_trajectory(future, torch.tensor([position], dtype=torch.float64), self.levels, self.cost,
                                    self.max_step)
        return self.levels[path[0, 0]].item()


@pytest.mark.parametrize("max_step, horizon", [(None, 7), (0.1, 16), (0.3, 5)])
def test_fast_oracle_policy_matches_naive_replanning(max_step, horizon):
    rng = np.random.default_rng(2)
    market = toy_market(rng.normal(0, 0.01, 120))
    levels, cost, start, end = make_levels(21), 0.001, 4, 110
    args = (market.returns, end, horizon, levels, cost, max_step)
    fast = OraclePolicy(*args)
    fast.chunk_size = 16  # exercise chunking
    fast_result = run_backtest(fast, market, start, end, 0.0005, lookback=4, max_step=max_step, initial_position=0.35)
    naive_result = run_backtest(NaiveOraclePolicy(*args), market, start, end, 0.0005, lookback=4, max_step=max_step,
                                initial_position=0.35)
    np.testing.assert_array_equal(fast_result.positions, naive_result.positions)


def test_oracle_never_reads_past_last_bar():
    returns = np.full(30, 0.01)
    returns[25:] = -0.5  # a crash after the last bar must not affect decisions
    market = toy_market(returns)
    policy = OraclePolicy(market.returns, last_bar=20, horizon=64, levels=make_levels(3), cost=0.0, max_step=None)
    result = run_backtest(policy, market, 2, 20, cost=0.0, lookback=2)
    assert (result.positions == 1.0).all()


def test_trend_baselines_go_long_in_uptrend_and_short_in_downtrend():
    up = toy_market(np.full(30, 0.01))
    down = toy_market(np.full(30, -0.01))
    for policy in (MACrossoverPolicy(3, 10), MomentumPolicy(10)):
        assert (run_backtest(policy, up, 12, 30, 0.0, lookback=2).positions == 1.0).all()
        assert (run_backtest(policy, down, 12, 30, 0.0, lookback=2).positions == -1.0).all()
        with pytest.raises(ValueError):
            run_backtest(policy, up, 5, 30, 0.0, lookback=2)  # not enough history for the warmup


def test_readout_allocation():
    levels = torch.tensor([-1.0, 0.0, 1.0])
    # 2 samples, 3 steps, 3 levels
    probs = torch.tensor([
        [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],  # expected levels 0.5, 1.0, 1.0
        [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # expected levels -0.5, -1.0, 0.0
    ])
    scores = torch.tensor([0.2, 0.9])
    assert readout_allocation(probs, levels) == pytest.approx(0.0)  # mean of 0.5 and -0.5
    assert readout_allocation(probs, levels, readout_step=2) == pytest.approx(0.0)
    assert readout_allocation(probs, levels, readout="prefix_mean", readout_step=3) == pytest.approx(
        ((0.5 + 1.0 + 1.0) / 3 + (-0.5 - 1.0 + 0.0) / 3) / 2)
    assert readout_allocation(probs, levels, aggregation="best_q", scores=scores) == pytest.approx(-0.5)
    assert readout_allocation(probs, levels, readout_step=3, aggregation="best_q", scores=scores) == pytest.approx(0.0)
    assert readout_allocation(probs, levels, aggregation="q_weighted", scores=scores) == pytest.approx(
        (0.2 * 0.5 + 0.9 * -0.5) / 1.1)
    assert readout_allocation(probs, levels, aggregation="q_weighted", scores=torch.zeros(2)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        readout_allocation(probs, levels, aggregation="best_q")


def test_make_policy_builds_all_configured_policies():
    cfg = load_config()
    market = toy_market(np.full(300, 0.001))
    names = [make_policy(name, cfg, market, last_bar=299).name for name in cfg.backtest.policies]
    assert names == cfg.backtest.policies


def test_performance_metrics_by_hand():
    market = toy_market([0.0, 0.02, -0.01, -0.01, 0.03])
    result = run_backtest(ScriptedPolicy([1.0, 1.0, 1.0, 0.0]), market, 1, 5, cost=0.0, lookback=1)
    m = performance(result, "1h")
    # pnl per bar: 0.02, -0.01, -0.01, 0.0 -> equity 0.02, 0.01, 0.0, 0.0
    assert m["net_pnl"] == pytest.approx(0.0)
    assert m["max_drawdown"] == pytest.approx(0.02)
    assert m["hit_rate"] == pytest.approx(1 / 3)
    assert m["mean_abs_position"] == pytest.approx(0.75)
    assert m["bars"] == 4 and m["years"] == pytest.approx(4 / (365.25 * 24))
