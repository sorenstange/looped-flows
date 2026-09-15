import itertools

import numpy as np
import pytest
import torch

from src.oracle import bar_returns, make_levels, oracle_trajectory, tolerance_match, trajectory_pnl

LEVELS = make_levels(3)


def brute_force_best_pnl(returns: torch.Tensor, a0: float, levels: torch.Tensor, cost: float,
                         max_step: float | None = None) -> float:
    horizon = returns.shape[0]
    paths = torch.tensor(list(itertools.product(levels.tolist(), repeat=horizon)), dtype=torch.float64)
    start = torch.full((len(paths),), a0, dtype=torch.float64)
    if max_step is not None:
        steps = torch.cat([start[:, None], paths], dim=1).diff(dim=1).abs()
        feasible = (steps <= max_step + 1e-9).all(dim=1)
        paths, start = paths[feasible], start[feasible]
    pnl = trajectory_pnl(paths, returns.expand(len(paths), -1), start, cost)
    return pnl.max().item()


@pytest.mark.parametrize("cost", [0.0, 0.0005, 0.005, 0.05])
@pytest.mark.parametrize("num_levels, max_step, horizon", [
    (3, None, 6), (5, None, 6), (5, 0.5, 6), (9, 0.25, 5), (21, 0.1, 3), (21, 0.3, 3),
])
def test_matches_brute_force(cost, num_levels, max_step, horizon):
    levels = make_levels(num_levels)
    gen = torch.Generator().manual_seed(0)
    returns = torch.randn(30, horizon, generator=gen, dtype=torch.float64) * 0.01
    a0 = torch.rand(30, generator=gen, dtype=torch.float64) * 2 - 1
    path, pnl = oracle_trajectory(returns, a0, levels, cost, max_step)
    for i in range(len(returns)):
        expected = brute_force_best_pnl(returns[i], a0[i].item(), levels, cost, max_step)
        assert pnl[i].item() == pytest.approx(expected, abs=1e-12)
    assert torch.allclose(trajectory_pnl(levels[path], returns, a0, cost), pnl)


def test_max_step_limits_every_move():
    levels = make_levels(21)
    gen = torch.Generator().manual_seed(1)
    returns = torch.randn(200, 64, generator=gen, dtype=torch.float64) * 0.02
    a0 = torch.rand(200, generator=gen, dtype=torch.float64) * 2 - 1
    path, _ = oracle_trajectory(returns, a0, levels, cost=0.0005, max_step=0.1)
    steps = torch.cat([a0[:, None], levels[path]], dim=1).diff(dim=1).abs()
    assert steps.max() <= 0.1 + 1e-9


def test_max_step_ramps_into_a_trend():
    levels = make_levels(21)
    returns = torch.full((1, 12), 0.01, dtype=torch.float64)
    path, _ = oracle_trajectory(returns, torch.zeros(1), levels, cost=0.0005, max_step=0.1)
    expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]
    assert levels[path][0].tolist() == pytest.approx(expected)


def test_tolerance_match():
    target = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    predicted = torch.tensor([[0.1, 0.2, 0.3, 0.8],  # mean error 0.1 over all steps
                              [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)  # mean error 0.25
    assert tolerance_match(predicted, target, tolerance=0.1).tolist() == [True, False]
    assert tolerance_match(predicted, target, tolerance=0.05).tolist() == [False, False]
    assert tolerance_match(predicted, target, tolerance=0.15, steps=2).tolist() == [True, True]


def test_make_levels():
    levels = make_levels(21)
    assert levels[0] == -1 and levels[10] == 0 and levels[-1] == 1
    assert levels.tolist()[11] == 0.1  # exact decimal after rounding


def test_zero_cost_follows_return_sign():
    returns = torch.tensor([[0.01, -0.02, 0.03, -0.001]], dtype=torch.float64)
    path, _ = oracle_trajectory(returns, torch.zeros(1), LEVELS, cost=0.0)
    assert LEVELS[path].tolist() == [[1.0, -1.0, 1.0, -1.0]]


def test_small_moves_are_not_worth_the_cost():
    returns = torch.tensor([[0.0003, -0.0004, 0.0002, -0.0001]], dtype=torch.float64)
    path, pnl = oracle_trajectory(returns, torch.zeros(1), LEVELS, cost=0.0005)
    assert LEVELS[path].tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert pnl.item() == 0.0


def test_costs_create_holding_through_small_adverse_moves():
    returns = torch.tensor([[0.02, -0.0001, 0.02]], dtype=torch.float64)
    path, _ = oracle_trajectory(returns, torch.zeros(1), LEVELS, cost=0.0005)
    assert LEVELS[path].tolist() == [[1.0, 1.0, 1.0]]


def test_zero_returns_keep_current_level():
    path, _ = oracle_trajectory(torch.zeros(1, 5, dtype=torch.float64), torch.tensor([1.0]), LEVELS, cost=0.0005)
    assert LEVELS[path].tolist() == [[1.0] * 5]


def test_trajectory_pnl_by_hand():
    positions = torch.tensor([[1.0, 1.0, -1.0]])
    returns = torch.tensor([[0.02, -0.01, -0.03]])
    pnl = trajectory_pnl(positions, returns, torch.tensor([0.5]), cost=0.001)
    # gross 0.02 - 0.01 + 0.03 = 0.04; turnover 0.5 + 0 + 2 = 2.5
    assert pnl.item() == pytest.approx(0.04 - 0.0025)


def test_bar_returns():
    close = np.array([100.0, 110.0, 99.0])
    np.testing.assert_allclose(bar_returns(close, "simple")[1:], [0.1, -0.1])
    np.testing.assert_allclose(bar_returns(close, "log")[1:], np.log([1.1, 0.9]))
    assert np.isnan(bar_returns(close)[0])
