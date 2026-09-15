from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.backtest import run_backtest
from src.config import load_config
from src.data import clean_ohlcv
from src.dataset import MarketData
from src.model_backtest import ModelDecider, ModelPolicy, check_compatible, run_model_backtest
from src.modules import Denoiser
from src.policies import readout_allocation, readout_allocations
from tests.conftest import make_bars

TINY = ["features.lookback=16", "features.rolling_window=100", "oracle.horizon=8", "model.width=32", "model.heads=4",
        "model.mlp_width=64", "model.inner_steps=2", "model.patch_size=4", "train.method=direct"]
START, END = 200, 260


def setup(extra=()):
    cfg = load_config(overrides=TINY + list(extra))
    market = MarketData.from_bars(clean_ohlcv(make_bars(), "1h"), cfg)
    torch.manual_seed(0)
    return cfg, market, Denoiser(cfg).eval()


class ContextOnlyModel(torch.nn.Module):
    """Deterministic stand-in whose prediction ignores a_0, so chains cannot differ from a sequential run."""

    def __init__(self, cfg):
        super().__init__()
        self.horizon, self.num_levels = cfg.oracle.horizon, cfg.oracle.num_levels
        self.weight = torch.nn.Parameter(torch.randn(self.horizon, self.num_levels, generator=torch.Generator()
                                                     .manual_seed(0)))

    def forward(self, context, position, noisy, t, state=None):
        logits = context.mean(dim=(1, 2))[:, None, None] * self.weight * 20
        return SimpleNamespace(logits=logits, q_logit=torch.zeros(context.shape[0]), state=None)


def sequential(decider, cost, max_step, initial=0.0):
    return run_backtest(ModelPolicy(decider, "model"), decider.market, START, END, cost, decider.cfg.features.lookback,
                        max_step, initial)


@pytest.mark.parametrize("max_step", [None, 0.1])
def test_single_chain_equals_sequential_policy(max_step):
    cfg, market, model = setup()
    decider = ModelDecider(model, cfg, cfg, market)
    reference = sequential(decider, 0.0005, max_step, initial=0.3)
    chained = run_model_backtest(decider, START, END, 0.0005, max_step, 0.3, chains=1, burn_in=0, name="model")
    np.testing.assert_allclose(chained.positions, reference.positions, atol=1e-12)
    np.testing.assert_allclose(chained.pnl, reference.pnl, atol=1e-12)
    if max_step is not None:
        assert np.abs(np.diff(np.concatenate([[0.3], chained.positions]))).max() <= max_step + 1e-9


@pytest.mark.parametrize("chains, burn_in", [(7, 0), (7, 3), (60, 5)])
def test_chains_match_sequential_when_positions_do_not_feed_back(chains, burn_in):
    cfg, market, _ = setup()
    decider = ModelDecider(ContextOnlyModel(cfg), cfg, cfg, market)
    reference = sequential(decider, 0.0005, None)
    chained = run_model_backtest(decider, START, END, 0.0005, None, 0.0, chains=chains, burn_in=burn_in, name="m")
    np.testing.assert_allclose(chained.positions, reference.positions, atol=1e-12)


def test_full_burn_in_reproduces_the_sequential_run_in_every_chain():
    cfg, market, model = setup()
    decider = ModelDecider(model, cfg, cfg, market)
    reference = sequential(decider, 0.0005, 0.1)
    chained = run_model_backtest(decider, START, END, 0.0005, 0.1, 0.0, chains=5, burn_in=END - START, name="model")
    np.testing.assert_allclose(chained.positions, reference.positions, atol=1e-12)


def test_looped_flow_model_backtest_runs():
    cfg, market, model = setup(["train.method=looped_flow", "backtest.inference_samples=2",
                                "backtest.inference_flow_steps=3"])
    decider = ModelDecider(model, cfg, cfg, market)
    assert (decider.samples, decider.flow_steps) == (2, 3)
    result = run_model_backtest(decider, START, END, 0.0005, 0.1, 0.0, chains=8, burn_in=4, name="model")
    assert result.positions.shape == (END - START,)
    assert np.isfinite(result.pnl).all() and np.abs(result.positions).max() <= 1.0


def test_check_compatible():
    cfg = load_config(overrides=TINY)
    check_compatible(cfg, load_config(overrides=TINY + ["backtest.chains=3"]))
    with pytest.raises(ValueError, match="features.lookback"):
        check_compatible(cfg, load_config(overrides=TINY + ["features.lookback=32"]))
    with pytest.raises(ValueError, match="oracle.cost"):
        check_compatible(cfg, load_config(overrides=TINY + ["oracle.cost=0.001"]))


@pytest.mark.parametrize("aggregation", ["mean", "best_q", "q_weighted"])
def test_batched_readout_matches_single(aggregation):
    gen = torch.Generator().manual_seed(0)
    probs = torch.softmax(torch.randn(5, 3, 8, 21, generator=gen), dim=-1)
    scores = torch.rand(5, 3, generator=gen)
    levels = torch.linspace(-1, 1, 21, dtype=torch.float64)
    batched = readout_allocations(probs, levels, "prefix_mean", 4, aggregation, scores)
    single = [readout_allocation(probs[i], levels, "prefix_mean", 4, aggregation, scores[i]) for i in range(5)]
    np.testing.assert_allclose(batched.numpy(), single)
