from types import SimpleNamespace

import pytest
import torch

from src.config import load_config
from src.features import FEATURE_NAMES
from src.losses import level_log_probs
from src.modules import Denoiser
from src.sampling import backtrack, predict, sample

LOOKBACK, HORIZON, LEVELS = 16, 8, 21
TINY = [f"features.lookback={LOOKBACK}", f"oracle.horizon={HORIZON}", "model.width=32", "model.heads=4",
        "model.mlp_width=64", "model.inner_steps=2", "model.patch_size=4", "train.method=looped_flow"]


def tiny(extra=()):
    cfg = load_config(overrides=TINY + list(extra))
    torch.manual_seed(0)
    return cfg, Denoiser(cfg).eval()


def inputs(batch=3):
    gen = torch.Generator().manual_seed(1)
    return torch.randn(batch, LOOKBACK, len(FEATURE_NAMES), generator=gen), torch.rand(batch, generator=gen) * 2 - 1


def test_shapes_and_sample_diversity():
    cfg, model = tiny()
    context, position = inputs()
    out = sample(model, context, position, cfg, samples=4, flow_steps=6, gamma=5.0,
                 generator=torch.Generator().manual_seed(0))
    assert out.probs.shape == (3, 4, HORIZON, LEVELS) and out.levels.shape == (3, 4, HORIZON)
    assert out.q_logit.shape == (3, 4)
    assert torch.allclose(out.probs.sum(-1), torch.ones(3, 4, HORIZON), atol=1e-5)
    assert not torch.allclose(out.probs[:, 0], out.probs[:, 1])  # different noise per sample


def test_gamma_zero_is_euler_integration():
    cfg, model = tiny()
    context, position = inputs(batch=2)
    steps, sigma = 5, 1 / LEVELS ** 0.5
    out = sample(model, context, position, cfg, samples=1, flow_steps=steps, gamma=0.0,
                 generator=torch.Generator().manual_seed(7))

    # Reference: forward Euler on the probability flow coupled with the recurrence (Eq. 8).
    x = torch.randn((2, HORIZON, LEVELS), generator=torch.Generator().manual_seed(7)) * sigma
    state = None
    with torch.no_grad():
        for i in range(steps):
            t = i / steps
            result = model(context, position, x, torch.full((2,), t), state)
            state = result.state
            prediction = level_log_probs(result.logits, cfg.train.loss).exp().float()
            x = x + (1 / steps) * (prediction - x) / (1 - t)
    assert torch.allclose(out.probs[:, 0], prediction, atol=1e-6)
    assert torch.equal(out.levels[:, 0], x.argmax(dim=-1))


class IdealDenoiser(torch.nn.Module):
    """Always predicts the true solution with (numerically) full confidence."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target, self.horizon, self.num_levels = target, target.shape[1], LEVELS

    def forward(self, context, position, noisy, t, state=None):
        repeats = noisy.shape[0] // self.target.shape[0]
        logits = 60.0 * torch.nn.functional.one_hot(self.target.repeat_interleave(repeats, 0), LEVELS).float()
        return SimpleNamespace(logits=logits, q_logit=torch.zeros(noisy.shape[0]), state=None)


@pytest.mark.parametrize("gamma", [0.0, 1.0, 5.0])
def test_ideal_denoiser_transports_noise_to_the_solution(gamma):
    cfg = load_config(overrides=TINY + ["train.loss=softmax"])
    target = torch.randint(0, LEVELS, (3, HORIZON), generator=torch.Generator().manual_seed(3))
    context, position = inputs()
    out = sample(IdealDenoiser(target), context, position, cfg, samples=5, flow_steps=8, gamma=gamma,
                 generator=torch.Generator().manual_seed(0))
    assert torch.equal(out.levels, target[:, None].expand(3, 5, HORIZON))


def test_backtracking_preserves_the_interpolant_marginal():
    gen = torch.Generator().manual_seed(0)
    sigma, t, dt, gamma = 0.3, 0.6, 0.05, 5.0
    x1 = torch.ones(200_000)
    x_t = t * x1 + sigma * (1 - t) * torch.randn(200_000, generator=gen)
    x_bar, s = backtrack(x_t, t, dt, gamma, sigma * torch.randn(200_000, generator=gen))
    assert s == pytest.approx((1 - gamma * dt) * t)
    assert (x_bar - s * x1).mean().item() == pytest.approx(0.0, abs=2e-3)
    assert (x_bar - s * x1).std().item() == pytest.approx(sigma * (1 - s), rel=1e-2)


def test_predict_direct_is_a_single_deterministic_sample():
    cfg, model = tiny(["train.method=direct"])
    context, position = inputs()
    first, second = predict(model, context, position, cfg), predict(model, context, position, cfg)
    assert first.probs.shape == (3, 1, HORIZON, LEVELS)
    assert torch.equal(first.probs, second.probs)
