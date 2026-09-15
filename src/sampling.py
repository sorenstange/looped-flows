"""Generating allocation trajectories from a trained denoiser (paper Algorithm 2 and Appendix C)."""

from dataclasses import dataclass

import torch

from src.config import Config
from src.losses import level_log_probs
from src.modules import Denoiser


@dataclass
class Samples:
    probs: torch.Tensor  # (B, K, H, L) level probabilities predicted at the last sampler step
    levels: torch.Tensor  # (B, K, H) level indices of the final flow state, round(x_1)
    q_logit: torch.Tensor  # (B, K) confidence logits at the last sampler step


def backtrack(x: torch.Tensor, t: float, dt: float, gamma: float, noise: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Noise backtracking (Eq. 18): move x_t back to an earlier time s = a t with fresh noise.

    If x_t | x_1 ~ N(t x_1, σ² (1 - t)²) and `noise` ~ N(0, σ²), the result has law N(s x_1, σ² (1 - s)²).
    """
    a = min(1.0, max(0.0, 1.0 - gamma * dt))
    s = a * t
    return a * x + max(0.0, (1 - s) ** 2 - (a - s) ** 2) ** 0.5 * noise, s


@torch.no_grad()
def sample(model: Denoiser, context: torch.Tensor, position: torch.Tensor, cfg: Config, samples: int | None = None,
           flow_steps: int | None = None, gamma: float | None = None,
           generator: torch.Generator | None = None) -> Samples:
    """Sample K trajectories per decision with the stochastic sampler of Algorithm 2 (γ = 0: Euler, Eq. 8).

    context: (B, lookback, F); position: (B,). The recurrent state is carried across sampler steps.
    """
    samples = samples if samples is not None else cfg.inference.samples
    steps = flow_steps if flow_steps is not None else cfg.inference.flow_steps
    gamma = gamma if gamma is not None else cfg.inference.gamma
    batch, device = context.shape[0], context.device
    shape = (batch * samples, model.horizon, model.num_levels)
    sigma = cfg.flow.noise_scale if cfg.flow.noise_scale is not None else 1 / model.num_levels ** 0.5

    context = context.repeat_interleave(samples, dim=0)
    position = position.repeat_interleave(samples, dim=0)
    x = torch.randn(shape, generator=generator, device=device) * sigma
    state = None
    for i in range(steps):
        t, t_next = i / steps, (i + 1) / steps
        noise = torch.randn(shape, generator=generator, device=device) * sigma if gamma > 0 \
            else torch.zeros(shape, device=device)
        x_bar, s = backtrack(x, t, t_next - t, gamma, noise)
        out = model(context, position, x_bar, torch.full((shape[0],), s, device=device), state)
        state = out.state
        prediction = level_log_probs(out.logits, cfg.train.loss).exp().to(x.dtype)
        x = x_bar + (t_next - s) * (prediction - x_bar) / (1 - s)

    return Samples(prediction.view(batch, samples, *shape[1:]), x.argmax(dim=-1).view(batch, samples, -1),
                   out.q_logit.view(batch, samples))


@torch.no_grad()
def predict(model: Denoiser, context: torch.Tensor, position: torch.Tensor, cfg: Config,
            generator: torch.Generator | None = None, **overrides) -> Samples:
    """Trajectory samples for the configured training method: the direct predictor makes one deterministic
    prediction (K = 1), the looped flow runs the sampler."""
    if cfg.train.method == "direct":
        batch = context.shape[0]
        empty = torch.zeros(batch, model.horizon, model.num_levels, device=context.device)
        out = model(context, position, empty, torch.zeros(batch, device=context.device))
        probs = level_log_probs(out.logits, cfg.train.loss).exp().to(torch.float32)
        return Samples(probs[:, None], out.logits.argmax(dim=-1)[:, None], out.q_logit[:, None])
    return sample(model, context, position, cfg, generator=generator, **overrides)
