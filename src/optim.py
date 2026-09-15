"""Optimizers and learning-rate schedules."""

import math

import torch

from src.config import TrainConfig


class AdamAtan2(torch.optim.Optimizer):
    """Adam with atan2 in place of the epsilon-guarded division (Everett et al., 2024), decoupled weight decay.

    update = a * atan2(m_hat, b * sqrt(v_hat)); scale-invariant and needs no epsilon.
    """

    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 0.0,
                 a: float = 1.27, b: float = 1.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                grad = param.grad
                param.mul_(1 - group["lr"] * group["weight_decay"])
                state["exp_avg"].lerp_(grad, 1 - beta1)
                state["exp_avg_sq"].lerp_(grad * grad, 1 - beta2)
                m_hat = state["exp_avg"] / (1 - beta1 ** state["step"])
                v_hat = state["exp_avg_sq"] / (1 - beta2 ** state["step"])
                param.add_(torch.atan2(m_hat, group["b"] * v_hat.sqrt()), alpha=-group["lr"] * group["a"])
        return loss


def make_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    betas = (cfg.betas[0], cfg.betas[1])
    if cfg.optimizer == "adam_atan2":
        return AdamAtan2(params, lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def lr_factor(step: int, cfg: TrainConfig) -> float:
    """Multiplier of the base learning rate at optimizer step `step` (0-based): linear warmup, then constant or cosine."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.lr_schedule == "constant":
        return 1.0
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
