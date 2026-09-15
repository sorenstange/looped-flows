"""Categorical losses over allocation levels."""

import torch
import torch.nn.functional as F


def log_stablemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log of StableMax (Prieto et al., 2025): s(x) = x + 1 for x >= 0, 1 / (1 - x) otherwise, normalized.

    Grows linearly instead of exponentially, which avoids softmax collapse; computed in float64 as in TRM.
    The negative branch only ever sees non-positive inputs: `torch.where` differentiates both branches, and 1 / (1 - x)
    at x = 1 would turn the gradient into NaN even though that branch is not selected.
    """
    x = logits.to(torch.float64)
    positive = x >= 0
    s = torch.where(positive, x + 1, 1 / (1 - torch.where(positive, torch.zeros_like(x), x)))
    return torch.log(s) - torch.log(s.sum(dim=dim, keepdim=True))


def level_log_probs(logits: torch.Tensor, loss: str) -> torch.Tensor:
    """Log probabilities over levels, normalized consistently with the training loss."""
    if loss == "stablemax":
        return log_stablemax(logits)
    if loss == "softmax":
        return F.log_softmax(logits.to(torch.float64), dim=-1)
    raise ValueError(f"unknown loss {loss!r}")


def level_cross_entropy(logits: torch.Tensor, target: torch.Tensor, loss: str) -> torch.Tensor:
    """Mean cross-entropy of (B, H, K) logits against (B, H) level indices."""
    log_probs = level_log_probs(logits, loss)
    return -log_probs.gather(-1, target[..., None]).mean()
