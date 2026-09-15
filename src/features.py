"""Stationary per-bar input features."""

from typing import Sequence

import numpy as np
import pandas as pd
import torch

FEATURE_NAMES = ("log_return", "log_open_close", "log_high_close", "log_low_close", "log_volume")


def bar_features(bars: pd.DataFrame) -> np.ndarray:
    """Features of shape (T, len(FEATURE_NAMES)), float32.

    Row t only uses bars t-1 and t, so it is known at the close of bar t. Row 0 has no previous close and holds NaN
    in `log_return`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(bars["close"].to_numpy())
        features = np.stack([
            np.concatenate([[np.nan], np.diff(log_close)]),
            np.log(bars["open"].to_numpy()) - log_close,
            np.log(bars["high"].to_numpy()) - log_close,
            np.log(bars["low"].to_numpy()) - log_close,
            np.log1p(bars["volume"].to_numpy()),
        ], axis=1)
    return features.astype(np.float32)


def standardize_windows(windows: torch.Tensor, channels: Sequence[int], eps: float = 1e-6) -> torch.Tensor:
    """Z-score the given channels of (B, L, F) windows over the time axis, each window using only its own bars."""
    if not channels:
        return windows
    out = windows.clone()
    selected = windows[..., list(channels)]
    mean = selected.mean(dim=1, keepdim=True)
    std = selected.std(dim=1, keepdim=True)
    out[..., list(channels)] = (selected - mean) / (std + eps)
    return out
