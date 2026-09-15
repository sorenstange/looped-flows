"""Stationary per-bar input features."""

from typing import Sequence

import numpy as np
import pandas as pd
import torch

FEATURE_NAMES = ("log_return", "log_high_close", "log_low_close", "taker_buy_share", "log_volume", "log_vol_to_cost")
PRICE_FEATURES = FEATURE_NAMES[:3]  # scaled by the trailing RMS of log returns
ACTIVITY_FEATURES = FEATURE_NAMES[3:5]  # z-scored (trailing in rolling mode, per sample window in window mode)
MIN_COST = 1e-5  # floor for the cost in log_vol_to_cost, so a zero-cost config stays finite
EPS = 1e-8

# No open-based feature: on Binance perpetuals each bar opens at the previous close, so log(open / close) would
# equal -log_return.


def raw_bar_features(bars: pd.DataFrame) -> np.ndarray:
    """Unscaled (T, 5) features: log return, high and low relative to close (log ratios), taker-buy share of volume
    (0.5 for bars without volume), log1p volume.

    Row t only uses bars t-1 and t; row 0 has no previous close and holds NaN in `log_return`.
    """
    volume = bars["volume"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(bars["close"].to_numpy())
        taker_buy_share = np.where(volume > 0, bars["taker_buy_volume"].to_numpy() / volume, 0.5)
        return np.stack([
            np.concatenate([[np.nan], np.diff(log_close)]),
            np.log(bars["high"].to_numpy()) - log_close,
            np.log(bars["low"].to_numpy()) - log_close,
            taker_buy_share,
            np.log1p(volume),
        ], axis=1)


def bar_features(bars: pd.DataFrame, scaling: str, rolling_window: int, cost: float) -> np.ndarray:
    """Model input features of shape (T, len(FEATURE_NAMES)), float32. Every row t only uses bars <= t.

    - scaling "rolling": the price features are divided by the trailing RMS of log returns over `rolling_window` bars
      (no mean subtraction, so drift is kept); taker-buy share and log volume are z-scored with trailing mean and std
      over the same window.
    - scaling "window": all features stay unscaled here; the dataset z-scores the activity features per sample window
      (kept for comparison).
    - `log_vol_to_cost` = log(trailing RMS / cost) in both modes: how large typical moves are relative to the cost.

    The trailing statistics include bar t itself and are NaN until `rolling_window` returns are available, so the
    first bars of the data are unusable (warmup).
    """
    raw = raw_bar_features(bars)
    log_return = pd.Series(raw[:, 0])
    rms = np.sqrt((log_return ** 2).rolling(rolling_window, min_periods=rolling_window).mean().to_numpy())
    features = np.empty((len(raw), len(FEATURE_NAMES)))
    with np.errstate(divide="ignore", invalid="ignore"):
        if scaling == "rolling":
            features[:, :3] = raw[:, :3] / rms[:, None]
            activity = pd.DataFrame(raw[:, 3:5]).rolling(rolling_window, min_periods=rolling_window)
            features[:, 3:5] = (raw[:, 3:5] - activity.mean().to_numpy()) / (activity.std().to_numpy() + EPS)
        elif scaling == "window":
            features[:, :5] = raw
        else:
            raise ValueError(f"unknown feature scaling {scaling!r}")
        features[:, 5] = np.log(rms / max(cost, MIN_COST))
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
