import numpy as np
import pandas as pd
import pytest

from src.data import clean_ohlcv, interval_to_timedelta
from src.features import FEATURE_NAMES, bar_features, raw_bar_features


@pytest.mark.parametrize("interval, expected", [("1m", "1min"), ("15m", "15min"), ("4h", "4h"), ("1d", "1D"),
                                                ("1w", "7D")])
def test_interval_to_timedelta(interval, expected):
    assert interval_to_timedelta(interval) == pd.Timedelta(expected)


@pytest.mark.parametrize("interval", ["1M", "h", "0h", "1.5h"])
def test_interval_to_timedelta_rejects(interval):
    with pytest.raises(ValueError):
        interval_to_timedelta(interval)


def test_clean_fills_gaps_and_flags_bad_bars(bars):
    bars.iloc[20, bars.columns.get_loc("high")] = bars["low"].iloc[20] * 0.5  # high below low
    bars = bars.drop(bars.index[[10, 11]])
    bars = pd.concat([bars, bars.iloc[[30]]])  # duplicate row

    clean = clean_ohlcv(bars, "1h")

    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert (clean.index[1:] - clean.index[:-1] == pd.Timedelta("1h")).all()
    assert not clean["valid"].iloc[[10, 11, 20]].any()
    assert clean["valid"].drop(clean.index[[10, 11, 20]]).all()
    filled = clean.iloc[10]
    assert filled["open"] == filled["high"] == filled["low"] == filled["close"] == clean["close"].iloc[9]
    assert filled["volume"] == 0


@pytest.mark.parametrize("scaling", ["rolling", "window"])
def test_features_are_causal_with_warmup(bars, scaling):
    window = 100
    features = bar_features(bars, scaling, window, cost=0.0005)
    changed = bars.copy()
    changed.iloc[500:, :] *= 1.7
    changed.iloc[500:, changed.columns.get_loc("volume")] *= 3.0
    np.testing.assert_array_equal(features[:500], bar_features(changed, scaling, window, cost=0.0005)[:500])
    assert features.shape == (len(bars), len(FEATURE_NAMES)) and features.dtype == np.float32
    assert np.isnan(features[:window]).any(axis=1).all()  # warmup: trailing statistics need `window` returns
    assert np.isfinite(features[window:]).all()


def test_rolling_scaling_values(bars):
    window, cost = 50, 0.0005
    features = bar_features(bars, "rolling", window, cost)
    raw = raw_bar_features(bars)
    t = 300
    rms = np.sqrt(np.mean(raw[t - window + 1:t + 1, 0] ** 2))  # trailing window includes bar t
    np.testing.assert_allclose(features[t, :3], raw[t, :3] / rms, rtol=1e-5)
    for i, name in ((3, "taker_buy_share"), (4, "log_volume")):
        assert FEATURE_NAMES[i] == name
        past = raw[t - window + 1:t + 1, i]
        np.testing.assert_allclose(features[t, i], (raw[t, i] - past.mean()) / past.std(ddof=1), rtol=1e-4)
    np.testing.assert_allclose(features[t, 5], np.log(rms / cost), rtol=1e-5)
    np.testing.assert_allclose(raw[t, 3], bars["taker_buy_volume"].iloc[t] / bars["volume"].iloc[t])


def test_taker_buy_share_is_neutral_without_volume(bars):
    bars = bars.copy()
    bars.iloc[10, bars.columns.get_loc("volume")] = 0.0
    bars.iloc[10, bars.columns.get_loc("taker_buy_volume")] = 0.0
    raw = raw_bar_features(bars)
    assert raw[10, FEATURE_NAMES.index("taker_buy_share")] == 0.5
    assert np.isfinite(raw[1:]).all()


def test_rolling_scaling_keeps_drift_and_volatility_level(bars):
    trending = bars.copy()
    trending[["open", "high", "low", "close"]] *= np.exp(np.arange(len(bars)) * 0.002)[:, None]
    window = 100
    base = bar_features(bars, "rolling", window, cost=0.0005)[window:]
    trend = bar_features(trending, "rolling", window, cost=0.0005)[window:]
    assert trend[:, 0].mean() > base[:, 0].mean() + 0.1  # drift is not subtracted
    calm = bars.copy()
    calm[["open", "high", "low", "close"]] = 100 * np.exp(np.log(bars[["open", "high", "low", "close"]] / 100) * 0.1)
    calm_features = bar_features(calm, "rolling", window, cost=0.0005)[window:]
    np.testing.assert_allclose(calm_features[:, 0], base[:, 0], rtol=1e-3, atol=1e-4)  # same scaled returns
    assert calm_features[:, 5].mean() < base[:, 5].mean() - 2  # but a lower volatility level vs the cost
