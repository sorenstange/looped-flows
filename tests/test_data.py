import numpy as np
import pandas as pd
import pytest

from src.data import clean_ohlcv, interval_to_timedelta
from src.features import FEATURE_NAMES, bar_features


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


def test_features_are_causal(bars):
    features = bar_features(bars)
    changed = bars.copy()
    changed.iloc[500:, :] *= 1.7
    changed_features = bar_features(changed)
    np.testing.assert_array_equal(features[:500], changed_features[:500])
    assert np.isnan(features[0, FEATURE_NAMES.index("log_return")])
    assert np.isfinite(features[1:]).all()
