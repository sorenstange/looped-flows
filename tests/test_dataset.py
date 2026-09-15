import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data import clean_ohlcv
from src.dataset import MarketData, build_datasets, make_loader, split_ranges, valid_anchors
from src.features import FEATURE_NAMES
from src.oracle import oracle_trajectory

LOOKBACK, HORIZON, ROLLING = 32, 8, 100
OVERRIDES = [f"features.lookback={LOOKBACK}", f"oracle.horizon={HORIZON}", f"features.rolling_window={ROLLING}",
             "splits.val_start=2024-02-15", "splits.test_start=2024-03-01"]


def make_market(bars: pd.DataFrame, overrides=()) -> MarketData:
    return MarketData.from_bars(clean_ohlcv(bars, "1h"), load_config(overrides=OVERRIDES + list(overrides)))


def test_valid_anchors_avoid_invalid_bars():
    valid = np.ones(100, dtype=bool)
    valid[50] = False
    anchors = valid_anchors(valid, lookback=10, horizon=5, lo=0, hi=100)
    assert anchors.min() == 10 and anchors.max() == 94
    assert all(not (a - 10 <= 50 <= a + 5) for a in anchors)
    assert set(range(10, 45)) | set(range(61, 95)) == set(anchors.tolist())


def test_splits_are_disjoint_with_purge_gap(bars):
    cfg = load_config(overrides=OVERRIDES)
    market = make_market(bars)
    datasets = build_datasets(cfg, market)
    spans = {name: (ds.anchors.min() - LOOKBACK, ds.anchors.max() + HORIZON) for name, ds in datasets.items()}
    ranges = split_ranges(market.times, cfg.splits, LOOKBACK, HORIZON)
    assert ranges["val"][0] - ranges["train"][1] == LOOKBACK + HORIZON
    assert spans["val"][0] - spans["train"][1] > LOOKBACK + HORIZON
    assert spans["test"][0] - spans["val"][1] > LOOKBACK + HORIZON
    assert market.times[spans["val"][1]] < pd.Timestamp("2024-03-01", tz="UTC")


def test_batch_contents(bars):
    cfg = load_config(overrides=OVERRIDES)
    market = make_market(bars)
    dataset = build_datasets(cfg, market)["train"]
    batch = dataset[list(range(0, 64))]

    features = len(FEATURE_NAMES)
    assert batch["context"].shape == (64, LOOKBACK, features) and batch["context"].dtype == torch.float32
    assert batch["target"].shape == (64, HORIZON) and batch["target"].dtype == torch.long
    assert torch.isfinite(batch["context"]).all() and torch.isfinite(batch["returns"]).all()
    assert batch["context"].abs().max() <= cfg.features.clip
    assert (batch["position"].abs() <= 1).all()
    assert dataset.anchors.min() - LOOKBACK >= ROLLING  # the warmup of the trailing statistics is never used

    anchor = batch["anchor"][3].item()
    expected_returns = market.returns[anchor + 1:anchor + 1 + HORIZON]
    np.testing.assert_allclose(batch["returns"][3].numpy(), expected_returns, rtol=1e-6)
    path, _ = oracle_trajectory(torch.from_numpy(expected_returns)[None], batch["position"][3:4].double(),
                                dataset.levels, cfg.oracle.cost, cfg.oracle.max_step)
    assert torch.equal(batch["target"][3], path[0])

    positions = torch.cat([batch["position"][:, None].double(), dataset.levels[batch["target"]]], dim=1)
    assert positions.diff(dim=1).abs().max() <= cfg.oracle.max_step + 1e-6

    single = dataset[3]
    assert single["context"].shape == (LOOKBACK, features) and torch.equal(single["target"], batch["target"][3])


def test_window_scaling_standardizes_activity_per_window(bars):
    overrides = ["features.scaling=window"]
    cfg = load_config(overrides=OVERRIDES + overrides)
    dataset = build_datasets(cfg, make_market(bars, overrides))["train"]
    context = dataset[list(range(64))]["context"]
    for name in ("taker_buy_share", "log_volume"):
        channel = context[..., FEATURE_NAMES.index(name)]
        assert torch.allclose(channel.mean(dim=1), torch.zeros(64), atol=1e-5)
    log_return = context[..., FEATURE_NAMES.index("log_return")]
    assert log_return.abs().max() < 0.1  # price features stay unscaled


def test_clipping(bars):
    bars = bars.copy()
    bars.iloc[1500, bars.columns.get_loc("close")] *= 1.5  # a flash spike far beyond 10 trailing RMS
    bars.iloc[1500, bars.columns.get_loc("high")] = bars["close"].iloc[1500]
    # A value can be at most sqrt(window) times the trailing RMS that includes it, so use a window where 10 is reachable.
    overrides = ["features.rolling_window=400"]
    cfg = load_config(overrides=OVERRIDES + overrides)
    dataset = build_datasets(cfg, make_market(bars, overrides))["test"]
    position = int(np.searchsorted(dataset.anchors, 1500))
    assert dataset.anchors[position] - LOOKBACK < 1500 <= dataset.anchors[position]
    assert dataset[position]["context"].abs().max() == cfg.features.clip


def test_context_has_no_lookahead(bars):
    cfg = load_config(overrides=OVERRIDES)
    dataset = build_datasets(cfg, make_market(bars))["train"]
    anchor = int(dataset.anchors[100])

    changed = bars.copy()
    changed.iloc[anchor + 1:, :4] *= 1.3
    changed.iloc[anchor + 1:, 4] *= 5.0
    changed_dataset = build_datasets(cfg, make_market(changed))["train"]

    assert changed_dataset.anchors[100] == anchor
    assert torch.equal(dataset[100]["context"], changed_dataset[100]["context"])
    assert not torch.equal(dataset[100]["returns"], changed_dataset[100]["returns"])


def test_initial_positions_reproducible_and_redrawn_per_epoch(bars):
    cfg = load_config(overrides=OVERRIDES)
    dataset = build_datasets(cfg, make_market(bars))["train"]
    first = dataset[list(range(200))]["position"]
    assert torch.equal(first, dataset[list(range(200))]["position"])
    assert torch.equal(first[50:100], dataset[list(range(50, 100))]["position"])
    assert first.min() < -0.8 and first.max() > 0.8
    dataset.set_epoch(1)
    assert not torch.equal(first, dataset[list(range(200))]["position"])


def test_loader_covers_dataset_once(bars):
    cfg = load_config(overrides=OVERRIDES)
    dataset = build_datasets(cfg, make_market(bars))["val"]
    loader = make_loader(dataset, batch_size=50, shuffle=True, seed=1)
    anchors = torch.cat([batch["anchor"] for batch in loader])
    assert sorted(anchors.tolist()) == sorted(dataset.anchors.tolist())
