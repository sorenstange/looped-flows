import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data import clean_ohlcv
from src.dataset import MarketData, build_datasets, make_loader, split_ranges, valid_anchors
from src.oracle import oracle_trajectory
from tests.conftest import make_bars

LOOKBACK, HORIZON = 32, 8
OVERRIDES = [f"features.lookback={LOOKBACK}", f"oracle.horizon={HORIZON}", "splits.val_start=2024-02-15",
             "splits.test_start=2024-03-01"]


def make_market(bars: pd.DataFrame) -> MarketData:
    return MarketData.from_bars(clean_ohlcv(bars, "1h"), "simple")


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

    assert batch["context"].shape == (64, LOOKBACK, 5) and batch["context"].dtype == torch.float32
    assert batch["target"].shape == (64, HORIZON) and batch["target"].dtype == torch.long
    assert torch.isfinite(batch["context"]).all() and torch.isfinite(batch["returns"]).all()
    assert (batch["position"].abs() <= 1).all()
    volume = batch["context"][..., 4]
    assert torch.allclose(volume.mean(dim=1), torch.zeros(64), atol=1e-5)

    anchor = batch["anchor"][3].item()
    expected_returns = market.returns[anchor + 1:anchor + 1 + HORIZON]
    np.testing.assert_allclose(batch["returns"][3].numpy(), expected_returns, rtol=1e-6)
    path, _ = oracle_trajectory(torch.from_numpy(expected_returns)[None], batch["position"][3:4].double(),
                                dataset.levels, cfg.oracle.cost, cfg.oracle.max_step)
    assert torch.equal(batch["target"][3], path[0])

    positions = torch.cat([batch["position"][:, None].double(), dataset.levels[batch["target"]]], dim=1)
    assert positions.diff(dim=1).abs().max() <= cfg.oracle.max_step + 1e-6

    single = dataset[3]
    assert single["context"].shape == (LOOKBACK, 5) and torch.equal(single["target"], batch["target"][3])


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
