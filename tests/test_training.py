import math

import pytest
import torch

from src.config import config_from_dict, config_to_dict, load_config
from src.data import clean_ohlcv
from src.dataset import MarketData
from src.losses import level_cross_entropy, level_log_probs, log_stablemax
from src.optim import AdamAtan2, lr_factor
from src.training import direct_forward, load_model, train
from tests.conftest import make_bars

TINY = ["features.lookback=16", "features.rolling_window=100", "oracle.horizon=8", "model.width=32",
        "model.heads=4", "model.mlp_width=64", "model.inner_steps=2", "model.patch_size=4", "loader.batch_size=16",
        "splits.val_start=2024-02-15", "splits.test_start=2024-03-01", "train.device=cpu", "train.warmup_steps=5",
        "train.log_every=50", "train.eval_batches=2"]


def test_stablemax():
    logits = torch.tensor([[-3.0, 0.0, 2.0]])
    log_probs = log_stablemax(logits)
    s = torch.tensor([1 / 4, 1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(log_probs.exp(), (s / s.sum())[None])
    target = torch.tensor([[2]])
    ce = level_cross_entropy(logits[:, None, :], target, "stablemax")
    assert ce.item() == pytest.approx(-math.log(3 / 4.25))
    assert torch.allclose(level_log_probs(logits, "softmax").exp().sum(), torch.tensor(1.0, dtype=torch.float64))


def test_stablemax_gradient_is_finite_at_one():
    logits = torch.tensor([[[1.0, -0.5, 0.0, 1.0 + 1e-12, 3.0]]], dtype=torch.float64, requires_grad=True)
    level_cross_entropy(logits, torch.tensor([[0]]), "stablemax").backward()
    assert torch.isfinite(logits.grad).all()


def test_adam_atan2_minimizes_and_bounds_updates():
    param = torch.nn.Parameter(torch.tensor([5.0, -3.0]))
    optimizer = AdamAtan2([param], lr=0.05)
    previous = param.detach().clone()
    for _ in range(500):
        optimizer.zero_grad()
        (param ** 2).sum().backward()
        optimizer.step()
        assert (param.detach() - previous).abs().max() <= 0.05 * 1.27 * math.pi / 2 + 1e-6
        previous = param.detach().clone()
    assert param.detach().abs().max() < 0.1


def test_lr_factor():
    cfg = load_config(overrides=["train.warmup_steps=10", "train.max_steps=110", "train.lr_schedule=cosine",
                                 "train.min_lr_ratio=0.1"]).train
    assert lr_factor(0, cfg) == pytest.approx(0.1)
    assert lr_factor(9, cfg) == pytest.approx(1.0)
    assert lr_factor(10, cfg) == pytest.approx(1.0)
    assert lr_factor(60, cfg) == pytest.approx(0.55)
    assert lr_factor(110, cfg) == pytest.approx(0.1)
    assert lr_factor(500, load_config(overrides=["train.warmup_steps=10"]).train) == 1.0


def test_config_dict_roundtrip():
    cfg = load_config("configs/smoke.yaml")
    assert config_from_dict(config_to_dict(cfg)) == cfg


def tiny_market(cfg):
    return MarketData.from_bars(clean_ohlcv(make_bars(), "1h"), cfg)


def test_direct_predictor_overfits_a_small_subset(tmp_path):
    cfg = load_config(overrides=TINY + ["train.overfit_samples=16", "train.max_steps=400", "train.lr=3e-3",
                                        "train.ema_decay=null", "train.eval_every=400", "train.checkpoint_every=400",
                                        "train.optimizer=adamw", "train.weight_decay=0.0"])
    result = train(cfg, market=tiny_market(cfg), out_dir=tmp_path)

    train_logs = [entry for entry in result.history if entry["split"] == "train"]
    final = [entry for entry in result.history if entry["split"] == "train_subset"][-1]
    assert train_logs[0]["ce"] > 2.0  # starts near log(21) = 3.04
    assert all(later["ce"] < earlier["ce"] for earlier, later in zip(train_logs, train_logs[1:]))
    assert final["ce"] < 0.4 * train_logs[0]["ce"] and final["acc_step"] > 0.75, final
    assert final["acc_step"] > 3 * final["acc_hold"]  # far beyond "keep the current position"
    assert (tmp_path / "metrics.jsonl").exists() and (tmp_path / "config.yaml").exists()


def test_checkpoint_roundtrip(tmp_path):
    cfg = load_config(overrides=TINY + ["train.max_steps=3", "train.eval_every=3", "train.checkpoint_every=3"])
    market = tiny_market(cfg)
    train(cfg, market=market, out_dir=tmp_path)
    checkpoint = tmp_path / "checkpoints" / "step_0000003.pt"
    loaded_cfg, model = load_model(checkpoint)
    assert loaded_cfg == cfg

    from src.dataset import build_datasets
    batch = build_datasets(cfg, market)["val"][list(range(4))]
    raw_checkpoint = torch.load(checkpoint, weights_only=False)
    _, raw_model = load_model(checkpoint, use_ema=False)
    with torch.no_grad():
        ema_out = direct_forward(model, batch).logits
        raw_out = direct_forward(raw_model, batch).logits
    assert raw_checkpoint["ema"] is not None and not torch.allclose(ema_out, raw_out)
    assert torch.isfinite(ema_out).all()
