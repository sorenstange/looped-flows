import math

import pytest
import torch

from src.config import config_from_dict, config_to_dict, load_config
from src.data import clean_ohlcv
from src.dataset import MarketData
from src.losses import level_cross_entropy, level_log_probs, log_stablemax
from src.optim import AdamAtan2, lr_factor
from src.training import (Trainer, direct_forward, interpolant, load_model, pseudotarget_probability,
                          sample_flow_times, train)
from tests.conftest import make_bars

TINY = ["features.lookback=16", "features.rolling_window=100", "oracle.horizon=8", "model.width=32",
        "model.heads=4", "model.mlp_width=64", "model.inner_steps=2", "model.patch_size=4", "loader.batch_size=16",
        "splits.val_start=2024-02-15", "splits.test_start=2024-03-01", "train.device=cpu", "train.warmup_steps=5",
        "train.log_every=50", "train.eval_batches=2", "train.sample_eval_batches=1", "inference.samples=2",
        "inference.flow_steps=4"]


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


@pytest.mark.parametrize("sampler", ["sorted", "random_start"])
def test_sample_flow_times(sampler):
    times = sample_flow_times(500, 6, sampler, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert times.shape == (500, 6)
    assert (times.diff(dim=1) >= 0).all() and (times >= 0).all() and (times < 1).all()
    if sampler == "random_start":  # t_0 is larger in expectation than under the sorted sampler
        sorted_times = sample_flow_times(500, 6, "sorted", torch.device("cpu"), torch.Generator().manual_seed(0))
        assert times[:, 0].mean() > sorted_times[:, 0].mean() + 0.2


def test_interpolant():
    x0, x1 = torch.full((2, 1, 3), 2.0), torch.ones(2, 1, 3)
    out = interpolant(x0, x1, torch.tensor([0.0, 0.75]))
    assert torch.allclose(out[0], x0[0]) and torch.allclose(out[1], torch.full((1, 3), 1.25))


def test_pseudotarget_probability():
    cfg = load_config(overrides=["flow.pseudotargets=true", "flow.pseudotarget_ramp_steps=100"])
    assert [pseudotarget_probability(s, cfg) for s in (0, 50, 100, 500)] == [0.0, 0.5, 1.0, 1.0]
    assert pseudotarget_probability(500, load_config()) == 0.0


FLOW = TINY + ["train.method=looped_flow", "flow.steps=4", "train.ema_decay=null"]


def record_denoiser_inputs(model):
    """(noisy input, flow time, starts a rollout) of every training-mode denoiser call (evaluation is skipped)."""
    calls = []

    def hook(module, args):
        if module.training:
            calls.append((args[2].detach().clone(), args[3].clone(), args[4] is None))

    model.register_forward_pre_hook(hook)
    return calls


@pytest.mark.parametrize("share_noise", [True, False])
def test_looped_flow_rollout_shares_noise_across_steps(tmp_path, share_noise):
    cfg = load_config(overrides=FLOW + ["train.max_steps=4", "act.halting=false", "train.eval_every=1000",
                                        "train.checkpoint_every=1000", f"flow.share_noise={share_noise}"])
    trainer = Trainer(cfg, tiny_market(cfg), tmp_path)
    calls = record_denoiser_inputs(trainer.model)
    trainer.fit()
    rollout = calls[:4]
    assert [starts for _, _, starts in rollout] == [True, False, False, False]
    (i0, t0, _), (i1, t1, _) = rollout[0], rollout[1]
    t0, t1 = t0[:, None, None].double(), t1[:, None, None].double()
    # Eq. 16: with shared noise, two interpolants recover the clean one-hot target exactly.
    x1 = ((1 - t0) * i1.double() - (1 - t1) * i0.double()) / (t1 - t0)
    is_one_hot = torch.allclose(x1, torch.nn.functional.one_hot(x1.argmax(-1), x1.shape[-1]).double(), atol=1e-3)
    assert is_one_hot == share_noise


def test_looped_flow_halts_when_confident(tmp_path):
    cfg = load_config(overrides=FLOW + ["train.max_steps=5", "act.exploration_prob=0.0", "train.lr=1e-6",
                                        "train.eval_every=1000", "train.checkpoint_every=1000"])
    trainer = Trainer(cfg, tiny_market(cfg), tmp_path)
    with torch.no_grad():
        trainer.model.q_head.bias.fill_(10.0)  # q > 1/2 for every sample: halt after the first step
    calls = record_denoiser_inputs(trainer.model)
    trainer.fit()
    assert len(calls) == 5 and all(starts for _, _, starts in calls)  # every optimizer step starts a new rollout


def test_looped_flow_learns_to_read_the_interpolant(tmp_path):
    cfg = load_config(overrides=FLOW + ["train.overfit_samples=16", "train.max_steps=200", "train.lr=3e-3",
                                        "train.optimizer=adamw", "train.weight_decay=0.0", "train.loss=softmax",
                                        "act.halting=false", "train.eval_every=200", "train.checkpoint_every=200"])
    result = train(cfg, market=tiny_market(cfg), out_dir=tmp_path)
    final = [entry for entry in result.history if entry["split"] == "train_subset"][-1]
    train_logs = [entry for entry in result.history if entry["split"] == "train"]
    assert "active_frac" in train_logs[0] and "flow_t" in train_logs[0]
    assert {"sample_mae_first", "sample_pnl", "sample_pnl_vs_oracle", "sample_spread_first"} <= final.keys()
    assert final["sample_spread_first"] > 0  # different noise -> different samples
    # At t = 3/4 the interpolant nearly reveals the target, at t = 0 it is pure noise.
    assert final["ce_last_t"] < 0.75 * final["ce_first_t"], final


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
