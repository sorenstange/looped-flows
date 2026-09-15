import pytest
import torch

from src.config import load_config
from src.features import FEATURE_NAMES
from src.modules import Denoiser, RecurrentState

LOOKBACK, HORIZON, LEVELS = 16, 8, 21
TINY = [f"features.lookback={LOOKBACK}", f"oracle.horizon={HORIZON}", "model.width=32", "model.heads=4",
        "model.mlp_width=64", "model.cycles=3", "model.inner_steps=2", "model.patch_size=4"]


def tiny_model(extra=()) -> tuple[Denoiser, object]:
    cfg = load_config(overrides=TINY + list(extra))
    torch.manual_seed(0)
    return Denoiser(cfg), cfg


def inputs(batch: int = 3, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    context = torch.randn(batch, LOOKBACK, len(FEATURE_NAMES), generator=gen)
    position = torch.rand(batch, generator=gen) * 2 - 1
    target = torch.nn.functional.one_hot(torch.randint(0, LEVELS, (batch, HORIZON), generator=gen), LEVELS).float()
    t = torch.rand(batch, generator=gen)
    noisy = (1 - t)[:, None, None] * torch.randn(batch, HORIZON, LEVELS, generator=gen) / LEVELS ** 0.5 \
        + t[:, None, None] * target
    return context, position, noisy, t


def test_output_shapes():
    model, _ = tiny_model()
    out = model(*inputs())
    assert out.logits.shape == (3, HORIZON, LEVELS)
    assert out.q_logit.shape == (3,)
    assert model.num_tokens == 1 + LOOKBACK // 4 + 1 + HORIZON
    assert out.state.h.shape == out.state.l.shape == (3, model.num_tokens, 32)
    assert not out.state.h.requires_grad and not out.state.l.requires_grad
    assert torch.allclose(out.q_logit, torch.full((3,), -5.0))  # confidence head starts near zero


def test_paper_size_parameter_count():
    model = Denoiser(load_config())
    params = sum(p.numel() for p in model.parameters())
    assert 6.5e6 < params < 7.5e6  # the paper's ~7M-parameter transformer
    assert model.num_tokens == 1 + 256 + 1 + 64


def test_gradients_flow_only_through_the_last_cycle():
    model, cfg = tiny_model()
    grad_modes = []
    model.blocks[0].register_forward_hook(lambda *_: grad_modes.append(torch.is_grad_enabled()))
    out = model(*inputs())
    calls_per_cycle = cfg.model.inner_steps + 1
    assert grad_modes == [False] * (cfg.model.cycles - 1) * calls_per_cycle + [True] * calls_per_cycle

    (out.logits.square().mean() + out.q_logit.mean()).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name


def test_incoming_state_is_detached():
    model, _ = tiny_model()
    args = inputs()
    state = model(*args).state
    h = state.h.clone().requires_grad_(True)
    l = state.l.clone().requires_grad_(True)
    out = model(*args, state=RecurrentState(h, l))
    out.logits.sum().backward()
    assert h.grad is None and l.grad is None


def test_recurrent_state_changes_the_prediction():
    model, _ = tiny_model()
    args = inputs()
    first = model(*args)
    second = model(*args, state=first.state)
    assert not torch.allclose(first.logits, second.logits)


def test_samples_do_not_interact():
    model, _ = tiny_model()
    context, position, noisy, t = inputs()
    base = model(context, position, noisy, t).logits
    context2, noisy2 = context.clone(), noisy.clone()
    context2[1] += 1.0
    noisy2[2] = torch.flip(noisy2[2], dims=[0])
    changed = model(context2, position, noisy2, t).logits
    assert torch.allclose(base[0], changed[0], atol=1e-6)
    assert not torch.allclose(base[1], changed[1]) and not torch.allclose(base[2], changed[2])


@pytest.mark.parametrize("which", ["context", "position", "noisy", "t"])
def test_every_input_affects_the_output(which):
    model, _ = tiny_model()
    args = dict(zip(["context", "position", "noisy", "t"], inputs()))
    base = model(**args).logits
    changed = dict(args)
    changed[which] = args[which] + 0.5
    assert not torch.allclose(base, model(**changed).logits)


def test_trajectory_order_matters():
    model, _ = tiny_model()
    context, position, noisy, t = inputs(batch=1)
    base = model(context, position, noisy, t).logits
    reversed_out = model(context, position, torch.flip(noisy, dims=[1]), t).logits
    assert not torch.allclose(base, torch.flip(reversed_out, dims=[1]), atol=1e-4)  # RoPE breaks permutation symmetry


def test_patch_size_must_divide_lookback():
    with pytest.raises(ValueError):
        load_config(overrides=TINY + ["model.patch_size=3"])
    with pytest.raises(ValueError):
        load_config(overrides=TINY + ["model.heads=3"])
