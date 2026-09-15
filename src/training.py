"""Training loop, losses and metrics for the denoiser.

Methods (`train.method`):
- direct: one denoiser call on an empty trajectory (zeros, flow time 0) predicting the oracle trajectory; the
  direct-predictor baseline with the same backbone.
"""

import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from src.config import Config, config_from_dict, config_to_dict, save_config
from src.dataset import MarketData, WindowDataset, build_datasets, make_loader
from src.losses import level_cross_entropy, level_log_probs
from src.modules import Denoiser, DenoiserOutput
from src.optim import lr_factor, make_optimizer
from src.oracle import make_levels, tolerance_match


@dataclass
class TrainResult:
    out_dir: Path
    history: list[dict] = field(default_factory=list)  # one entry per log / eval event


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def direct_forward(model: Denoiser, batch: dict[str, torch.Tensor]) -> DenoiserOutput:
    """Direct predictor: a single denoiser call with an empty trajectory input."""
    context = batch["context"]
    batch_size = context.shape[0]
    empty = torch.zeros(batch_size, model.horizon, model.num_levels, device=context.device)
    return model(context, batch["position"], empty, torch.zeros(batch_size, device=context.device))


def losses_and_metrics(out: DenoiserOutput, batch: dict[str, torch.Tensor], levels: torch.Tensor,
                       cfg: Config) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Level cross-entropy + weighted confidence BCE, and detached metrics (all batch means)."""
    target = batch["target"]
    ce = level_cross_entropy(out.logits, target, cfg.train.loss)
    predicted = out.logits.argmax(dim=-1)
    q_target = tolerance_match(levels[predicted], levels[target], cfg.act.tolerance, cfg.act.steps).to(torch.float32)
    bce = F.binary_cross_entropy_with_logits(out.q_logit.float(), q_target)
    loss = ce.float() + cfg.act.loss_weight * bce

    with torch.no_grad():
        target_levels = levels[target]
        expected = level_log_probs(out.logits, cfg.train.loss).exp() @ levels  # (B, H) expected level
        hold = ((batch["position"].double() + 1) / (2 / (len(levels) - 1))).round().long()  # level nearest to a_0
        metrics = {
            "loss": loss.detach(),
            "ce": ce.detach(),
            "bce": bce.detach(),
            "acc_step": (predicted == target).double().mean(),
            "acc_first": (predicted[:, 0] == target[:, 0]).double().mean(),
            "mae_expected": (expected - target_levels).abs().mean(),
            "mae_expected_first": (expected[:, 0] - target_levels[:, 0]).abs().mean(),
            "acc_hold": (hold[:, None] == target).double().mean(),  # naive baseline: keep the current position
            "q_target_rate": q_target.double().mean(),
            "q_acc": ((out.q_logit > 0).float() == q_target).double().mean(),
        }
    return loss, metrics


FORWARD = {"direct": direct_forward}


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: WindowDataset, cfg: Config, device: torch.device) -> dict[str, float]:
    """Mean metrics over `train.eval_batches` batches spread evenly over the dataset (a_0 draws of epoch 0)."""
    model.eval()
    levels = make_levels(cfg.oracle.num_levels).to(device)
    count = min(len(dataset), cfg.train.eval_batches * cfg.loader.batch_size)
    positions = np.linspace(0, len(dataset) - 1, count).astype(np.int64)
    totals: dict[str, float] = {}
    for chunk in np.array_split(positions, max(1, int(np.ceil(count / cfg.loader.batch_size)))):
        batch = to_device(dataset[chunk.tolist()], device)
        with autocast(device, cfg.train.precision):
            out = FORWARD[cfg.train.method](model, batch)
        _, metrics = losses_and_metrics(out, batch, levels, cfg)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value.item() * len(chunk)
    return {key: value / count for key, value in totals.items()}


def train(cfg: Config, market: MarketData | None = None, out_dir: str | Path | None = None) -> TrainResult:
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = Path(out_dir) if out_dir is not None else \
        Path(cfg.output_dir) / "train" / f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.train.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    datasets = build_datasets(cfg, market)
    train_set = datasets["train"]
    overfit = cfg.train.overfit_samples is not None
    if overfit:
        train_set = WindowDataset(train_set.market, train_set.anchors[:cfg.train.overfit_samples], cfg,
                                  seed=cfg.loader.seed)
    eval_sets = {"val": datasets["val"]} if not overfit else {"train_subset": train_set}

    model = Denoiser(cfg).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.train.ema_decay)) \
        if cfg.train.ema_decay is not None else None
    optimizer = make_optimizer(model.parameters(), cfg.train)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg.train))
    levels = make_levels(cfg.oracle.num_levels).to(device)
    forward = FORWARD[cfg.train.method]

    result = TrainResult(out_dir)
    log_file = (out_dir / "metrics.jsonl").open("a")
    running: dict[str, torch.Tensor] = {}
    running_steps, tic = 0, time.perf_counter()
    params = sum(p.numel() for p in model.parameters())
    print(f"training {cfg.train.method} ({params / 1e6:.2f}M params) on {device}: {len(train_set)} train samples, "
          f"output {out_dir}")

    def record(entry: dict) -> None:
        result.history.append(entry)
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    step, epoch = 0, 0
    while step < cfg.train.max_steps:
        if not overfit:
            train_set.set_epoch(epoch)  # fresh a_0 draws; overfitting keeps them fixed
        loader = make_loader(train_set, cfg.loader.batch_size, shuffle=True, seed=cfg.loader.seed + epoch,
                             num_workers=cfg.loader.num_workers, drop_last=len(train_set) >= cfg.loader.batch_size)
        for batch in loader:
            model.train()
            batch = to_device(batch, device)
            with autocast(device, cfg.train.precision):
                out = forward(model, batch)
            loss, metrics = losses_and_metrics(out, batch, levels, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.train.grad_clip if cfg.train.grad_clip else float("inf"))
            if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite loss or gradient at step {step + 1}: loss {loss.item()}, "
                                         f"grad norm {grad_norm.item()}")
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1

            metrics["grad_norm"] = grad_norm.detach()
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value.detach().double()
            running_steps += 1

            if step % cfg.train.log_every == 0 or step == cfg.train.max_steps:
                elapsed = time.perf_counter() - tic
                entry = {"step": step, "epoch": epoch, "split": "train", "lr": scheduler.get_last_lr()[0],
                         "steps_per_s": running_steps / elapsed,
                         **{key: (value / running_steps).item() for key, value in running.items()}}
                record(entry)
                print(f"step {step:>6}  loss {entry['loss']:.4f}  ce {entry['ce']:.4f}  acc {entry['acc_step']:.3f} "
                      f"(hold {entry['acc_hold']:.3f})  mae {entry['mae_expected']:.3f}  "
                      f"q {entry['q_target_rate']:.2f}  lr {entry['lr']:.2e}  {entry['steps_per_s']:.1f} it/s")
                running, running_steps, tic = {}, 0, time.perf_counter()

            if step % cfg.train.eval_every == 0 or step == cfg.train.max_steps:
                eval_model = ema.module if ema is not None else model
                for name, dataset in eval_sets.items():
                    entry = {"step": step, "epoch": epoch, "split": name, **evaluate(eval_model, dataset, cfg, device)}
                    record(entry)
                    print(f"  eval {name}: ce {entry['ce']:.4f}  acc {entry['acc_step']:.3f} "
                          f"(hold {entry['acc_hold']:.3f})  acc_first {entry['acc_first']:.3f}  "
                          f"mae {entry['mae_expected']:.3f}  mae_first {entry['mae_expected_first']:.3f}  "
                          f"q {entry['q_target_rate']:.2f}")

            if step % cfg.train.checkpoint_every == 0 or step == cfg.train.max_steps:
                save_checkpoint(out_dir / "checkpoints" / f"step_{step:07d}.pt", model, ema, optimizer, step, cfg)

            if step >= cfg.train.max_steps:
                break
        epoch += 1

    log_file.close()
    return result


def save_checkpoint(path: Path, model: Denoiser, ema: AveragedModel | None, optimizer: torch.optim.Optimizer,
                    step: int, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "config": config_to_dict(cfg),
        "model": model.state_dict(),
        "ema": ema.module.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
    }, path)


def load_model(path: str | Path, device: str | torch.device = "cpu", use_ema: bool = True) -> tuple[Config, Denoiser]:
    """Model (EMA weights if available and requested) and config from a checkpoint, in eval mode."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(checkpoint["config"])
    model = Denoiser(cfg).to(device)
    weights = checkpoint["ema"] if use_ema and checkpoint["ema"] is not None else checkpoint["model"]
    model.load_state_dict(weights)
    return cfg, model.eval()
