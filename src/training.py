"""Training loop, losses and metrics for the denoiser.

Methods (`train.method`):
- direct: one denoiser call on an empty trajectory (zeros, flow time 0) predicting the oracle trajectory; the
  direct-predictor baseline with the same backbone.
- looped_flow: paper Algorithm 1. Per batch, sample sorted flow times t_0 < ... < t_{k-1} and one noise x_0; at
  step i the denoiser sees I_i = (1 - t_i) x_0 + t_i x_1 and the recurrent state of step i-1 (detached), and every
  step is its own optimizer step. A sample halts (stops contributing to later steps) once its confidence q > 1/2,
  subject to a random minimum number of steps for a fraction `act.exploration_prob` of samples.

Every optimizer step counts towards `train.max_steps`, logging, evaluation and checkpoints.
"""

import contextlib
import json
import os
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
from src.losses import level_log_probs, sample_cross_entropy
from src.modules import Denoiser, DenoiserOutput
from src.optim import lr_factor, make_optimizer
from src.oracle import make_levels, tolerance_match, trajectory_pnl
from src.sampling import predict

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def noise_scale(cfg: Config) -> float:
    return cfg.flow.noise_scale if cfg.flow.noise_scale is not None else 1 / cfg.oracle.num_levels ** 0.5


def sample_flow_times(batch_size: int, steps: int, sampler: str, device: torch.device,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """(B, steps) increasing flow times in [0, 1): the first `steps` of k+1 sorted draws (paper Appendix E)."""
    if sampler == "sorted":
        draws = torch.rand(batch_size, steps + 1, generator=generator, device=device)
    elif sampler == "random_start":
        start = torch.rand(batch_size, 1, generator=generator, device=device)
        draws = torch.cat([start, start + (1 - start) * torch.rand(batch_size, steps, generator=generator,
                                                                    device=device)], dim=1)
    else:
        raise ValueError(f"unknown time sampler {sampler!r}")
    return draws.sort(dim=1).values[:, :steps]


def interpolant(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """I_t = (1 - t) x_0 + t x_1 for (B, H, K) tensors and (B,) times."""
    t = t[:, None, None]
    return (1 - t) * x0 + t * x1


def pseudotarget_probability(step: int, cfg: Config) -> float:
    if not cfg.flow.pseudotargets:
        return 0.0
    return min(1.0, step / cfg.flow.pseudotarget_ramp_steps)


def direct_forward(model: Denoiser, batch: dict[str, torch.Tensor]) -> DenoiserOutput:
    """Direct predictor: a single denoiser call with an empty trajectory input."""
    context = batch["context"]
    batch_size = context.shape[0]
    empty = torch.zeros(batch_size, model.horizon, model.num_levels, device=context.device)
    return model(context, batch["position"], empty, torch.zeros(batch_size, device=context.device))


def losses_and_metrics(out: DenoiserOutput, batch: dict[str, torch.Tensor], levels: torch.Tensor, cfg: Config,
                       mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Level cross-entropy + weighted confidence BCE and detached metrics, averaged over the samples in `mask`."""
    target = batch["target"]
    weights = torch.ones(target.shape[0], device=target.device) if mask is None else mask.to(torch.float32)
    total = weights.sum().clamp(min=1)

    def mean(values: torch.Tensor) -> torch.Tensor:
        return (values.to(torch.float64) * weights).sum() / total

    ce = sample_cross_entropy(out.logits, target, cfg.train.loss)  # (B,)
    predicted = out.logits.argmax(dim=-1)
    q_target = tolerance_match(levels[predicted], levels[target], cfg.act.tolerance, cfg.act.steps).to(torch.float32)
    bce = F.binary_cross_entropy_with_logits(out.q_logit.float(), q_target, reduction="none")
    loss = mean(ce + cfg.act.loss_weight * bce).float()

    with torch.no_grad():
        target_levels = levels[target]
        expected = level_log_probs(out.logits, cfg.train.loss).exp() @ levels  # (B, H) expected level
        hold = ((batch["position"].double() + 1) / (2 / (len(levels) - 1))).round().long()  # level nearest to a_0
        metrics = {
            "loss": loss.detach(),
            "ce": mean(ce.detach()),
            "bce": mean(bce.detach()),
            "acc_step": mean((predicted == target).double().mean(dim=1)),
            "acc_first": mean((predicted[:, 0] == target[:, 0]).double()),
            "mae_expected": mean((expected - target_levels).abs().mean(dim=1)),
            "mae_expected_first": mean((expected[:, 0] - target_levels[:, 0]).abs()),
            "acc_hold": mean((hold[:, None] == target).double().mean(dim=1)),  # naive: keep the current position
            "q_target_rate": mean(q_target.double()),
            "q_acc": mean(((out.q_logit > 0).float() == q_target).double()),
        }
    return loss, metrics


class WandbLogger:
    """Thin wrapper so that training works identically with logging disabled."""

    def __init__(self, cfg: Config, out_dir: Path):
        self.run = None
        mode = os.environ.get("WANDB_MODE", cfg.wandb.mode)
        if not cfg.wandb.enabled or mode == "disabled":
            return
        from dotenv import load_dotenv
        import wandb

        load_dotenv(REPO_ROOT / ".env", override=False)  # WANDB_API_KEY; never logged or stored in configs
        self.run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=mode, dir=out_dir,
                              name=out_dir.name, config=config_to_dict(cfg),
                              tags=list(cfg.wandb.tags) + [cfg.train.method, cfg.data.interval])
        (out_dir / "wandb_url.txt").write_text(str(self.run.url))
        print(f"wandb run: {self.run.url}")

    def log(self, entry: dict) -> None:
        if self.run is None:
            return
        split = entry["split"]
        self.run.log({f"{split}/{key}": value for key, value in entry.items()
                      if key not in ("step", "split") and isinstance(value, (int, float))}, step=entry["step"])

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


class Trainer:
    def __init__(self, cfg: Config, market: MarketData | None, out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = resolve_device(cfg.train.device)
        torch.manual_seed(cfg.train.seed)

        datasets = build_datasets(cfg, market)
        self.train_set = datasets["train"]
        self.overfit = cfg.train.overfit_samples is not None
        if self.overfit:
            self.train_set = WindowDataset(self.train_set.market, self.train_set.anchors[:cfg.train.overfit_samples],
                                           cfg, seed=cfg.loader.seed)
        self.eval_sets = {"train_subset": self.train_set} if self.overfit else {"val": datasets["val"]}

        self.model = Denoiser(cfg).to(self.device)
        self.ema = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(cfg.train.ema_decay)) \
            if cfg.train.ema_decay is not None else None
        self.optimizer = make_optimizer(self.model.parameters(), cfg.train)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda s: lr_factor(s, cfg.train))
        self.levels = make_levels(cfg.oracle.num_levels).to(self.device)

        self.step = 0
        self.epoch = 0
        self.result = TrainResult(out_dir)
        self.log_file = (out_dir / "metrics.jsonl").open("a")
        self.wandb = WandbLogger(cfg, out_dir)
        self._running: dict[str, torch.Tensor] = {}
        self._running_steps = 0
        self._tic = time.perf_counter()

    @property
    def done(self) -> bool:
        return self.step >= self.cfg.train.max_steps

    def optimize(self, loss: torch.Tensor, metrics: dict[str, torch.Tensor]) -> None:
        """One optimizer step, followed by logging, evaluation and checkpointing when due."""
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = self.cfg.train.grad_clip if self.cfg.train.grad_clip is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite loss or gradient at step {self.step + 1}: loss {loss.item()}, "
                                     f"grad norm {grad_norm.item()}")
        self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update_parameters(self.model)
        self.step += 1

        metrics["grad_norm"] = grad_norm.detach()
        for key, value in metrics.items():
            self._running[key] = self._running.get(key, 0.0) + value.detach().double()
        self._running_steps += 1

        cfg = self.cfg.train
        if self.step % cfg.log_every == 0 or self.done:
            self._log_train()
        if self.step % cfg.eval_every == 0 or self.done:
            self._evaluate()
        if self.step % cfg.checkpoint_every == 0 or self.done:
            save_checkpoint(self.out_dir / "checkpoints" / f"step_{self.step:07d}.pt", self.model, self.ema,
                            self.optimizer, self.step, self.cfg)

    def record(self, entry: dict) -> None:
        self.result.history.append(entry)
        self.log_file.write(json.dumps(entry) + "\n")
        self.log_file.flush()
        self.wandb.log(entry)

    def _log_train(self) -> None:
        elapsed = time.perf_counter() - self._tic
        entry = {"step": self.step, "epoch": self.epoch, "split": "train", "lr": self.scheduler.get_last_lr()[0],
                 "steps_per_s": self._running_steps / elapsed,
                 **{key: (value / self._running_steps).item() for key, value in self._running.items()}}
        self.record(entry)
        extra = f"  active {entry['active_frac']:.2f}" if "active_frac" in entry else ""
        print(f"step {self.step:>6}  loss {entry['loss']:.4f}  ce {entry['ce']:.4f}  acc {entry['acc_step']:.3f} "
              f"(hold {entry['acc_hold']:.3f})  mae {entry['mae_expected']:.3f}  q {entry['q_target_rate']:.2f}"
              f"{extra}  lr {entry['lr']:.2e}  {entry['steps_per_s']:.1f} it/s")
        self._running, self._running_steps, self._tic = {}, 0, time.perf_counter()

    def _evaluate(self) -> None:
        eval_model = self.ema.module if self.ema is not None else self.model
        for name, dataset in self.eval_sets.items():
            entry = {"step": self.step, "epoch": self.epoch, "split": name,
                     **evaluate(eval_model, dataset, self.cfg, self.device),
                     **sample_metrics(eval_model, dataset, self.cfg, self.device)}
            self.record(entry)
            if "sample_mae_first" in entry:
                print(f"  sampled {name}: mae_first {entry['sample_mae_first']:.3f}  mae {entry['sample_mae']:.3f}  "
                      f"acc {entry['sample_acc_step']:.3f}  spread_first {entry['sample_spread_first']:.3f}  "
                      f"pnl {entry['sample_pnl']:.4f} ({entry['sample_pnl_vs_oracle']:.1%} of oracle)")
            last = "_last_t" if self.cfg.train.method == "looped_flow" else ""
            print(f"  eval {name}: ce {entry['ce' + last]:.4f}  acc {entry['acc_step' + last]:.3f} "
                  f"(hold {entry['acc_hold' + last]:.3f})  acc_first {entry['acc_first' + last]:.3f}  "
                  f"mae {entry['mae_expected' + last]:.3f}  mae_first {entry['mae_expected_first' + last]:.3f}  "
                  f"q {entry['q_target_rate' + last]:.2f}"
                  + (f"  | first t: ce {entry['ce_first_t']:.4f}  mae_first {entry['mae_expected_first_first_t']:.3f}"
                     if last else ""))

    def fit(self) -> TrainResult:
        params = sum(p.numel() for p in self.model.parameters())
        print(f"training {self.cfg.train.method} ({params / 1e6:.2f}M params) on {self.device}: "
              f"{len(self.train_set)} train samples, output {self.out_dir}")
        batch_step = {"direct": self._direct_batch, "looped_flow": self._looped_flow_batch}[self.cfg.train.method]
        try:
            while not self.done:
                if not self.overfit:
                    self.train_set.set_epoch(self.epoch)  # fresh a_0 draws; overfitting keeps them fixed
                loader = make_loader(self.train_set, self.cfg.loader.batch_size, shuffle=True,
                                     seed=self.cfg.loader.seed + self.epoch, num_workers=self.cfg.loader.num_workers,
                                     drop_last=len(self.train_set) >= self.cfg.loader.batch_size)
                for batch in loader:
                    self.model.train()
                    batch_step(to_device(batch, self.device))
                    if self.done:
                        break
                self.epoch += 1
        finally:
            self.log_file.close()
            self.wandb.finish()
        return self.result

    def _direct_batch(self, batch: dict[str, torch.Tensor]) -> None:
        with autocast(self.device, self.cfg.train.precision):
            out = direct_forward(self.model, batch)
        loss, metrics = losses_and_metrics(out, batch, self.levels, self.cfg)
        self.optimize(loss, metrics)

    def _looped_flow_batch(self, batch: dict[str, torch.Tensor]) -> None:
        cfg, device = self.cfg, self.device
        target = batch["target"]
        batch_size, steps = target.shape[0], cfg.flow.steps
        x1 = F.one_hot(target, cfg.oracle.num_levels).to(torch.float32)
        sigma = noise_scale(cfg)
        times = sample_flow_times(batch_size, steps, cfg.flow.time_sampler, device)
        x0 = torch.randn_like(x1) * sigma
        use_pseudotargets = torch.rand(batch_size, device=device) < pseudotarget_probability(self.step, cfg)
        explore = torch.rand(batch_size, device=device) < cfg.act.exploration_prob
        min_steps = torch.where(explore, torch.randint(2, steps + 1, (batch_size,), device=device), 1)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        state, previous_prediction = None, None

        for i in range(steps):
            if not cfg.flow.share_noise and i > 0:
                x0 = torch.randn_like(x1) * sigma
            clean = x1
            if previous_prediction is not None:  # App. D: the first interpolant always uses the true target
                clean = torch.where(use_pseudotargets[:, None, None], previous_prediction, x1)
            active = ~halted
            with autocast(device, cfg.train.precision):
                out = self.model(batch["context"], batch["position"], interpolant(x0, clean, times[:, i]),
                                 times[:, i], state)
            loss, metrics = losses_and_metrics(out, batch, self.levels, cfg, mask=active)
            metrics["active_frac"] = active.double().mean()
            metrics["flow_t"] = (times[:, i].double() * active).sum() / active.sum()
            self.optimize(loss, metrics)

            state = out.state
            previous_prediction = level_log_probs(out.logits.detach(), cfg.train.loss).exp().to(torch.float32)
            if cfg.act.halting:
                halted |= (out.q_logit.detach() > 0) & (i + 1 >= min_steps)
            if halted.all() or self.done:
                break


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: WindowDataset, cfg: Config, device: torch.device) -> dict[str, float]:
    """Mean metrics over `train.eval_batches` batches spread evenly over the dataset (a_0 draws of epoch 0).

    looped_flow: teacher-forced rollout over the fixed grid t_i = i / k with seeded shared noise and no halting;
    reports each metric averaged over steps, at the first step (`*_first_t`, prediction from pure noise) and at the
    last step (`*_last_t`).
    """
    model.eval()
    levels = make_levels(cfg.oracle.num_levels).to(device)
    count = min(len(dataset), cfg.train.eval_batches * cfg.loader.batch_size)
    positions = np.linspace(0, len(dataset) - 1, count).astype(np.int64)
    generator = torch.Generator(device=device).manual_seed(cfg.train.seed)
    totals: dict[str, float] = {}

    def add(metrics: dict[str, torch.Tensor], suffix: str, weight: float) -> None:
        for key, value in metrics.items():
            totals[key + suffix] = totals.get(key + suffix, 0.0) + value.item() * weight

    for chunk in np.array_split(positions, max(1, int(np.ceil(count / cfg.loader.batch_size)))):
        batch = to_device(dataset[chunk.tolist()], device)
        if cfg.train.method == "direct":
            with autocast(device, cfg.train.precision):
                out = direct_forward(model, batch)
            add(losses_and_metrics(out, batch, levels, cfg)[1], "", len(chunk))
            continue
        steps = cfg.flow.steps
        x1 = F.one_hot(batch["target"], cfg.oracle.num_levels).to(torch.float32)
        x0 = torch.randn(x1.shape, generator=generator, device=device) * noise_scale(cfg)
        state = None
        for i in range(steps):
            t = torch.full((len(chunk),), i / steps, device=device)
            with autocast(device, cfg.train.precision):
                out = model(batch["context"], batch["position"], interpolant(x0, x1, t), t, state)
            state = out.state
            metrics = losses_and_metrics(out, batch, levels, cfg)[1]
            add(metrics, "", len(chunk) / steps)
            if i == 0:
                add(metrics, "_first_t", len(chunk))
            if i == steps - 1:
                add(metrics, "_last_t", len(chunk))
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def sample_metrics(model: Denoiser, dataset: WindowDataset, cfg: Config, device: torch.device) -> dict[str, float]:
    """Metrics of generated trajectories (the honest measure for flow models), evenly spaced over the dataset.

    Uses `predict` with the inference settings (direct: one prediction; looped flow: K sampler runs). The mean
    expected-level trajectory over the K samples is compared to the oracle and traded on the true future returns.
    """
    model.eval()
    batches = cfg.train.sample_eval_batches
    count = min(len(dataset), batches * cfg.loader.batch_size)
    if count == 0:
        return {}
    levels = make_levels(cfg.oracle.num_levels).to(device)
    positions = np.linspace(0, len(dataset) - 1, count).astype(np.int64)
    generator = torch.Generator(device=device).manual_seed(cfg.train.seed)
    totals = {"sample_mae_first": 0.0, "sample_mae": 0.0, "sample_acc_step": 0.0, "sample_spread_first": 0.0}
    pnl, oracle_pnl = 0.0, 0.0
    for chunk in np.array_split(positions, max(1, int(np.ceil(count / cfg.loader.batch_size)))):
        batch = to_device(dataset[chunk.tolist()], device)
        with autocast(device, cfg.train.precision):
            out = predict(model, batch["context"], batch["position"], cfg, generator=generator)
        expected = out.probs.double() @ levels  # (B, K, H)
        mean_path = expected.mean(dim=1)
        target_levels = levels[batch["target"]]
        weight = len(chunk)
        totals["sample_mae_first"] += (mean_path[:, 0] - target_levels[:, 0]).abs().mean().item() * weight
        totals["sample_mae"] += (mean_path - target_levels).abs().mean().item() * weight
        totals["sample_acc_step"] += (out.levels == batch["target"][:, None]).double().mean().item() * weight
        totals["sample_spread_first"] += expected[:, :, 0].std(dim=1).nan_to_num().mean().item() * weight
        pnl += trajectory_pnl(mean_path, batch["returns"].double(), batch["position"], cfg.oracle.cost).sum().item()
        oracle_pnl += batch["oracle_pnl"].double().sum().item()
    metrics = {key: value / count for key, value in totals.items()}
    metrics["sample_pnl"] = pnl / count
    metrics["sample_pnl_vs_oracle"] = pnl / oracle_pnl if oracle_pnl else 0.0
    return metrics


def train(cfg: Config, market: MarketData | None = None, out_dir: str | Path | None = None) -> TrainResult:
    out_dir = Path(out_dir) if out_dir is not None else \
        Path(cfg.output_dir) / "train" / f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.train.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    return Trainer(cfg, market, out_dir).fit()


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
