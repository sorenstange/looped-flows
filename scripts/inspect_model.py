"""Qualitative inspection plots for a trained checkpoint; saved to <output_dir>/inspect/<timestamp>/.

Answers "what does this model actually predict?" rather than "what does it score?": planned trajectories against the
oracle's, the level distributions behind them, how quality decays over the horizon, whether the traded position tracks
price, and whether more sampling compute helps.

    uv run python -m scripts.inspect_model --config configs/default.yaml \
        inspect.checkpoint=outputs/train/<run>/checkpoints/step_0010000.pt
    uv run python -m scripts.inspect_model --config configs/smoke.yaml \
        inspect.checkpoint=<path> inspect.decisions=256 inspect.scaling_curve=false

Sampling follows the checkpoint (method, loss, noise scale); readout, K and n follow the run config, so the same
checkpoint can be inspected under different inference settings without retraining.
"""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.backtest import decision_range, performance, run_backtest
from src.config import Config, load_config, parse_args, save_config
from src.dataset import MarketData, WindowDataset, build_datasets
from src.model_backtest import ModelDecider, check_compatible, run_model_backtest
from src.oracle import make_levels
from src.policies import make_policy
from src.sampling import predict
from src.training import autocast, load_model, resolve_device, to_device

PLAN_COLOR, ORACLE_COLOR, HOLD_COLOR = "#1f77b4", "#000000", "#999999"


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.overrides)
    inspect = cfg.inspect
    checkpoint = inspect.checkpoint if inspect.checkpoint is not None else cfg.backtest.checkpoint
    if checkpoint is None:
        raise SystemExit("no checkpoint: set inspect.checkpoint=<path/to/step_*.pt> (or backtest.checkpoint)")

    device = resolve_device(inspect.device)
    model_cfg, model = load_model(checkpoint, device)
    market = MarketData.load(cfg)
    check_compatible(model_cfg, cfg)
    # Sampling follows the checkpoint, readout and sample counts the run config (as in ModelDecider).
    infer_cfg = replace(model_cfg, inference=cfg.inference)
    samples = inspect.samples if inspect.samples is not None else cfg.inference.samples
    flow_steps = inspect.flow_steps if inspect.flow_steps is not None else cfg.inference.flow_steps
    dataset = build_datasets(cfg, market)[inspect.split]
    levels = make_levels(cfg.oracle.num_levels).to(device)

    out_dir = Path(cfg.output_dir) / "inspect" / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_config(cfg, out_dir / "config.yaml")
    method = model_cfg.train.method
    subtitle = (f"{method}, {Path(checkpoint).name}, {cfg.inspect.split} split, "
                f"K={samples}, n={flow_steps}, γ={cfg.inference.gamma:g}")
    print(f"inspecting {checkpoint}\n  {subtitle}\n  device {device}, {len(dataset)} anchors in the split")

    data = collect(model, dataset, infer_cfg, device, levels, inspect.decisions, inspect.examples, samples,
                   flow_steps, inspect.seed)
    summary = summarize(data, cfg)
    print_summary(summary)

    plot_horizon(data, cfg, out_dir / "horizon.png", subtitle)
    plot_trajectories(data, dataset, cfg, out_dir / "trajectories.png", subtitle)
    plot_heatmaps(data, dataset, cfg, out_dir / "heatmaps.png", subtitle)
    plot_calibration(data, cfg, out_dir / "calibration.png", subtitle)
    timeline = run_timeline(model, model_cfg, cfg, market, out_dir / "timeline.png", subtitle)
    summary["timeline"] = timeline

    if inspect.scaling_curve and method == "looped_flow":
        grid = scaling_curve(model, dataset, infer_cfg, device, levels, inspect.scaling_decisions, samples,
                             flow_steps, inspect.seed)
        plot_scaling(grid, out_dir / "scaling.png", subtitle)
        summary["scaling"] = grid
    elif inspect.scaling_curve:
        print("scaling curve skipped: the direct predictor makes one deterministic prediction (K = 1, n = 1)")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"saved to {out_dir}")


# --------------------------------------------------------------------------------------- sampling


@torch.no_grad()
def collect(model: torch.nn.Module, dataset: WindowDataset, cfg: Config, device: torch.device,
            levels: torch.Tensor, decisions: int, examples: int, samples: int, flow_steps: int,
            seed: int) -> dict:
    """Sample trajectories for `decisions` anchors spread evenly over the split.

    Aggregates are kept for every decision; the full per-sample paths and level probabilities are kept only for the
    `examples` decisions that the per-decision plots draw, since (N, K, H, L) would not fit in memory.
    """
    count = min(decisions, len(dataset))
    positions = np.unique(np.linspace(0, len(dataset) - 1, count).astype(np.int64))
    example_at = set(positions[np.linspace(0, len(positions) - 1, min(examples, len(positions))).astype(np.int64)])
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = max(1, cfg.loader.batch_size // max(1, samples // 4))  # K samples widen the effective batch

    chunks = np.array_split(positions, max(1, int(np.ceil(len(positions) / batch_size))))
    out: dict[str, list] = {key: [] for key in ("plan", "target", "a0", "spread", "q", "anchor")}
    picked: list[dict] = []
    for chunk in chunks:
        batch = to_device(dataset[chunk.tolist()], device)
        with autocast(device, cfg.train.precision):
            prediction = predict(model, batch["context"], batch["position"], cfg, generator=generator,
                                 samples=samples, flow_steps=flow_steps)
        expected = prediction.probs.double() @ levels.double()  # (B, K, H) expected level per sample
        plan = expected.mean(dim=1)  # (B, H) what the mean aggregation would trade
        out["plan"].append(plan.cpu().numpy())
        out["target"].append(levels[batch["target"]].cpu().numpy())
        out["a0"].append(batch["position"].double().cpu().numpy())
        out["spread"].append(expected.std(dim=1).nan_to_num().cpu().numpy())
        out["q"].append(prediction.q_logit.float().sigmoid().mean(dim=1).cpu().numpy())
        out["anchor"].append(batch["anchor"].cpu().numpy())
        for i, position in enumerate(chunk):
            if position in example_at:
                picked.append({"anchor": int(batch["anchor"][i]), "paths": expected[i].cpu().numpy(),
                               "probs": prediction.probs[i].float().mean(dim=0).cpu().numpy(),
                               "target": levels[batch["target"][i]].cpu().numpy(),
                               "a0": float(batch["position"][i])})

    data = {key: np.concatenate(value) for key, value in out.items()}
    data["examples"] = picked
    return data


@torch.no_grad()
def scaling_curve(model: torch.nn.Module, dataset: WindowDataset, cfg: Config, device: torch.device,
                  levels: torch.Tensor, decisions: int, max_samples: int, max_flow_steps: int,
                  seed: int) -> dict:
    """First-step MAE over a grid of sampler steps n and sample counts K (success criterion 3)."""
    steps = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= max_flow_steps] or [max_flow_steps]
    counts = [k for k in (1, 4, 16) if k <= max_samples] or [max_samples]
    grid: dict = {"flow_steps": steps, "samples": counts, "mae_first": {}}
    print(f"scaling curve over n {steps} x K {counts} on {decisions} decisions")
    for k in counts:
        maes = []
        for n in steps:
            data = collect(model, dataset, cfg, device, levels, decisions, 0, k, n, seed)
            maes.append(float(np.abs(data["plan"][:, 0] - data["target"][:, 0]).mean()))
        grid["mae_first"][k] = maes
        print(f"  K={k:>2}: " + "  ".join(f"n={n}: {mae:.3f}" for n, mae in zip(steps, maes)))
    return grid


# --------------------------------------------------------------------------------------- metrics


def level_index(values: np.ndarray, num_levels: int) -> np.ndarray:
    """Nearest level index of continuous allocations in [-1, 1]."""
    return np.clip(np.rint((values + 1) * (num_levels - 1) / 2), 0, num_levels - 1)


def summarize(data: dict, cfg: Config) -> dict:
    """Headline numbers, each next to the two baselines that make them readable (see `plot_horizon`)."""
    plan, target, a0 = data["plan"], data["target"], data["a0"]
    num_levels = cfg.oracle.num_levels
    hold = np.broadcast_to(a0[:, None], target.shape)
    correct = level_index(plan, num_levels) == level_index(target, num_levels)
    return {
        "decisions": int(len(plan)),
        "mae": float(np.abs(plan - target).mean()),
        "mae_first": float(np.abs(plan[:, 0] - target[:, 0]).mean()),
        "mae_first_hold": float(np.abs(a0 - target[:, 0]).mean()),  # trivial "keep a_0" baseline
        "mae_first_flat": float(np.abs(target[:, 0]).mean()),  # trivial "always flat" baseline
        "mae_hold": float(np.abs(hold - target).mean()),
        "acc": float(correct.mean()),
        "acc_first": float(correct[:, 0].mean()),
        "acc_hold": float((level_index(hold, num_levels) == level_index(target, num_levels)).mean()),
        "spread_first": float(data["spread"][:, 0].mean()),
        "plan_abs_mean": float(np.abs(plan[:, 0]).mean()),
        "target_abs_mean": float(np.abs(target[:, 0]).mean()),
        "q_mean": float(data["q"].mean()),
    }


def print_summary(s: dict) -> None:
    print(f"\n{s['decisions']} decisions")
    print(f"  {'first-step MAE':<22}{s['mae_first']:.3f}   (hold a_0 {s['mae_first_hold']:.3f}, "
          f"flat {s['mae_first_flat']:.3f})")
    print(f"  {'MAE over all steps':<22}{s['mae']:.3f}   (hold a_0 {s['mae_hold']:.3f})")
    print(f"  {'first-step accuracy':<22}{s['acc_first']:.3f}   (hold a_0 {s['acc_hold']:.3f})")
    print(f"  {'mean |planned pos|':<22}{s['plan_abs_mean']:.3f}   (oracle {s['target_abs_mean']:.3f})")
    print(f"  {'spread across K':<22}{s['spread_first']:.3f}")
    print(f"  {'mean confidence q':<22}{s['q_mean']:.3f}")
    if s["mae_first"] > s["mae_first_hold"]:
        print("  note: first-step MAE is worse than simply keeping a_0, which the step limit bounds by "
              "oracle.max_step")


# --------------------------------------------------------------------------------------- plots


def finish(fig, path: Path, subtitle: str) -> None:
    fig.suptitle(subtitle, fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path.name}")


def plot_horizon(data: dict, cfg: Config, path: Path, subtitle: str) -> None:
    """Where in the horizon the model has any signal, against the two trivial baselines."""
    plan, target, a0 = data["plan"], data["target"], data["a0"]
    steps = np.arange(1, target.shape[1] + 1)
    hold = np.broadcast_to(a0[:, None], target.shape)
    correct = level_index(plan, cfg.oracle.num_levels) == level_index(target, cfg.oracle.num_levels)
    hold_correct = level_index(hold, cfg.oracle.num_levels) == level_index(target, cfg.oracle.num_levels)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, np.abs(plan - target).mean(axis=0), color=PLAN_COLOR, label="model")
    axes[0].plot(steps, np.abs(hold - target).mean(axis=0), color=HOLD_COLOR, ls="--", label="keep $a_0$")
    axes[0].plot(steps, np.abs(target).mean(axis=0), color=ORACLE_COLOR, ls=":", label="always flat")
    axes[0].set_title("MAE vs trajectory step")
    axes[0].set_ylabel("mean |planned - oracle| (allocation)")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(steps, correct.mean(axis=0), color=PLAN_COLOR, label="model")
    axes[1].plot(steps, hold_correct.mean(axis=0), color=HOLD_COLOR, ls="--", label="keep $a_0$")
    axes[1].axhline(1 / cfg.oracle.num_levels, color=ORACLE_COLOR, ls=":", label="uniform")
    axes[1].set_title("exact-level accuracy vs trajectory step")
    axes[1].set_ylabel("share of decisions")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(steps, data["spread"].mean(axis=0), color=PLAN_COLOR)
    axes[2].set_title("disagreement across the K samples")
    axes[2].set_ylabel("std of expected level")
    axes[2].set_ylim(bottom=0)

    for ax in axes:
        ax.set_xlabel("trajectory step (bars ahead)")
        ax.grid(alpha=0.3)
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    finish(fig, path, subtitle)


def plot_trajectories(data: dict, dataset: WindowDataset, cfg: Config, path: Path, subtitle: str) -> None:
    """Individual plans against the oracle's, with the price path they were made for."""
    picked = data["examples"]
    if not picked:
        return
    rows = int(np.ceil(len(picked) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.1 * rows), squeeze=False)
    market, horizon = dataset.market, cfg.oracle.horizon
    steps = np.arange(1, horizon + 1)

    for ax, example in zip(axes.ravel(), picked):
        anchor = example["anchor"]
        for sample_path in example["paths"]:
            ax.plot(steps, sample_path, color=PLAN_COLOR, alpha=0.25, linewidth=0.7)
        ax.plot(steps, example["paths"].mean(axis=0), color=PLAN_COLOR, linewidth=2, label="model (mean of K)")
        ax.step(steps, example["target"], where="mid", color=ORACLE_COLOR, linewidth=1.4, label="oracle")
        ax.scatter([0], [example["a0"]], color=HOLD_COLOR, zorder=5, s=25, label="$a_0$")
        ax.set_ylim(-1.15, 1.15)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("allocation")
        ax.set_xlabel("bars ahead")
        ax.set_title(str(market.times[anchor]), fontsize=9)

        price = market.close[anchor:anchor + horizon + 1]
        twin = ax.twinx()
        twin.plot(np.arange(len(price)), 100 * (price / price[0] - 1), color="#d62728", alpha=0.5, linewidth=1)
        twin.set_ylabel("price vs decision bar (%)", color="#d62728", fontsize=8)
        twin.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
    axes.ravel()[0].legend(fontsize=8, loc="upper left")
    for ax in axes.ravel()[len(picked):]:
        ax.axis("off")
    finish(fig, path, subtitle)


def plot_heatmaps(data: dict, dataset: WindowDataset, cfg: Config, path: Path, subtitle: str) -> None:
    """The level distribution behind each plan: sharp and following the oracle, or diffuse and hedging?"""
    picked = data["examples"]
    if not picked:
        return
    rows = int(np.ceil(len(picked) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.1 * rows), squeeze=False)
    horizon, steps = cfg.oracle.horizon, np.arange(1, cfg.oracle.horizon + 1)

    for ax, example in zip(axes.ravel(), picked):
        image = ax.imshow(example["probs"].T, origin="lower", aspect="auto", cmap="magma",
                          extent=(0.5, horizon + 0.5, -1 - 1 / (cfg.oracle.num_levels - 1),
                                  1 + 1 / (cfg.oracle.num_levels - 1)))
        ax.step(steps, example["target"], where="mid", color="#00d0ff", linewidth=1.3, label="oracle")
        ax.set_title(str(dataset.market.times[example["anchor"]]), fontsize=9)
        ax.set_xlabel("bars ahead")
        ax.set_ylabel("allocation level")
        fig.colorbar(image, ax=ax, label="probability", pad=0.01)
    axes.ravel()[0].legend(fontsize=8, loc="upper left")
    for ax in axes.ravel()[len(picked):]:
        ax.axis("off")
    finish(fig, path, subtitle)


def plot_calibration(data: dict, cfg: Config, path: Path, subtitle: str) -> None:
    """Step 1 is the bar that gets traded: is the planned position related to the oracle's, and does it use the range?"""
    plan, target, a0 = data["plan"][:, 0], data["target"][:, 0], data["a0"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist2d(plan, target, bins=(40, cfg.oracle.num_levels), range=((-1, 1), (-1.05, 1.05)), cmap="magma")
    axes[0].plot([-1, 1], [-1, 1], color="#00d0ff", linewidth=1, ls="--", label="perfect")
    axes[0].set_xlabel("planned allocation (step 1)")
    axes[0].set_ylabel("oracle allocation (step 1)")
    axes[0].set_title(f"correlation {np.corrcoef(plan, target)[0, 1]:.3f}")
    axes[0].legend(fontsize=8)

    bins = np.linspace(-1.05, 1.05, 43)
    axes[1].hist(target, bins=bins, color=ORACLE_COLOR, alpha=0.45, label="oracle")
    axes[1].hist(plan, bins=bins, color=PLAN_COLOR, alpha=0.6, label="model")
    axes[1].hist(a0, bins=bins, color=HOLD_COLOR, histtype="step", label="$a_0$")
    axes[1].set_title("step-1 allocation distribution")
    axes[1].set_xlabel("allocation")
    axes[1].set_ylabel("decisions")
    axes[1].legend(fontsize=8)

    # A model that only echoes its current position has planned - a_0 concentrated at zero.
    axes[2].hist(plan - a0, bins=np.linspace(-1, 1, 61), color=PLAN_COLOR, alpha=0.7, label="model")
    axes[2].hist(target - a0, bins=np.linspace(-1, 1, 61), color=ORACLE_COLOR, histtype="step", label="oracle")
    if cfg.oracle.max_step is not None:
        for sign in (-1, 1):
            axes[2].axvline(sign * cfg.oracle.max_step, color="#d62728", ls=":", linewidth=1)
        axes[2].set_title(f"move from $a_0$ (oracle limited to ±{cfg.oracle.max_step:g})")
    else:
        axes[2].set_title("move from $a_0$")
    axes[2].set_xlabel("planned - current allocation")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    finish(fig, path, subtitle)


def run_timeline(model: torch.nn.Module, model_cfg: Config, cfg: Config, market: MarketData, path: Path,
                 subtitle: str) -> dict:
    """Traded position against price over one contiguous stretch, next to the oracle's position.

    Uses the real backtest engine (`run_model_backtest`), so a_0 feeds back exactly as it would when trading.
    """
    split_cfg = replace(cfg, backtest=replace(cfg.backtest, split=cfg.inspect.split))
    start, end = decision_range(market, split_cfg)
    bars = min(cfg.inspect.timeline_bars, end - start)
    if cfg.inspect.timeline_start is not None:
        begin = max(start, min(int(market.times.searchsorted(pd.Timestamp(cfg.inspect.timeline_start, tz="UTC"))),
                               end - bars))
    else:  # the window with the largest absolute move, where a directional model has the most to show
        cumulative = np.concatenate([[0.0], np.nancumsum(market.returns[start + 1:end + 1])])
        begin = start + int(np.argmax(np.abs(cumulative[bars:] - cumulative[:-bars])))
    finish_bar = begin + bars

    decider = ModelDecider(model, model_cfg, cfg, market)
    cost = cfg.backtest.cost if cfg.backtest.cost is not None else cfg.oracle.cost
    max_step = cfg.oracle.max_step if cfg.backtest.enforce_max_step else None
    chains = max(1, min(cfg.backtest.chains, bars // max(1, 2 * cfg.backtest.burn_in)))
    model_result = run_model_backtest(decider, begin, finish_bar, cost, max_step, cfg.backtest.initial_position,
                                      chains, cfg.backtest.burn_in, name="model")
    oracle = make_policy("oracle", cfg, market, last_bar=finish_bar)
    oracle_result = run_backtest(oracle, market, begin, finish_bar, cost, cfg.features.lookback, max_step,
                                 cfg.backtest.initial_position)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.4]})
    times = model_result.times
    axes[0].plot(times, market.close[begin:finish_bar], color="#d62728", linewidth=1)
    axes[0].set_ylabel("close")
    axes[0].set_title(f"{cfg.data.symbol} {cfg.data.interval}, {bars} bars from {times[0]}", fontsize=9)

    axes[1].plot(times, oracle_result.positions, color=ORACLE_COLOR, linewidth=1, label="oracle (hindsight)")
    axes[1].plot(times, model_result.positions, color=PLAN_COLOR, linewidth=1, label="model")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("allocation")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend(fontsize=8)

    axes[2].plot(times, np.cumsum(model_result.pnl), color=PLAN_COLOR, label="model")
    axes[2].plot(times, np.cumsum(model_result.gross), color=PLAN_COLOR, ls=":", alpha=0.7, label="model, gross")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("cum. PnL")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    finish(fig, path, subtitle)

    metrics = {"start": str(times[0]), "bars": int(bars), "chains": int(chains),
               "model": performance(model_result, cfg.data.interval),
               "oracle": performance(oracle_result, cfg.data.interval)}
    print(f"  timeline {times[0]} .. {times[-1]}: model net PnL {metrics['model']['net_pnl']:.3f} "
          f"(fees {metrics['model']['fees']:.3f}), oracle {metrics['oracle']['net_pnl']:.3f}")
    return metrics


def plot_scaling(grid: dict, path: Path, subtitle: str) -> None:
    """Success criterion 3: does spending more inference compute measurably improve the prediction?"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, maes in grid["mae_first"].items():
        ax.plot(grid["flow_steps"], maes, marker="o", label=f"K = {k}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid["flow_steps"])
    ax.set_xticklabels([str(n) for n in grid["flow_steps"]])
    ax.set_xlabel("sampler steps n")
    ax.set_ylabel("first-step MAE")
    ax.set_title("inference-time scaling")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    finish(fig, path, subtitle)


if __name__ == "__main__":
    main()
