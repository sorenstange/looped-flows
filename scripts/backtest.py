"""Backtest the configured policies (and optionally a trained model) on one split; save metrics, per-bar positions
and an equity-curve plot to <output_dir>/backtest/<timestamp>/.

    uv run python -m scripts.backtest --config configs/default.yaml backtest.split=val oracle.max_step=0.2
    uv run python -m scripts.backtest --config configs/smoke.yaml backtest.checkpoint=outputs/train/<run>/checkpoints/step_0000300.pt
"""

import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import decision_range, performance, run_backtest
from src.config import load_config, parse_args, save_config
from src.dataset import MarketData
from src.model_backtest import ModelDecider, check_compatible, run_model_backtest
from src.policies import RULE_BASED, make_policy
from src.training import load_model, resolve_device


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.overrides)
    market = MarketData.load(cfg)
    _, end = decision_range(market, cfg)
    policies = [make_policy(name, cfg, market, last_bar=end) for name in cfg.backtest.policies]
    start, _ = decision_range(market, cfg, warmup=max((policy.warmup for policy in policies), default=0))
    cost = cfg.backtest.cost if cfg.backtest.cost is not None else cfg.oracle.cost
    max_step = cfg.oracle.max_step if cfg.backtest.enforce_max_step else None

    results = [run_backtest(policy, market, start, end, cost, cfg.features.lookback, max_step,
                            cfg.backtest.initial_position)
               for policy in policies]
    if cfg.backtest.clipped_baselines and max_step is None and cfg.oracle.max_step is not None:
        # Step-limited variants, so models can be compared against each baseline's stronger version.
        results += [replace(run_backtest(policy, market, start, end, cost, cfg.features.lookback,
                                         cfg.oracle.max_step, cfg.backtest.initial_position),
                            policy=f"{policy.name}_clip")
                    for policy in policies if policy.name in RULE_BASED]
    if cfg.backtest.checkpoint is not None:
        model_cfg, model = load_model(cfg.backtest.checkpoint, resolve_device(cfg.backtest.device))
        check_compatible(model_cfg, cfg)
        decider = ModelDecider(model, model_cfg, cfg, market)
        tic = time.perf_counter()
        results.append(run_model_backtest(decider, start, end, cost, max_step, cfg.backtest.initial_position,
                                          cfg.backtest.chains, cfg.backtest.burn_in,
                                          name=f"model_{model_cfg.train.method}"))
        print(f"model backtest ({model_cfg.train.method}, K={decider.samples}, n={decider.flow_steps}, "
              f"{cfg.backtest.chains} chains) took {time.perf_counter() - tic:.0f}s")
    metrics = {result.policy: performance(result, cfg.data.interval) for result in results}

    out_dir = Path(cfg.output_dir) / "backtest" / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_config(cfg, out_dir / "config.yaml")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.concat({result.policy: result.to_frame() for result in results}, axis=1).to_parquet(out_dir / "positions.parquet")
    plot_equity(results, out_dir / "equity.png",
                f"{cfg.data.symbol} {cfg.data.interval} {cfg.backtest.split}, cost {cost:g}, max step {max_step}")

    period = f"{market.times[start]} .. {market.times[end]}"
    print(f"{cfg.backtest.split} split, decisions {period}, cost {cost:g}, max step {max_step}")
    print(f"{'policy':<18}{'net pnl':>9}{'fees':>8}{'ann.ret':>9}{'ann.vol':>9}{'sharpe':>8}{'max dd':>8}"
          f"{'turn/bar':>9}{'|pos|':>7}{'hit':>7}")
    for name, m in metrics.items():
        hit = f"{m['hit_rate']:.1%}" if m["hit_rate"] is not None else "-"
        print(f"{name:<18}{m['net_pnl']:>9.3f}{m['fees']:>8.3f}{m['annual_return']:>9.1%}{m['annual_vol']:>9.1%}"
              f"{m['sharpe']:>8.2f}{m['max_drawdown']:>8.3f}{m['turnover_per_bar']:>9.3f}"
              f"{m['mean_abs_position']:>7.2f}{hit:>7}")
    print(f"saved to {out_dir}")


def plot_equity(results, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for result in results:
        ax.plot(result.times, np.cumsum(result.pnl), label=result.policy, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("cumulative net PnL (units of notional)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
