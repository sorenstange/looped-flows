"""Sweep the receding-horizon oracle (hindsight upper bound) over max step, horizon and oracle planning cost.

Every grid point is backtested on `backtest.split` over the same decision bars and charged `backtest.cost`; buy & hold
and the MA crossover are included for reference. Saves results.csv, heatmaps and the resolved config to
<output_dir>/sweep_oracle/<timestamp>/.

    uv run python -m scripts.sweep_oracle --config configs/default.yaml data.update=false
"""

import itertools
import time
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
from src.policies import make_policy

HEATMAP_METRICS = [("net_pnl", "net PnL"), ("sharpe", "Sharpe"), ("turnover_per_bar", "turnover / bar"),
                   ("mean_direction_bars", "bars per direction")]


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.overrides)
    market = MarketData.load(cfg)
    sweep = cfg.oracle_sweep
    reference = ["buy_hold", "ma_crossover"]
    warmup = max(make_policy(name, cfg, market, last_bar=1).warmup for name in reference)
    # The longest horizon sets the purge gap, so every grid point trades exactly the same bars.
    start, end = decision_range(market, cfg, warmup=warmup, horizon=max(sweep.horizons))
    cost = cfg.backtest.cost if cfg.backtest.cost is not None else cfg.oracle.cost

    rows = []
    for name in reference:
        max_step = cfg.oracle.max_step if cfg.backtest.enforce_max_step else None
        result = run_backtest(make_policy(name, cfg, market, end), market, start, end, cost, cfg.features.lookback,
                              max_step, cfg.backtest.initial_position)
        rows.append({"policy": name, **performance(result, cfg.data.interval)})

    grid = list(itertools.product(sweep.costs, sweep.max_steps, sweep.horizons))
    for i, (oracle_cost, max_step, horizon) in enumerate(grid, 1):
        tic = time.perf_counter()
        point = load_config(args.config, list(args.overrides) + [
            f"oracle.cost={oracle_cost}", f"oracle.max_step={'null' if max_step is None else max_step}",
            f"oracle.horizon={horizon}"])
        executed_step = max_step if cfg.backtest.enforce_max_step else None
        result = run_backtest(make_policy("oracle", point, market, end), market, start, end, cost,
                              cfg.features.lookback, executed_step, cfg.backtest.initial_position)
        metrics = performance(result, cfg.data.interval)
        rows.append({"policy": "oracle", "oracle_cost": oracle_cost, "max_step": max_step, "horizon": horizon,
                     **metrics})
        print(f"[{i:>3}/{len(grid)}] cost {oracle_cost:g} max_step {max_step} H {horizon:>3}: "
              f"net pnl {metrics['net_pnl']:7.3f}  sharpe {metrics['sharpe']:6.2f}  "
              f"turnover {metrics['turnover_per_bar']:.3f}  dir bars {metrics['mean_direction_bars']:6.1f}  "
              f"({time.perf_counter() - tic:.1f}s)")

    results = pd.DataFrame(rows)
    out_dir = Path(cfg.output_dir) / "sweep_oracle" / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_config(cfg, out_dir / "config.yaml")
    results.to_csv(out_dir / "results.csv", index=False)
    plot_heatmaps(results[results["policy"] == "oracle"], out_dir / "heatmaps.png",
                  f"{cfg.data.symbol} {cfg.data.interval} {cfg.backtest.split} {market.times[start]:%Y-%m-%d}.."
                  f"{market.times[end]:%Y-%m-%d}, backtest cost {cost:g}")
    for row in rows[:len(reference)]:
        print(f"reference {row['policy']}: net pnl {row['net_pnl']:.3f}  sharpe {row['sharpe']:.2f}")
    print(f"saved to {out_dir}")


def plot_heatmaps(results: pd.DataFrame, path: Path, title: str) -> None:
    costs = sorted(results["oracle_cost"].unique())
    fig, axes = plt.subplots(len(costs), len(HEATMAP_METRICS), figsize=(4.2 * len(HEATMAP_METRICS), 3.4 * len(costs)),
                             squeeze=False)
    for row, oracle_cost in enumerate(costs):
        subset = results[results["oracle_cost"] == oracle_cost].copy()
        subset["max_step"] = subset["max_step"].map(lambda s: "none" if pd.isna(s) else f"{s:g}")
        for col, (metric, label) in enumerate(HEATMAP_METRICS):
            table = subset.pivot(index="max_step", columns="horizon", values=metric)
            table = table.reindex([s for s in dict.fromkeys(subset["max_step"])])
            ax = axes[row][col]
            ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(table.columns)), [f"{h:g}" for h in table.columns])
            ax.set_yticks(range(len(table.index)), table.index)
            ax.set_xlabel("horizon H")
            ax.set_ylabel("max step")
            ax.set_title(f"{label} (oracle cost {oracle_cost:g})", fontsize=9)
            for (y, x), value in np.ndenumerate(table.to_numpy()):
                ax.text(x, y, f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
