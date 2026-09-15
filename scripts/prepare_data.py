"""Download/update market data, build the chronological splits and report oracle target statistics.

Saves the resolved config and the report to <output_dir>/prepare_data/<timestamp>/.

    uv run python -m scripts.prepare_data --config configs/default.yaml oracle.cost=0.001
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import load_config, parse_args, save_config
from src.dataset import MarketData, build_datasets
from src.oracle import oracle_stats

MAX_STAT_SAMPLES = 20_000


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.overrides)
    market = MarketData.load(cfg)
    datasets = build_datasets(cfg, market)

    report = {
        "bars": len(market.times),
        "first_bar": str(market.times[0]),
        "last_bar": str(market.times[-1]),
        "invalid_bars": int((~market.valid).sum()),
        "splits": {},
    }
    for name, dataset in datasets.items():
        # Evenly spaced subset keeps the statistics cheap while covering the whole split period.
        positions = np.linspace(0, len(dataset) - 1, min(len(dataset), MAX_STAT_SAMPLES)).astype(np.int64)
        batch = dataset[positions.tolist()]
        report["splits"][name] = {
            "samples": len(dataset),
            "first_anchor": str(market.times[dataset.anchors[0]]),
            "last_anchor": str(market.times[dataset.anchors[-1]]),
            "oracle": oracle_stats(batch["target"], dataset.levels, batch["position"], batch["oracle_pnl"]),
        }

    out_dir = Path(cfg.output_dir) / "prepare_data" / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_config(cfg, out_dir / "config.yaml")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"{report['bars']} bars {report['first_bar']} .. {report['last_bar']}, {report['invalid_bars']} invalid")
    for name, split in report["splits"].items():
        oracle = split["oracle"]
        print(f"{name:>5}: {split['samples']:>7} samples  {split['first_anchor']} .. {split['last_anchor']}")
        print(f"       oracle short/flat/long {oracle['short_frac']:.1%}/{oracle['flat_frac']:.1%}/"
              f"{oracle['long_frac']:.1%}  |position| {oracle['mean_abs_position']:.2f}  "
              f"full {oracle['full_position_frac']:.1%}  turnover/bar {oracle['turnover_per_bar']:.3f}")
        print(f"       holding {oracle['mean_holding_bars']:.1f} bars at a level, "
              f"{oracle['mean_direction_bars']:.1f} bars per direction  "
              f"pnl mean {oracle['pnl_mean']:.4f} median {oracle['pnl_median']:.4f}")
    print(f"saved config and report to {out_dir}")


if __name__ == "__main__":
    main()
