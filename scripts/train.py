"""Train a denoiser (see src/training.py for the methods).

Saves config.yaml, metrics.jsonl and checkpoints to <output_dir>/train/<timestamp>-<method>/.

    uv run python -m scripts.train --config configs/smoke.yaml
    uv run python -m scripts.train --config configs/smoke.yaml train.overfit_samples=64
"""

from src.config import load_config, parse_args
from src.training import train


def main() -> None:
    args = parse_args(__doc__)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
