# Shared setup for the LSF jobs in this folder. Sourced after the #BSUB block, which LSF requires to come first.
#
# Environment variables a submitter may set:
#   REPO        repo root on the cluster (default below)
#   CONFIG      config file (default configs/default.yaml)
#   CHECKPOINT  checkpoint for the evaluation jobs, e.g. outputs/train/<run>/checkpoints/step_0100000.pt
#   EXTRA       extra `key=value` overrides appended to every command in the job

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
cd "$REPO" || { echo "repo not found: $REPO (set REPO=...)"; exit 1; }
mkdir -p outputs/cluster

# Overrides are passed unquoted so the shell splits them into words; disable globbing so list values such as
# `backtest.policies=[buy_hold]` are not treated as filename patterns.
set -f

CONFIG=${CONFIG:-configs/default.yaml}

# Compute nodes have no outbound network, so every job runs with data.update=false and the parquet cache must
# already be in place. See documents/cluster.md for how to copy it across.
CACHE=data/raw/futures_BTCUSDT_5m.parquet
[ -f "$CACHE" ] || { echo "missing $CACHE - copy the cache across first (documents/cluster.md)"; exit 1; }

# Sweeps evaluate on a slice of val so a whole grid fits in one job; export SWEEP_SPLIT= to use the full split.
# Model selection may use a slice, but the final val comparison between methods should use all of it.
SWEEP_SPLIT=${SWEEP_SPLIT-splits.test_start=2024-04-01}

echo "job ${LSB_JOBID:-local} on $(hostname), $(git rev-parse --short HEAD 2>/dev/null || echo 'no git'), config $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

run() {
    echo "+ uv run python -m $*"
    uv run python -m "$@" || exit 1
}

require_checkpoint() {
    [ -n "$CHECKPOINT" ] || { echo "set CHECKPOINT=<path/to/step_*.pt> before submitting"; exit 1; }
    [ -f "$CHECKPOINT" ] || { echo "no such checkpoint: $CHECKPOINT"; exit 1; }
    echo "checkpoint $CHECKPOINT"
}
