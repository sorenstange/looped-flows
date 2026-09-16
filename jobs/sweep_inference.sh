#!/bin/sh
### Inference-time scaling on val: sampler steps n, stochasticity gamma, samples K (success criterion 3).
### One backtest per grid point, each writing its own outputs/backtest/<timestamp>/ with metrics.json.
###
###     CHECKPOINT=outputs/train/<run>/checkpoints/step_0100000.pt bsub < jobs/sweep_inference.sh
###
### Evaluates on the val slice set by SWEEP_SPLIT in _common.sh so the grid fits in one job; export SWEEP_SPLIT=
### to use the whole split (much slower). Every point uses the same bars, so the numbers are comparable.

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J sweep_inf
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/sweep_inf_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/sweep_inf_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"
require_checkpoint

# Only the model matters here, so skip the rule-based baselines (the oracle is the expensive one).
BASE="backtest.checkpoint=$CHECKPOINT backtest.policies=[buy_hold] backtest.clipped_baselines=false"

echo "=== flow steps n (K = 16, gamma = 5) ==="
for N in 1 2 4 8 16 32; do
    run scripts.backtest --config "$CONFIG" $BASE \
        backtest.inference_samples=16 backtest.inference_flow_steps=$N \
        $SWEEP_SPLIT data.update=false $EXTRA
done

echo "=== samples K (n = 32, gamma = 5) ==="
for K in 1 2 4 8 16 32; do
    run scripts.backtest --config "$CONFIG" $BASE \
        backtest.inference_samples=$K backtest.inference_flow_steps=32 \
        $SWEEP_SPLIT data.update=false $EXTRA
done

echo "=== stochasticity gamma (K = 16, n = 32); gamma = 0 is deterministic Euler ==="
for GAMMA in 0 1 5 10; do
    run scripts.backtest --config "$CONFIG" $BASE \
        backtest.inference_samples=16 backtest.inference_flow_steps=32 inference.gamma=$GAMMA \
        $SWEEP_SPLIT data.update=false $EXTRA
done
