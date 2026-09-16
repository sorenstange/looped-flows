#!/bin/sh
### Remaining val ablations that need no retraining: how the K samples are aggregated, and whether execution is
### step-limited. Retraining ablations have their own jobs (train_pseudotargets.sh, train_maxstep.sh).
###
###     CHECKPOINT=outputs/train/<run>/checkpoints/step_0100000.pt bsub < jobs/sweep_ablations.sh

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J sweep_abl
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/sweep_abl_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/sweep_abl_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"
require_checkpoint

BASE="backtest.checkpoint=$CHECKPOINT backtest.policies=[buy_hold] backtest.clipped_baselines=false"

echo "=== aggregation over the K samples (q is the confidence head) ==="
for AGG in mean best_q q_weighted; do
    run scripts.backtest --config "$CONFIG" $BASE \
        inference.aggregation=$AGG \
        $SWEEP_SPLIT data.update=false $EXTRA
done

echo "=== execution with and without the step limit ==="
for ENFORCE in false true; do
    run scripts.backtest --config "$CONFIG" $BASE \
        backtest.enforce_max_step=$ENFORCE \
        $SWEEP_SPLIT data.update=false $EXTRA
done
