#!/bin/sh
### How the sampled trajectories become the traded allocation (vision.md "Open questions" -> readout step).
### Step 1 is the next bar's planned position; later steps and the prefix mean are smoother and should pay fewer
### fees, at the cost of acting on a stale plan.
###
###     CHECKPOINT=outputs/train/<run>/checkpoints/step_0100000.pt bsub < jobs/sweep_readout.sh

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J sweep_readout
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/sweep_readout_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/sweep_readout_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"
require_checkpoint

BASE="backtest.checkpoint=$CHECKPOINT backtest.policies=[buy_hold] backtest.clipped_baselines=false"

echo "=== readout step (expected level at step j) ==="
for STEP in 1 2 5 10 20; do
    run scripts.backtest --config "$CONFIG" $BASE \
        inference.readout=step inference.readout_step=$STEP \
        $SWEEP_SPLIT data.update=false $EXTRA
done

echo "=== prefix mean (average of steps 1..j) ==="
for STEP in 2 5 10 20; do
    run scripts.backtest --config "$CONFIG" $BASE \
        inference.readout=prefix_mean inference.readout_step=$STEP \
        $SWEEP_SPLIT data.update=false $EXTRA
done
