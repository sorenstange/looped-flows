#!/bin/sh
### Full val backtest of one checkpoint alongside every baseline and the receding-horizon oracle, net of fees.
### This is the receding-horizon (MPC) evaluation: at each bar the model samples K trajectories, trades the first
### step of the readout, and that position becomes the next bar's a_0.
###
###     CHECKPOINT=outputs/train/<run>/checkpoints/step_0100000.pt bsub < jobs/backtest.sh
###
### Uses the cheap development defaults K=4, n=16. For the final numbers at the real inference settings, add
###     EXTRA="backtest.inference_samples=null backtest.inference_flow_steps=null"   (roughly 8x slower)

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J backtest
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/backtest_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/backtest_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"
require_checkpoint

run scripts.backtest --config "$CONFIG" \
    backtest.checkpoint="$CHECKPOINT" \
    backtest.split="${SPLIT:-val}" \
    data.update=false $EXTRA
