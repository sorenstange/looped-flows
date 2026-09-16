#!/bin/sh
### Target smoothness vs learnability: the same looped flow trained on a max step of 0.2 instead of 0.1.
### The oracle sweep showed 0.2 has more headroom (33.3 vs 24.8 net PnL on val) but cannot measure whether the
### coarser target is easier or harder to learn. Compare this run against jobs/train_flow.sh on val.
###
### oracle.max_step changes the training target, so the resulting checkpoint is only comparable to a backtest run
### at the same setting; scripts/backtest.py picks it up from the checkpoint's own config.

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J flow_step02
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_step02_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_step02_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"

MAX_STEP=${MAX_STEP:-0.2}

run scripts.train --config "$CONFIG" \
    train.method=looped_flow \
    oracle.max_step=$MAX_STEP \
    train.sample_eval_batches=1 \
    train.checkpoint_every=2000 \
    data.update=false $EXTRA
