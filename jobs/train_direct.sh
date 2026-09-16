#!/bin/sh
### Direct predictor baseline: one denoiser call on an empty trajectory, same backbone and target as the flow.
### Success criterion 2 is that the looped flow beats this. Far cheaper per step (no 16-step rollout).

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J direct
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/direct_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/direct_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"

run scripts.train --config "$CONFIG" \
    train.method=direct \
    train.sample_eval_batches=1 \
    train.checkpoint_every=2000 \
    data.update=false $EXTRA
