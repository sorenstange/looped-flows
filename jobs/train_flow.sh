#!/bin/sh
### Looped flow (Algorithm 1) at the paper defaults: StableMax loss + Adam-atan2.
### Roadmap Phase 4: "Train direct predictor and looped flow on the default config; compare on val".

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J flow
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"

run scripts.train --config "$CONFIG" \
    train.method=looped_flow \
    train.sample_eval_batches=1 \
    train.checkpoint_every=2000 \
    data.update=false $EXTRA
