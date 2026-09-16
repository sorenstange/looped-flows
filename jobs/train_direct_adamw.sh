#!/bin/sh
### Direct predictor with softmax + AdamW: the fourth cell of the (method x loss/optimizer) comparison, so the
### optimizer choice is not confounded with the training method.

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J direct_adamw
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/direct_adamw_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/direct_adamw_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"

run scripts.train --config "$CONFIG" \
    train.method=direct \
    train.loss=softmax train.optimizer=adamw \
    train.sample_eval_batches=1 \
    train.checkpoint_every=2000 \
    data.update=false $EXTRA
