#!/bin/sh
### Looped flow with softmax cross-entropy + AdamW instead of the paper's StableMax + Adam-atan2.
### Roadmap Phase 4: the paper setting learned far more slowly in the CPU smoke run; this decides it at full scale.
### Submit alongside jobs/train_flow.sh so both are read at the same step count.

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J flow_adamw
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_adamw_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_adamw_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"

run scripts.train --config "$CONFIG" \
    train.method=looped_flow \
    train.loss=softmax train.optimizer=adamw \
    train.sample_eval_batches=1 \
    train.checkpoint_every=2000 \
    data.update=false $EXTRA
