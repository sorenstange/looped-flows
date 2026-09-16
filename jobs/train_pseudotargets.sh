#!/bin/sh
### Ablation: pseudotargets on (paper Appendix D) instead of always interpolating towards the true oracle target.
### CLAUDE.md flags this as a recipe that may not transfer - the market target has many near-optimal solutions,
### so bootstrapping towards the model's own previous prediction can reinforce a wrong plan.

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J flow_pseudo
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_pseudo_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/flow_pseudo_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"

run scripts.train --config "$CONFIG" \
    train.method=looped_flow \
    flow.pseudotargets=true \
    train.sample_eval_batches=1 \
    train.checkpoint_every=2000 \
    data.update=false $EXTRA
