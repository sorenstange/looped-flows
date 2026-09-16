#!/bin/sh
### Learning-rate sweep as an LSF job array: one GPU per value, all four running side by side.
### Roadmap Phase 4: "Hyperparameter selection on val". Copy this file as the pattern for other sweeps
### (model width, flow.steps k, flow.noise_scale sigma, features.lookback) by changing VALUES and the override.
###
###     bsub < jobs/train_sweep_lr.sh          # submits the whole array
###     bjobs -A                               # array summary

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J lr[1-4]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/lr_%J_%I.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/lr_%J_%I.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"

VALUES="3e-5 1e-4 3e-4 1e-3"
LR=$(echo $VALUES | cut -d' ' -f"${LSB_JOBINDEX:-1}")
[ -n "$LR" ] || { echo "no value for array index ${LSB_JOBINDEX}"; exit 1; }
echo "array index ${LSB_JOBINDEX}: lr $LR"

# Shorter than a full run: a sweep only needs enough steps to rank the settings.
run scripts.train --config "$CONFIG" \
    train.method=looped_flow \
    train.lr="$LR" \
    train.max_steps=30000 \
    train.sample_eval_batches=1 \
    train.checkpoint_every=5000 \
    "wandb.tags=[lr_sweep]" \
    data.update=false $EXTRA
