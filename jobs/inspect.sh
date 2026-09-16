#!/bin/sh
### Qualitative inspection plots for one checkpoint: planned vs oracle trajectories, level-probability heatmaps,
### MAE and accuracy along the horizon against the "keep a_0" and "always flat" baselines, step-1 calibration,
### a position-vs-price timeline, and the inference-time scaling curve.
###
###     CHECKPOINT=outputs/train/<run>/checkpoints/step_0100000.pt bsub < jobs/inspect.sh

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J inspect
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/inspect_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/inspect_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"
require_checkpoint

run scripts.inspect_model --config "$CONFIG" \
    inspect.checkpoint="$CHECKPOINT" \
    inspect.split="${SPLIT:-val}" \
    data.update=false $EXTRA
