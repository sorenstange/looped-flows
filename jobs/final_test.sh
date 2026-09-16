#!/bin/sh
### Phase 5: the one-shot evaluation on the held-out test split, at the real inference settings.
###
### The test split is used ONCE, after configs and checkpoints are frozen (documents/roadmap.md). Running it during
### development turns it into a second validation set and the final numbers stop meaning what they claim, so this
### job refuses to start unless you say so explicitly:
###
###     CHECKPOINT=<path> CONFIRM_FINAL=yes bsub < jobs/final_test.sh

#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J final_test
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u s204229@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/d3/0/155487/looped-flows/outputs/cluster/final_test_%J.out
#BSUB -e /zhome/d3/0/155487/looped-flows/outputs/cluster/final_test_%J.err

REPO=${REPO:-/zhome/d3/0/155487/looped-flows}
. "$REPO/jobs/_common.sh"
require_checkpoint

[ "$CONFIRM_FINAL" = "yes" ] || {
    echo "refusing to touch the test split: set CONFIRM_FINAL=yes if this really is the frozen final evaluation"
    exit 1
}

# Full inference settings (K and n from `inference`, not the cheap development defaults), all baselines, whole split.
run scripts.backtest --config "$CONFIG" \
    backtest.checkpoint="$CHECKPOINT" \
    backtest.split=test \
    backtest.inference_samples=null backtest.inference_flow_steps=null \
    data.update=false $EXTRA

run scripts.inspect_model --config "$CONFIG" \
    inspect.checkpoint="$CHECKPOINT" \
    inspect.split=test \
    data.update=false $EXTRA
