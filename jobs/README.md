# Cluster jobs

LSF batch scripts for the Phase 4 experiments in `documents/roadmap.md`. Submit from the repo root:

```sh
bsub < jobs/train_flow.sh
CHECKPOINT=outputs/train/20260916-101500-looped_flow/checkpoints/step_0100000.pt bsub < jobs/backtest.sh
```

Every script is a `#BSUB` header followed by `. jobs/_common.sh`, which holds the shared setup: `cd` to the repo,
create `outputs/cluster/`, check that the parquet cache exists, and print the host, GPU and git commit so a log
identifies its own code version. All jobs run with `data.update=false` — compute nodes have no outbound network,
so the cache must be copied across first (`documents/cluster.md`).

## Environment variables

| variable | meaning |
|----------|---------|
| `REPO` | repo root on the cluster (default `/zhome/d3/0/155487/looped-flows`, also hard-coded in the `#BSUB -o/-e` paths) |
| `CONFIG` | config file, default `configs/default.yaml` |
| `CHECKPOINT` | checkpoint for the evaluation jobs; they refuse to start without it |
| `SPLIT` | split for `backtest.sh` / `inspect.sh`, default `val` |
| `EXTRA` | extra `key=value` overrides appended to every command in the job |
| `SWEEP_SPLIT` | val slice the sweeps evaluate on, default `splits.test_start=2024-04-01` (Q1 2024); export empty for the whole split |

## Training

| job | what it answers |
|-----|-----------------|
| `train_flow.sh` | looped flow at the paper defaults (StableMax + Adam-atan2) |
| `train_direct.sh` | direct predictor, same backbone — success criterion 2 compares against this |
| `train_flow_adamw.sh` | softmax + AdamW instead; the paper setting learned far slower in the smoke run |
| `train_direct_adamw.sh` | fourth cell of the (method × optimizer) grid, so the two are not confounded |
| `train_maxstep.sh` | target smoothness: `oracle.max_step=0.2` (override with `MAX_STEP=`) |
| `train_pseudotargets.sh` | pseudotargets on (paper App. D) |
| `train_sweep_lr.sh` | job array over four learning rates; the pattern to copy for other hyperparameters |

All training jobs set `train.sample_eval_batches=1` and `train.checkpoint_every=2000`. The defaults (4 and 10000)
spend 6–10 minutes per eval on the sampler and leave 1.4 h of work unsaved between checkpoints, which a 24 h
walltime kill would discard.

## Evaluation

| job | what it answers |
|-----|-----------------|
| `backtest.sh` | full val backtest against every baseline and the oracle, net of fees |
| `inspect.sh` | the qualitative plots from `scripts/inspect_model.py` |
| `sweep_inference.sh` | inference-time scaling: n, K, γ (success criterion 3) |
| `sweep_readout.sh` | readout step 1 vs later steps vs prefix mean |
| `sweep_ablations.sh` | sample aggregation (mean / best-q / q-weighted) and execution with vs without the step limit |
| `final_test.sh` | **Phase 5 only**: the one-shot test-split evaluation |

The sweeps run one backtest per grid point, each writing its own `outputs/backtest/<timestamp>/` with `config.yaml`
and `metrics.json`, so results are reproducible from the output alone and nothing clobbers anything. They skip the
rule-based baselines (`backtest.policies=[buy_hold]`) because only the model varies across a grid.

## The test split

`documents/roadmap.md` reserves the test split for the single final evaluation after configs and checkpoints are
frozen. Every job here defaults to `val`; `final_test.sh` additionally refuses to run without `CONFIRM_FINAL=yes`.

## Monitoring

```sh
bjobs                                  # running jobs; bjobs -A for array summaries
bpeek <jobid>                          # live stdout
tail -f outputs/cluster/flow_<jobid>.out
bkill <jobid>
```
