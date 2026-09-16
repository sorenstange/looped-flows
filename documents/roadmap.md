# Roadmap

Prioritized checklist, top to bottom. Each item names what "done" means. Check items off as they land, add new
items where they belong in the order, and move decisions into `vision.md` once they are made.

Rule for the whole project: **the test split is only used for the final evaluation (Phase 5).** All development,
model selection and sweeps use train / val.

## Phase 0 — Foundations ✅
- [x] Vision and spec (`documents/vision.md`), paper transcription (`documents/paper.md`), `CLAUDE.md`
- [x] YAML config system with inheritance, CLI overrides and resolved-config saving (`src/config.py`)
- [x] Binance download + parquet cache, cleaning with validity flags (`src/data.py`)
- [x] Causal features, per-window standardization (`src/features.py`)
- [x] Exact DP oracle: 21 levels, max step 0.1, brute-force tested (`src/oracle.py`)
- [x] Chronological splits with purge gap, batched `WindowDataset`, loader (`src/dataset.py`)
- [x] Data report script (`scripts/prepare_data.py`)

## Phase 1 — Evaluation harness (no model yet)
- [x] **Backtest engine + baselines** (`src/backtest.py`, `src/policies.py`, `scripts/backtest.py`)
  - bar-by-bar engine: policy sees only bars ≤ t, position carried between bars, step limit applied at execution,
    same timing and cost formula as the oracle (`trajectory_pnl`)
  - metrics: net/gross PnL, fees, annualized return/vol/Sharpe, max drawdown, turnover, exposure, hit rate
  - policies: flat, buy & hold, random, MA crossover, momentum, receding-horizon oracle (hindsight upper bound)
  - outputs: resolved config, metrics JSON, per-bar positions, equity-curve plot
  - val results (2024-01-25 .. 2024-12-31, 1h, cost 0.05%, free and step-limited): see `vision.md` → Baseline results
- [x] **Oracle upper-bound sweep** over `max_step` × `H` × oracle cost (val split, receding-horizon oracle, backtest
  at the real fee); results and recommended defaults in `vision.md` → Oracle upper-bound sweep
  - receding-horizon oracle made fast: batched backward DP for all bars + one step per bar (~170 s → ~5 s per val
    backtest); `scripts/sweep_oracle.py`, 75 grid points in ~12 min
  - decided (user confirmed): max step 0.1, `H` = 64, planning cost = fee
- [x] **Decide execution step limit**: decided (user) — no step limit at execution for any policy
  (`backtest.enforce_max_step: false`); rule-based baselines also run step-limited (`<name>_clip`,
  `backtest.clipped_baselines`) and models are compared against the stronger variant. Rationale in `vision.md` → Usage

- [x] **Switch to 5m bars** (user): `data.interval: 5m`; lookback 256, `H` = 64, max step 0.1 kept per bar; baselines
  scaled to the same wall-clock windows (MA 288 / 2016, momentum 2016); val backtest rerun, results in `vision.md`
- [x] **Re-run the oracle upper-bound sweep at 5m** with a reduced grid (max step {0.1, 0.2, 0.3} × `H` {16, 32, 64},
  ~17 min): 1h conclusions carry over, defaults kept; results in `vision.md`
- [ ] Optional data cleaning: flag zero-volume flat bars (exchange maintenance, 31 bars at 5m) as invalid

## Phase 2 — Remaining design decisions before modelling
- [x] **How the traded allocation is read out**: decided (user) — expected level, averaged over K samples (default
  K = 16, `inference.aggregation: mean`; best-Q as comparison), at a configurable readout (`inference.readout`,
  `inference.readout_step`, default step 1); implemented as `src.policies.readout_allocation`
- [x] **ACT / confidence head target**: decided (user) — tolerance match, mean |rounded predicted level − oracle
  level| ≤ τ over the first m steps (`act.tolerance` 0.1, `act.steps` all H, BCE weight 0.5); `src.oracle.tolerance_match`.
  q is not used for trading by default; `best_q` / `q_weighted` aggregation are ablations
- [x] **Return feature scale**: decided (user) — rolling scaling with a trailing window (default 1 week = 2016 bars):
  price features ÷ trailing RMS of log returns, log volume trailing z-score, new `log_vol_to_cost` channel, clip ±10;
  `features.scaling: window` kept for comparison. Checked on 5m data (train/val scale consistent)
- [x] **Replace the redundant `log_open_close` channel** (equals `-log_return` on perpetuals): replaced by the
  taker-buy share of volume, trailing z-score (user)

## Phase 3 — Model and training (CPU smoke tests first)
- [x] **Backbone** (`src/modules.py`): input projections (context features, `a_0`, noisy one-hot trajectory), time
  embedding, noncausal transformer with RoPE, SwiGLU, RMSNorm, TRM-style (h, ℓ) recurrence, output head, ACT head via a
  [Q] token; `model` config (patching optional); tests for shapes, last-cycle-only gradients, detached state, batch
  independence, input sensitivity. Paper size 7.11M parameters
- [x] **Direct predictor** (same backbone, no flow, no loop) + minimal training loop: the first learned baseline, and a
  check that data, model and optimization work end to end (overfit a tiny subset)
  - `src/training.py` (`train.method: direct`), `src/losses.py` (StableMax, softmax), `src/optim.py` (Adam-atan2,
    AdamW, warmup + constant/cosine), EMA, bf16 on CUDA, eval on evenly spaced val batches, JSONL metrics,
    checkpoints with embedded config (`load_model`), `scripts/train.py`
  - fixed a NaN: StableMax's unused `torch.where` branch had an infinite gradient at logit 1.0
  - smoke (CPU, 0.12M params, 300 steps, softmax + AdamW): val CE 3.04 → 2.71, step accuracy 0.157 vs 0.099 for
    "keep the current position", first-step MAE 0.48 → 0.26; overfitting 64 real samples reaches CE 0.78, MAE 0.05
  - StableMax + Adam-atan2 (paper) barely learned in the same 300 steps; compare both on the cluster
- [x] **Looped-flow training** (Algorithm 1): sorted flow times, shared `(c, x0, x1)`, stop-gradient between steps,
  ACT loss and halting, optional pseudotargets; EMA, warmup, gradient clipping, checkpointing, metric logging
  - `train.method: looped_flow`, `flow` config (k = 16, sorted / random-start times, σ = 1/√21, shared noise,
    pseudotargets with ramp), per-sample halting with exploration (`act.halting`, `act.exploration_prob`); one
    optimizer step per flow step; evaluation = teacher-forced rollout on the grid t_i = i/k with metrics averaged
    over steps, at the first step (pure noise) and at the last step
  - Weights & Biases logging (`wandb` config, key from `.env`), project `looped-flows`; tests force it off
  - smoke (CPU, 300 steps): val CE at t = 15/16 falls to 0.37, but CE at t = 0 stays ≈ 3.19 (no better than chance).
    The late-step metrics are inflated by the nearly revealed target in the interpolant; generation quality needs the
    sampler (next item)
- [x] **Sampler** (Algorithm 2): noise backtracking with `γ`; test that `γ = 0` equals Euler integration
  - `src/sampling.py`: `sample` (K samples per decision, uniform grid with `inference.flow_steps`, `inference.gamma`,
    recurrent state carried across steps), `predict` (direct: one prediction; looped flow: sampler), `backtrack`
  - tests: γ = 0 equals Euler (Eq. 8); an ideal denoiser lands exactly on the solution for γ ∈ {0, 1, 5};
    backtracking preserves the interpolant marginal N(s x₁, σ²(1 − s)²)
  - evaluation now includes metrics of sampled trajectories for both methods (`sample_*`: MAE of the mean expected
    level vs. the oracle, accuracy, spread across samples, PnL of the mean trajectory on the true future returns
    relative to the oracle)
  - smoke checkpoints (300 CPU steps) on val: direct predictor first-step MAE 0.25, looped flow 0.49 (chance level;
    its t = 0 denoising had not learned yet), both with negative PnL: too little training to judge the method.
    Sampling at K = 16, n = 32 on CPU took ~24 min for 512 decisions, so sampled evaluation and backtests need GPUs
- [x] **Model policy** for the backtest: sample K trajectories per bar, feed their level probabilities (and confidence
  scores) to `readout_allocation`; batch decisions efficiently where possible
  - `src/model_backtest.py`: `ModelDecider` (context windows shared with training via `dataset.context_windows` →
    `predict` → batched `readout_allocations`), `run_model_backtest` with parallel chains and burn-in (user decision),
    `ModelPolicy` as the sequential reference, `check_compatible` (checkpoint vs. run config, incl. `oracle.cost`
    because it enters `log_vol_to_cost`); `backtest.checkpoint` in `scripts/backtest.py`
  - development backtests default to K = 4, n = 16 (`backtest.inference_samples` / `inference_flow_steps`, user
    decision); final evaluation uses the `inference` settings
  - tests: one chain equals the sequential policy; chains equal sequential when a_0 does not feed back; full burn-in
    reproduces the sequential run in every chain
  - smoke direct checkpoint on smoke val (May 2024): runs in 153 s on CPU; net PnL -0.17 with 0.21 fees (turnover
    0.048 per bar from rebalancing the expected position) vs. buy & hold +0.17
- [x] **Smoke run** on `configs/smoke.yaml` on CPU: loss decreases, sampler produces valid trajectories, backtest runs
  - both methods train (wandb), sample and backtest end to end on real 5m data
  - looped-flow smoke checkpoint, K = 2, n = 4, smoke val (May 2024): net PnL -0.56, almost all fees (turnover 0.129
    per bar, mean |position| 0.10): with an untrained flow and only 2 samples the averaged expected position is mostly
    sampling noise that gets traded every 5 minutes. Took 1217 s on CPU for 8.6k decisions
  - implication for Phase 4: fees from bar-to-bar noise in the traded position are the first thing to watch; levers
    are more samples K, a later readout step / prefix mean, and the no-trade band

## Phase 4 — Experiments (cluster)
- [ ] Cluster setup: CUDA torch, run scripts, output syncing
  - `documents/cluster.md` written (clone, CUDA torch, `scp` of the 39 MB parquet cache and `.env`, `bsub`, pulling
    checkpoints back); `train.sh` submits `scripts.train` with `data.update=false`
  - `jobs/` holds one LSF script per experiment below (`jobs/README.md`), sharing `jobs/_common.sh`; the first
    looped-flow run is training at ~2.4 it/s, and eval every 2000 steps costs several minutes of it
- [ ] Train direct predictor and looped flow on the default config; compare on val
- [ ] Loss and optimizer: StableMax + Adam-atan2 (paper) vs. softmax + AdamW at full scale (the paper setting learned
  far slower in the CPU smoke run)
- [ ] Hyperparameter selection on val (lookback, `H`, model size, k, σ, learning rate)
- [ ] Target smoothness vs. learnability: train with max step 0.1 and 0.2 (the oracle sweep cannot measure
  learnability) and compare on val
- [ ] Inference-time scaling on val: flow steps `n`, stochasticity `γ`, samples `K`
- [ ] Readout on val: step 1 vs. later steps (j ≈ 5–10) vs. prefix mean
- [ ] Ablations on val: pseudotargets on/off, ODE vs. SDE, mean vs. best-Q vs. q-weighted aggregation, execution with
  vs. without the step limit
- [ ] Tune the confidence-head tolerance τ: check how often the tolerance match is reached per flow step
- [ ] Optional: no-trade band (skip rebalances smaller than a threshold), if small per-bar changes of the expected
  position add noticeable fees

## Phase 5 — Final evaluation
- [ ] Freeze configs and checkpoints; evaluate all models and baselines **once** on the test split
- [ ] Report against the success criteria in `vision.md`

## Later (deferred in `vision.md`)
- [ ] Funding rates in oracle and backtest
- [ ] Slippage model
- [ ] Calendar features
- [ ] Multiple assets
- [ ] Walk-forward evaluation
