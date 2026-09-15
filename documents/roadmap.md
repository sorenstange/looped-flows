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
- [ ] **Backbone** (`src/modules.py`): input projections (context features, `a_0`, noisy one-hot trajectory), time
  embedding, noncausal transformer with RoPE, SwiGLU, RMSNorm, TRM-style (h, ℓ) recurrence, output head, ACT head;
  shape and gradient tests
- [ ] **Direct predictor** (same backbone, no flow, no loop) + minimal training loop: the first learned baseline, and a
  check that data, model and optimization work end to end (overfit a tiny subset)
- [ ] **Looped-flow training** (Algorithm 1): sorted flow times, shared `(c, x0, x1)`, stop-gradient between steps,
  ACT loss and halting, optional pseudotargets; EMA, warmup, gradient clipping, checkpointing, metric logging
- [ ] **Sampler** (Algorithm 2): noise backtracking with `γ`; test that `γ = 0` equals Euler integration
- [ ] **Model policy** for the backtest: sample K trajectories per bar, feed their level probabilities (and confidence
  scores) to `readout_allocation`; batch decisions efficiently where possible
- [ ] **Smoke run** on `configs/smoke.yaml` on CPU: loss decreases, sampler produces valid trajectories, backtest runs

## Phase 4 — Experiments (cluster)
- [ ] Cluster setup: CUDA torch, run scripts, output syncing
- [ ] Train direct predictor and looped flow on the default config; compare on val
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
