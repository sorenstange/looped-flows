# Trading with looped flows

## Goal
Create a model based on the looped flows framework (`2026_suleymanzade_looped-flows.pdf`) which, given a sequence of
OHLCV (crypto) data, generates a future allocation trajectory. Each allocation lies in [-1, 1]: -1 means full short,
0 means no allocation, 1 means full long.

## Core idea
Looped flows learn to transport noise into a solution `x1` conditioned on a problem `c`, with a stateful denoiser whose
recurrent state is carried across progressively less noisy steps. We map this onto trading:

| Paper                  | This project                                                        |
|------------------------|---------------------------------------------------------------------|
| Problem `c`            | Past OHLCV window (lookback) + current position `a_0`               |
| Solution `x1`          | Oracle allocation trajectory over the future horizon `H`            |
| Token vocabulary `V`   | 21 allocation levels {-1, -0.9, ..., 0.9, 1}                        |
| Multiple valid answers | Different plausible futures -> different sampled trajectories      |

## Training target: the oracle
The model is trained on an **oracle allocation trajectory**: the allocation that, with hindsight of the future prices
in the horizon, maximizes PnL net of transaction costs:

```
max_a  Σ_{t=1..H} a_t · r_{t+1} − c · |a_t − a_{t−1}|
s.t.   a_t ∈ {-1, -0.9, ..., 0.9, 1}
       |a_t − a_{t−1}| ≤ 0.1        (including the first move from a_0)
```

- `r_{t+1}` is the return of the bar following allocation `a_t`, `c` the proportional cost per unit of turnover
  (default **c = 0.05%**, the Binance USDT-M taker fee; a full flip from -1 to +1 costs 0.10%; configurable).
- `a_0` is the **current position** held right before the horizon. It is part of the model input, and the oracle starts
  from it, so the first step pays the true cost of changing position. `a_0` can be fractional (see Usage); in training
  it is sampled **uniformly from [-1, 1]** per example so the model learns the cost of switching from any position.
- **No terminal cost**: the position may remain open at the end of the horizon (avoids a bias toward flattening late
  in the trajectory).
- **Max step** (`oracle.max_step`, default 0.1): the position can change by at most 0.1 per bar, so going from flat to
  full takes 10 bars and a full flip 20. This rate limit is what shapes the targets (see below); forbidden moves are
  excluded from the DP.
- Solved exactly with dynamic programming over (time × level).
- Without the step limit, the oracle is bang-bang: a linear objective with linear costs only ever picks -1, 0 or 1,
  and at 1h / 0.05% it flips every ~2.3 bars, close to `sign(r_{t+1})`. Finer levels alone change nothing; the step
  limit is what turns targets into smooth ramps that follow multi-bar trends.

**Oracle regimes on the 1h train split** (analysis done before the switch to 5m; a_0 = 0, 5000 evenly spaced samples,
`H` = 64):

| Variant                | cost  | short/flat/long | mean \|pos\| | at ±1 | turnover/bar | bars per level | bars per direction | oracle PnL |
|------------------------|-------|-----------------|-------------|-------|--------------|----------------|--------------------|------------|
| 3 levels, no limit     | 0.05% | 49/0/51 %       | 1.00        | 99.7% | 0.855        | 2.3            | 2.3                | 0.243      |
| 3 levels, no limit     | 0.2%  | 47/1/51 %       | 0.99        | 98.5% | 0.451        | 4.2            | 4.2                | 0.183      |
| 21 levels, no limit    | 0.05% | identical to 3 levels                                                                                     |
| 21 levels, step 0.1    | 0.05% | 45/4/51 %       | 0.59        | 15.7% | 0.083        | 1.2            | 12.5               | 0.073      |
| 21 levels, step 0.1    | 0.2%  | 45/4/51 %       | 0.61        | 20.6% | 0.062        | 1.6            | 15.5               | 0.066      |

## Representation
- Allocations are **21 discrete levels**, evenly spaced over [-1, 1] (`oracle.num_levels`). With the step limit the
  oracle uses intermediate levels in its ramps.
- One-hot encoded, exactly as the paper's categorical flow: Gaussian-noise interpolant on one-hots, cross-entropy
  denoising loss.
- Fractional exposure emerges from model uncertainty: the expected level `Σ p_level · level` lies in [-1, 1].

## Market and data
- **Binance USDT-M perpetual futures, BTCUSDT** to start (shorting is native). Data from 2019-10-01, skipping the
  placeholder bars right after the September 2019 listing.
- Bar interval, lookback length and horizon `H` are configuration parameters (YAML configs in `configs/`), chosen by
  experiment. Defaults: **5m bars** (switched from 1h), lookback 256 bars (~21 h), `H` = 64 bars (~5.3 h), max step
  0.1 per bar (flat to full in 50 min). Bar counts of the model were kept when switching; the rule-based baselines
  were scaled to the same wall-clock windows (MA 1 day / 1 week, momentum 1 week).
- Returns are **simple returns** of each bar (close to close); positions have constant notional, so PnL is additive.
- **Missing or malformed bars** are put on a regular time grid (filled flat at the previous close) and flagged
  invalid; any sample whose context or horizon touches an invalid bar is dropped.
- **Splits**: a single chronological train / validation / test split, with a purge gap of at least `lookback + H` bars
  between consecutive sets so that no window or oracle horizon overlaps across them.

## Model
**Inputs (conditioning `c`)**: six per-bar features for each of the `lookback` context bars, plus the current
position `a_0`. Scaling uses **trailing (rolling) statistics** over `features.rolling_window` bars (default 2016 =
1 week at 5m, longer than the lookback), computed causally per bar including the bar itself:

| Feature           | Definition                                           | Scaling (`features.scaling: rolling`)               |
|-------------------|------------------------------------------------------|-----------------------------------------------------|
| `log_return`      | `log(close_t / close_{t−1})`                         | ÷ trailing RMS of log returns                       |
| `log_high_close`  | `log(high_t / close_t)`                              | ÷ trailing RMS of log returns                       |
| `log_low_close`   | `log(low_t / close_t)`                               | ÷ trailing RMS of log returns                       |
| `taker_buy_share` | `taker_buy_volume_t / volume_t` (0.5 if no volume)   | trailing z-score (mean and std)                     |
| `log_volume`      | `log1p(volume_t)`                                    | trailing z-score (mean and std)                     |
| `log_vol_to_cost` | `log(trailing RMS of log returns / oracle.cost)`     | none (unitless; ~0–3 at 5m)                         |

Design choices:
- **Divide only, no mean subtraction for price features**: the trailing mean of 5m returns is noise, and subtracting
  it would remove drift. One common scale for all price features keeps their relative sizes (range vs. return).
- **`log_vol_to_cost` restores the volatility level** that rolling scaling removes: whether moves cover the fixed cost
  is what the oracle depends on.
- **Clipping** to ±10 (`features.clip`) against flash moves (~0.05% of bars at 5m). A value can be at most
  √window times the trailing RMS that includes it (≈ 45 for 2016 bars), so the clip is what bounds outliers.
- The first `rolling_window` bars of the data are invalid (warmup; 5m data starts being usable on 2019-10-08).
- **No open-based feature**: on Binance perpetuals each bar opens at the previous close, so `log(open / close)` equals
  `-log_return`. The **taker-buy share** (aggressive buying as a fraction of volume) takes its place: it is a
  buying-pressure signal the other channels do not contain. Its spread varies by regime (yearly std 0.09–0.15 at 5m),
  hence the trailing z-score.
- `features.scaling: window` keeps the earlier variant (unscaled price features, taker-buy share and log volume
  z-scored per sample window) for comparison.
- Check on 5m data: scaled log returns have std 1.03 on train and 1.02 on val, while raw 5m log returns fall from
  std 0.0023 to 0.0017, i.e. the regime shift is absorbed; `log_vol_to_cost` averages 1.25 (train) and 1.08 (val);
  the z-scored taker-buy share has std 1.00 on both splits and is never clipped.

**Backbone**: a single non-causal transformer with rotary position embeddings over one joint sequence

```
[ context tokens (lookback) | a_0 token | H trajectory tokens ]
```

Only the trajectory tokens are noised. Context and `a_0` tokens are clean conditioning, as in the paper's ARC setup.
The denoiser follows the paper and TRM: flow-time embedding, a projection of the noisy one-hot interpolant, and a
two-state recurrence `z = (h, ℓ)` carried across flow steps with stop-gradient between steps. Target scale is
paper-sized (~5–7M parameters).

## Looped-flow components (first version)
- **Temporally aligned training**: `k` sorted flow times with decreasing noise, with the triplet `(c, x0, x1)` (noise
  included) shared across all steps of a rollout.
- **ACT / confidence head `q`** (`act` config): predicts a **tolerance match** with the oracle, trained with BCE
  (weight 0.5 as in the paper): `q = 1` if the mean |rounded predicted level − oracle level| over the first `m` steps
  is ≤ `τ` (defaults `τ` = 0.1, `m` = all H; `src.oracle.tolerance_match`). This replaces the paper's exact match,
  which is practically unreachable on market trajectories, while keeping its meaning ("the prediction is already close
  enough"), so halting during training works as in the paper.
- **q at inference: not used by default.** The traded allocation stays the plain mean over samples; best-Q and
  q-weighted averaging (`inference.aggregation: best_q | q_weighted`) are ablations on val.
- **Pseudotargets** (Appendix D): included as a switch. The paper's justification assumes one solution per problem,
  which does not hold here (the same past can lead to different oracle trajectories), so their effect must be checked
  with an ablation.
- **SDE sampler** (Algorithm 2, `γ > 0`), with `γ = 0` (Euler ODE) as a comparison.
- **Aggregation over `K` sampled trajectories**: mean (default, see Usage); best-Q ensembling as a comparison.

## Usage: receding horizon
At every bar the model samples `K` trajectories of length `H` (`inference.samples`, default 16), one allocation is
traded, and at the next bar it re-plans. The traded allocation becomes `a_0` for the next bar.

**Traded allocation = expected position, averaged over samples** (`src.policies.readout_allocation`):
1. For each sample, take the expected level `Σ p_level · level` of the predicted level probabilities at the readout.
2. Average over the `K` samples (`inference.aggregation: mean`). A single flow sample ends near one-hot, so model
   uncertainty only shows up *across* samples; the mean is the Monte Carlo expected oracle position, roughly
   P(up-trend) − P(down-trend), and acts as position sizing. `best_q` (the sample with the highest confidence score)
   is kept as a comparison; it discards the sizing information.

**Readout step** (`inference.readout`, `inference.readout_step`). The oracle's first step is always within
`a_0 ± 0.1`, so reading out step 1 makes the traded position integrate direction beliefs:
`a_exec ≈ a_0 + 0.1 · (P(up) − P(down))`: smooth and cheap, but at least 10 bars from flat to full. Reading out a later
step `j` (the position the plan wants to be at in `j` bars), or the mean over steps `1..j`, lets the model move straight
to its planned exposure while the training targets stay smooth.
- Default: `readout: step`, `readout_step: 1`. Compare against `j` ≈ 5–10 on val once a model exists.

**No step limit at execution** (`backtest.enforce_max_step: false`). The model may trade any allocation; this is safe
because `a_0` is sampled uniformly from [-1, 1] in training, so every reachable position has been seen as input. A model
trained on step-limited targets respects the limit largely by itself when reading out step 1. All policies trade under
the same rules; the rule-based baselines are additionally run with the step limit (`<name>_clip`), and models are
compared against each baseline's stronger variant.

Possible later refinement: a no-trade band (trade only if the change exceeds a threshold), since an expected position
changes slightly every bar.

## Evaluation
All evaluation is on the chronologically held-out test period.

- **Backtest metrics**: net PnL, Sharpe ratio, max drawdown, turnover, hit rate, all net of fees.
- **Baselines**: buy & hold, a simple momentum / moving-average crossover rule, and a direct (non-flow, non-looped)
  predictor of the oracle trajectory using the same backbone. Rule-based baselines are run with free execution and
  with the step limit; the stronger variant is the bar to beat.
- **Inference-time scaling**: how the metrics change with the number of flow steps `n`, the stochasticity `γ`, and the
  number of ensembled samples `K`.
- **Ablations**: pseudotargets on/off, ODE vs. SDE, mean vs. best-Q aggregation, readout step 1 vs. later steps /
  prefix mean, execution with vs. without the step limit.

### Baseline results (val split, 5m bars — current default)
Decisions 2024-01-03 .. 2024-12-31 (104,832 bars), cost 0.05%, starting flat, free execution plus step-limited
(`clip`) variants of the rule-based baselines. MA crossover 1 day / 1 week, momentum 1 week. The decision range starts
earlier than in the 1h run (the purge gap and lookback are shorter in wall-clock time), so the numbers are not directly
comparable across intervals (e.g. buy & hold 0.88 here vs 0.97 at 1h).

| Policy                          | Net PnL | Fees   | Ann. return | Ann. vol | Sharpe  | Max DD | Turnover/bar | Hit rate |
|---------------------------------|---------|--------|-------------|----------|---------|--------|--------------|----------|
| flat                            | 0.000   | 0.000  | 0.0%        | 0.0%     | 0.00    | 0.000  | 0.000        | –        |
| buy & hold, free                | 0.878   | 0.001  | 88.1%       | 54.2%    | 1.63    | 0.358  | 0.000        | 50.2%    |
| buy & hold, clip                | 0.877   | 0.001  | 88.0%       | 54.2%    | 1.62    | 0.358  | 0.000        | 50.2%    |
| random, free                    | -34.541 | 34.943 | -3466.0%    | 31.8%    | -108.91 | 34.542 | 0.667        | 50.0%    |
| random, clip                    | -5.046  | 4.980  | -506.3%     | 11.7%    | -43.13  | 5.047  | 0.095        | 49.9%    |
| MA crossover (1d / 1w), free    | 0.433   | 0.068  | 43.4%       | 54.2%    | 0.80    | 0.428  | 0.001        | 49.8%    |
| MA crossover (1d / 1w), clip    | 0.486   | 0.067  | 48.7%       | 54.0%    | 0.90    | 0.422  | 0.001        | 49.8%    |
| momentum (1w), free             | -0.769  | 0.846  | -77.2%      | 54.2%    | -1.42   | 1.120  | 0.016        | 49.7%    |
| momentum (1w), clip             | -0.265  | 0.235  | -26.6%      | 53.0%    | -0.50   | 0.758  | 0.004        | 49.7%    |
| oracle, receding (hindsight)    | 24.806  | 3.378  | 2489.2%     | 41.0%    | 60.69   | 0.044  | 0.064        | 57.0%    |

Findings at 5m:
- **Bars to beat on val**: buy & hold (Sharpe 1.63) and MA crossover, clip (Sharpe 0.90).
- **Much more headroom**: the oracle earns 24.8 (~28× buy & hold, vs ~8× at 1h), because it can exploit intraday
  moves; its fees are 3.4, i.e. it still trades heavily (turnover 0.064 per 5 minutes).
- **Fees matter ~12× more per unit of time.** Random trading loses 34.5 from fees alone, and weekly momentum loses
  0.85 in fees from flipping back and forth around its threshold at 5m resolution. A model that rebalances its expected
  position every bar will pay for small changes, so the no-trade band (roadmap Phase 4) becomes more relevant.
- The 5m targets have the same shape as at 1h (train split, uniform a_0: ~14 bars per direction, mean |position|
  0.62, 3.5% flat), i.e. ~70 minutes per direction in wall-clock time.
- Data quality: 731,993 bars without gaps; 31 zero-volume flat bars (runs up to 55 min, e.g. 2021-03-02) are exchange
  maintenance filled in by Binance and currently count as valid (negligible).

### Oracle upper-bound sweep (val split, 5m bars)
Reduced grid, planning cost 0.05%, same decision bars for all points (2024-01-03 .. 2024-12-31; buy & hold 0.88,
Sharpe 1.63). Results: `outputs/sweep_oracle/20260915-175514/`.

| max step | net PnL (H = 16 / 32 / 64) | Sharpe (H = 64) | turnover/bar | bars per direction (H = 16 / 32 / 64) |
|----------|----------------------------|-----------------|--------------|---------------------------------------|
| 0.1      | 23.50 / 24.69 / 24.81      | 60.7            | 0.064        | 16.0 / 18.5 / 19.4                    |
| 0.2      | 33.11 / 33.32 / 33.32      | 81.2            | 0.112        | 9.9 / 10.5 / 10.5                     |
| 0.3      | 39.30 / 39.36 / 39.36      | 95.2            | 0.152        | 15.7 / 16.1 / 16.1                    |

The 1h conclusions carry over: the horizon saturates by `H` ≈ 32 (64 loses nothing), larger steps buy headroom with
more turnover, and every setting leaves ≥ 27× buy & hold's PnL. Defaults stay at max step 0.1, `H` = 64.
Caveat on "bars per direction": it counts sign changes, and a step of 0.2 crosses exactly through 0 (0.2 → 0 → -0.2,
two sign changes) while 0.3 can jump from 0.1 to -0.2 (one), so the metric is not monotone in the step size; turnover
is the cleaner smoothness measure.

### Baseline results (val split, 1h bars — superseded by the 5m results above)
2024-01-25 .. 2024-12-31, 1h bars, cost 0.05%, starting flat. PnL is additive in units of notional. BTC rose strongly
in 2024, so buy & hold is a high bar on this split. `free` = no step limit at execution (default), `clip` = executed
moves limited to ±0.1 per bar.

| Policy                        | Net PnL | Fees  | Ann. return | Ann. vol | Sharpe | Max DD | Turnover/bar | Hit rate |
|-------------------------------|---------|-------|-------------|----------|--------|--------|--------------|----------|
| flat                          | 0.000   | 0.000 | 0.0%        | 0.0%     | 0.00   | 0.000  | 0.000        | –        |
| buy & hold, free              | 0.973   | 0.001 | 103.9%      | 52.2%    | 1.99   | 0.345  | 0.000        | 51.2%    |
| buy & hold, clip              | 0.977   | 0.001 | 104.3%      | 52.2%    | 2.00   | 0.345  | 0.000        | 51.2%    |
| random, free                  | -2.930  | 2.790 | -312.9%     | 30.2%    | -10.35 | 2.942  | 0.680        | 50.0%    |
| random, clip                  | -0.498  | 0.391 | -53.2%      | 11.4%    | -4.66  | 0.507  | 0.095        | 49.7%    |
| MA crossover (24 / 168), free | 0.562   | 0.065 | 60.0%       | 52.2%    | 1.15   | 0.398  | 0.016        | 49.9%    |
| MA crossover (24 / 168), clip | 0.589   | 0.055 | 63.0%       | 49.9%    | 1.26   | 0.319  | 0.014        | 50.1%    |
| momentum (168), free          | -0.257  | 0.245 | -27.5%      | 52.3%    | -0.53  | 0.720  | 0.060        | 49.5%    |
| momentum (168), clip          | -0.029  | 0.070 | -3.1%       | 49.7%    | -0.06  | 0.603  | 0.017        | 49.5%    |
| oracle, receding (hindsight)  | 7.966   | 0.333 | 850.9%      | 39.2%    | 21.69  | 0.033  | 0.081        | 56.3%    |

Bars to beat on val: buy & hold (Sharpe 2.00) and MA crossover, clip (Sharpe 1.26). The oracle is unaffected by the
execution rule because it respects its own step limit.

The receding-horizon oracle (first step of the H = 64 oracle, re-planned every bar) is the upper bound for any model
under this setup: ~8× buy & hold's PnL at lower volatility, so the targets leave a lot of headroom.

### Oracle upper-bound sweep (val split, 1h bars)
Receding-horizon oracle over max step × horizon `H` × oracle planning cost, all backtested at the real fee (0.05%) and
step limit on the same decision bars (2024-01-27 .. 2024-12-31). Buy & hold on this range: net PnL 0.93, Sharpe 1.91.
Full grid: `outputs/sweep_oracle/20260915-165311/` (`results.csv`, `heatmaps.png`). Planning cost 0.05%:

| max step | net PnL (H = 8 / 16 / 32 / 64) | Sharpe (H = 64) | turnover/bar | bars per direction (H = 64) |
|----------|--------------------------------|-----------------|--------------|-----------------------------|
| 0.1      | 6.42 / 7.60 / 7.89 / 7.91      | 21.7            | 0.08         | 22.7                        |
| 0.2      | 10.14 / 10.60 / 10.63 / 10.63  | 27.8            | 0.15         | 10.8                        |
| 0.3      | 12.43 / 12.58 / 12.59 / 12.59  | 33.0            | 0.21         | 11.9                        |
| 0.5      | 15.53 / 15.54 / 15.54 / 15.54  | 40.7            | 0.33         | 3.5                         |
| none     | 26.01 (all H)                  | 70.4            | 0.85         | 2.4                         |

Findings:
- **Horizon saturates early.** The first step only depends on the next ~20–30 bars: with max step 0.1, PnL stops
  improving at `H` ≈ 32 (64 and 128 are identical); with larger steps already at `H` ≈ 16. `H` = 64 loses nothing.
- **Planning cost barely matters.** Planning at 0.2% instead of 0.05% costs ~1.5% of oracle PnL and makes targets
  somewhat calmer (turnover 0.08 → 0.06, direction runs 22.7 → 24.5 bars at step 0.1).
- **max step trades headroom for smoothness.** Every setting leaves at least ~8× buy & hold's PnL, so headroom is not
  the constraint; what the upper bound cannot measure is learnability. Step 0.1 gives by far the most persistent
  targets (~23 bars per direction), larger steps approach the noisy bang-bang regime (3.5 bars at 0.5).
- **Decided defaults** (confirmed): max step 0.1, `H` = 64, oracle planning cost = fee (0.05%).
  The final choice of max step should come from trained models on val (Phase 4), comparing 0.1 and 0.2.

## Success criteria
The project counts as a success only if all three hold out of sample:
1. **Profitable vs. baselines**: net PnL and Sharpe beat buy & hold, the MA crossover and the direct predictor (for
   rule-based baselines, the stronger of the free and step-limited variants).
2. **Looped flow beats the direct predictor** with the same backbone.
3. **Inference-time scaling holds**: more compute (steps `n`, samples `K`) measurably improves trading metrics.

## Compute
Cluster / cloud GPUs are available, so paper-scale models and hyperparameter sweeps (bar interval, lookback, `H`,
`c`, `γ`, `K`) are feasible early on.

## Deferred
- Funding rates (oracle and backtest use price returns and fees only for now).
- Slippage on top of the taker fee.
- Calendar features (hour-of-day, day-of-week).
- Multiple assets.
- Walk-forward evaluation.

## Open questions
- **Tolerance `τ` of the confidence head**: 0.1 is a first guess; check on train/val how often it is reached at
  different flow steps (too rare → halting never triggers, too common → halting too early).
- **Readout step**: step 1 vs. later steps / prefix mean (to be compared on val with a trained model).
- **Choice of `max_step` and `H`**: upper-bound sweep done (see Evaluation → Oracle upper-bound sweep); defaults
  confirmed at 0.1 and 64, final max step to be decided with trained models (0.1 vs 0.2).
- Earlier finding without the step limit: longer bar intervals do not smooth the oracle (returns grow with the
  interval; 4h holds ~2.0 bars, 1d ~1.8 bars at 0.05%); only a higher cost does.
