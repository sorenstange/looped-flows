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

**Oracle regimes on the 1h train split** (a_0 = 0, 5000 evenly spaced samples, `H` = 64):

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
  experiment. Defaults to start from: 1h bars, lookback 256, `H` = 64.
- Returns are **simple returns** of each bar (close to close); positions have constant notional, so PnL is additive.
- **Missing or malformed bars** are put on a regular time grid (filled flat at the previous close) and flagged
  invalid; any sample whose context or horizon touches an invalid bar is dropped.
- **Splits**: a single chronological train / validation / test split, with a purge gap of at least `lookback + H` bars
  between consecutive sets so that no window or oracle horizon overlaps across them.

## Model
**Inputs (conditioning `c`)**: stationary per-bar features computed from the lookback window:
- log return `log(close_t / close_{t−1})`
- open, high and low relative to close (log ratios)
- log volume (`log1p`), z-scored over the window
- plus the current position `a_0`

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
- **ACT / confidence head `q`**: adapted to predict whether the **first (executed) step** matches the oracle. Exact
  match of the whole trajectory, as in the paper, is practically unreachable on noisy market data. `q` drives halting
  during training and best-Q selection at inference, and is a candidate trade-confidence signal.
- **Pseudotargets** (Appendix D): included as a switch. The paper's justification assumes one solution per problem,
  which does not hold here (the same past can lead to different oracle trajectories), so their effect must be checked
  with an ablation.
- **SDE sampler** (Algorithm 2, `γ > 0`), with `γ = 0` (Euler ODE) as a comparison.
- **Best-Q ensembling** over `K` sampled trajectories.

## Usage: receding horizon
At every bar the model generates trajectories of length `H`, but only the **first allocation is executed**; at the
next bar it re-plans. The trajectory serves as a plan that gives the first action context.

The executed allocation is the **expected level** of the first step, a fractional value in [-1, 1], so that model
uncertainty acts as position sizing:
- Default: sample `K` trajectories, pick the one with the highest `q` (best-Q), and execute the expected level of its
  first step.
- Alternative for comparison: average the first-step expected level over all `K` samples.

The executed allocation becomes `a_0` for the next bar.

## Evaluation
All evaluation is on the chronologically held-out test period.

- **Backtest metrics**: net PnL, Sharpe ratio, max drawdown, turnover, hit rate, all net of fees.
- **Baselines**: buy & hold, a simple momentum / moving-average crossover rule, and a direct (non-flow, non-looped)
  predictor of the oracle trajectory using the same backbone.
- **Inference-time scaling**: how the metrics change with the number of flow steps `n`, the stochasticity `γ`, and the
  number of ensembled samples `K`.
- **Ablations**: pseudotargets on/off, ODE vs. SDE, best-Q vs. mean aggregation.

## Success criteria
The project counts as a success only if all three hold out of sample:
1. **Profitable vs. baselines**: net PnL and Sharpe beat buy & hold, the MA crossover and the direct predictor.
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
- **Step limit at execution.** The oracle never moves more than 0.1 per bar, but the executed expected level is not
  constrained. Clip the executed allocation to `a_0 ± max_step` (consistent with the targets), or let the model move
  freely?
- **First step is nearly determined by `a_0`.** With the step limit the first target is one of `a_0 - 0.1`, `a_0`,
  `a_0 + 0.1` (snapped to the grid), so first-step accuracy and the first-step ACT head reduce to a 3-way direction
  choice. The ACT target and the evaluation of trajectory quality may need rethinking.
- **Choice of `max_step` and `H`**: with 0.1 per bar a full flip takes 20 of the 64 bars. Both are worth sweeping.
- Earlier finding without the step limit: longer bar intervals do not smooth the oracle (returns grow with the
  interval; 4h holds ~2.0 bars, 1d ~1.8 bars at 0.05%); only a higher cost does.
- Scale of the return features: only log volume is standardized per window so far; raw log returns are ~1e-2.
