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
| Token vocabulary `V`   | Allocation levels {-1, 0, 1}                                        |
| Multiple valid answers | Different plausible futures -> different sampled trajectories      |

## Training target: the oracle
The model is trained on an **oracle allocation trajectory**: the allocation that, with hindsight of the future prices
in the horizon, maximizes PnL net of transaction costs:

```
max_a  Σ_{t=1..H} a_t · r_{t+1} − c · |a_t − a_{t−1}|
s.t.   a_t ∈ {-1, 0, 1}
```

- `r_{t+1}` is the return of the bar following allocation `a_t`, `c` the proportional cost per unit of turnover
  (default **c = 0.05%**, the Binance USDT-M taker fee; a full flip from -1 to +1 costs 0.10%; configurable).
- `a_0` is the **current position** held right before the horizon. It is part of the model input, and the oracle starts
  from it, so the first step pays the true cost of changing position. `a_0` can be fractional (see Usage); in training
  it is sampled **uniformly from [-1, 1]** per example so the model learns the cost of switching from any position.
- **No terminal cost**: the position may remain open at the end of the horizon (avoids a bias toward flattening late
  in the trajectory).
- Solved exactly with dynamic programming over (time × level).
- Costs are what make the oracle non-trivial: without them it would just be `sign(r_{t+1})`. Costs create holding
  periods and flat regimes when a move does not cover the round trip.

## Representation
- Allocations are **three discrete levels {-1, 0, 1}**. With linear PnL and linear turnover costs the oracle essentially
  only ever uses these, so finer levels would add vocabulary without adding targets.
- One-hot encoded, exactly as the paper's categorical flow: Gaussian-noise interpolant on one-hots, cross-entropy
  denoising loss.
- Fractional exposure emerges from model uncertainty: the expected level `Σ p_level · level` lies in [-1, 1].

## Market and data
- **Binance USDT-M perpetual futures, BTCUSDT** to start (shorting is native).
- Bar interval, lookback length and horizon `H` are configuration parameters (omegaconf), chosen by experiment.
- **Splits**: a single chronological train / validation / test split, with a purge gap of at least `lookback + H` bars
  between consecutive sets so that no window or oracle horizon overlaps across them.

## Model
**Inputs (conditioning `c`)**: stationary per-bar features computed from the lookback window:
- log return `log(close_t / close_{t−1})`
- open, high and low relative to close (log ratios)
- log volume, z-scored over the window
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
- First milestone and its scope.
- Default bar interval / lookback / `H` to start sweeps from.
