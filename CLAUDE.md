# CLAUDE.md

Research project: a **looped flows** model (Suleymanzade et al., 2026) that generates future allocation trajectories
in {-1, 0, 1} for BTCUSDT perpetuals, trained on a cost-aware PnL **oracle**.

## Source of truth
- `documents/vision.md` holds the spec and every design decision taken so far. Read it before any design or
  implementation work.
- If a decision changes, or a new one is made with the user, **update `vision.md` in the same turn**, including its
  "Deferred" and "Open questions" sections.
- Don't silently depart from `vision.md`. If the spec seems wrong or underspecified, say so and ask.
- `documents/paper.md` is a cleaned Markdown transcription of the paper (`2026_suleymanzade_looped-flows.pdf`). Read
  or grep it instead of the PDF. Equations were re-typeset by hand, so check the PDF if exact notation matters.
  Sections that matter most: Alg. 1/2 (training/inference), Eq. 9 (loss), Eq. 10 (ACT), App. A (TRM-style
  architecture), App. B (noise-sharing shortcut), App. C (SDE sampler), App. D (pseudotargets), Table 6 / App. E
  (hyperparameters).

## Environment
- Windows 11. Primary shell is PowerShell 5.1 (no `&&`); Git Bash is also available.
- Python **3.14**, managed with **uv**. Run code with `uv run python ...`; add dependencies with `uv add <pkg>` (never pip).
- Dependencies: torch, numpy, pandas, omegaconf, python-binance, requests, matplotlib.
- **Application Control blocks `pyexpat`.** Anything that imports `xml.parsers.expat` fails (e.g. `pypdf`).
  Avoid such packages, or run them isolated.
- Reading the PDF directly (only if `paper.md` is insufficient): the Read tool can't render it (no poppler). Extract
  text with `uv run --no-project --with pymupdf python -c "import pymupdf; ..."` into the scratchpad, then Read that.
- Training runs happen on cluster/cloud GPUs. Code must run on CPU for smoke tests and on CUDA without changes
  (select the device from config, never hard-code it).

## Layout
Early stage: `src/data.py` and `src/modules.py` exist but are empty. Intended split (extend as needed, and keep this
list current):
- `src/data.py`: Binance kline download/caching, feature computation, oracle DP, windowing, chronological splits
- `src/modules.py`: denoiser (joint-sequence transformer, RoPE, SwiGLU, RMSNorm, two-state (h, ℓ) recurrence), ACT head
- training loop, sampler (Alg. 2), backtest and baselines: separate modules under `src/`
- configs: omegaconf YAML. Every tunable in vision.md (bar interval, lookback, H, c, k, n, γ, K, σ, ...) is a config
  field, not a constant.
- Raw and cached market data goes under `data/`, checkpoints under `checkpoints/`, and run outputs under `runs/` or
  `outputs/`; all of these are git-ignored. Never commit data, checkpoints or run outputs.

## Invariants: check these in every change touching data, oracle or backtest
1. **No lookahead.** Model inputs at decision bar t use only bars ≤ t. Feature normalization (e.g. the volume z-score)
   uses only the window itself, never global or future statistics.
2. **Timing convention.** Allocation `a_t` is decided at the close of bar t and earns `r_{t+1}`. The oracle, the
   backtest and the baselines must all use the same convention and the same cost formula `c·|a_t − a_{t−1}|`.
3. **Oracle is exact.** DP over (time × level), starting from a continuous `a_0 ∈ [-1, 1]`, no terminal cost.
   Verify against brute-force enumeration for small H in tests.
4. **Chronological splits with purge gap** ≥ `lookback + H` bars. No shuffling across time before splitting.
5. **Only trajectory tokens are noised**; context and `a_0` tokens are clean conditioning.
6. **Stop-gradient between recurrent steps**; noise `x0`, target `x1` and context are shared across the k steps of a
   rollout, with flow times sorted ascending.
7. Backtests report results **net of fees**, and the baselines use the identical backtest engine.

## Working style
- Build in small, verifiable increments. Each component gets a quick check before moving on: a unit test, a shape
  test, or a plot or summary statistic.
- Add pytest (`uv add --dev pytest`) when the first tests are written. Priority test targets: oracle DP vs brute force,
  no-lookahead in features and windows, split purge gaps, backtest PnL on hand-computed toy series, sampler with
  γ=0 matching Euler.
- Before training a model, sanity-check the targets: oracle turnover, holding times, class balance of {-1, 0, 1},
  and oracle PnL at different costs.
- Prefer tiny configs (small model, few bars) for fast local smoke runs; paper-scale settings are for the cluster.
- Be skeptical of good backtest results: first check for leakage, cost errors and timing off-by-ones.
- When a paper recipe doesn't transfer to trading (e.g. exact-match ACT, pseudotargets on multi-solution data),
  point it out and propose an adaptation rather than copying it blindly.
- Match existing code style; type hints on public functions; keep comments sparse and explanatory.
- Git: the repo has no commits yet. Commit only when asked.
