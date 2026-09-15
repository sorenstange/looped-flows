# CLAUDE.md

Research project: a **looped flows** model (Suleymanzade et al., 2026) that generates future allocation trajectories
over 21 levels in [-1, 1] for BTCUSDT perpetuals, trained on a cost-aware, rate-limited (≤ 0.1 per bar) PnL **oracle**.

## Source of truth
- `documents/vision.md` holds the spec and every design decision taken so far. Read it before any design or
  implementation work.
- If a decision changes, or a new one is made with the user, **update `vision.md` in the same turn**, including its
  "Deferred" and "Open questions" sections.
- Don't silently depart from `vision.md`. If the spec seems wrong or underspecified, say so and ask.
- `documents/roadmap.md` is the prioritized checklist of work. When asked what's next, take the top unchecked item.
  Check items off when they land, and add new items where they belong in the order.
- `documents/paper.md` is a cleaned Markdown transcription of the paper (`2026_suleymanzade_looped-flows.pdf`). Read
  or grep it instead of the PDF. Equations were re-typeset by hand, so check the PDF if exact notation matters.
  Sections that matter most: Alg. 1/2 (training/inference), Eq. 9 (loss), Eq. 10 (ACT), App. A (TRM-style
  architecture), App. B (noise-sharing shortcut), App. C (SDE sampler), App. D (pseudotargets), Table 6 / App. E
  (hyperparameters).

## Environment
- Windows 11. Primary shell is PowerShell 5.1 (no `&&`); Git Bash is also available.
- Python **3.14**, managed with **uv**. Run code with `uv run python ...`; add dependencies with `uv add <pkg>` (never pip).
- Dependencies: torch (CPU build locally), numpy, pandas, pyarrow, omegaconf, python-binance, matplotlib; pytest (dev).
- Commands:
  - tests: `uv run pytest -q -p no:warnings` (python-binance emits noisy websocket deprecation warnings)
  - scripts: `uv run python -m scripts.<name> --config configs/<file>.yaml key=value ...` (run from the repo root;
    imports are `from src.<module> import ...`)
  - data prep / oracle report: `uv run python -m scripts.prepare_data --config configs/smoke.yaml`
  - backtest: `uv run python -m scripts.backtest --config configs/default.yaml data.update=false`
  - oracle upper-bound sweep: `uv run python -m scripts.sweep_oracle --config configs/default.yaml data.update=false`
  - list-valued overrides need quoting in PowerShell: `"backtest.policies=[buy_hold,ma_crossover]"`
- **Application Control blocks `pyexpat`.** Anything that imports `xml.parsers.expat` fails (e.g. `pypdf`).
  Avoid such packages, or run them isolated.
- Reading the PDF directly (only if `paper.md` is insufficient): the Read tool can't render it (no poppler). Extract
  text with `uv run --no-project --with pymupdf python -c "import pymupdf; ..."` into the scratchpad, then Read that.
- Training runs happen on cluster/cloud GPUs. Code must run on CPU for smoke tests and on CUDA without changes
  (select the device from config, never hard-code it).

## Layout (keep this list current)
- `src/config.py`: config schema (dataclasses), `load_config`, `save_config`, `parse_args`, `validate`
- `src/data.py`: Binance kline download (`fetch_ohlcv`), parquet cache with incremental top-up (`load_ohlcv`),
  regular time grid with a `valid` flag per bar (`clean_ohlcv`)
- `src/features.py`: `FEATURE_NAMES` (6 channels: log return, high/low vs close, taker-buy share, log volume,
  `log_vol_to_cost`), `raw_bar_features`, `bar_features` (trailing rolling scaling or per-window mode; NaN during the
  warmup), `standardize_windows`. No open-based feature: perpetual bars open at the previous close.
- `src/oracle.py`: timing convention (see module docstring), `bar_returns`, `trajectory_pnl`, `oracle_stats`,
  `tolerance_match` (confidence-head target), and the
  DP split in two: `oracle_rest_values` (backward pass, independent of a_0, batched over windows) and `oracle_step`
  (choose one step given the position); `oracle_trajectory` combines them. The backtest must reuse `trajectory_pnl` /
  the same convention.
- `src/dataset.py`: `MarketData`, `split_ranges`, `valid_anchors`, `WindowDataset` (batched indexing, oracle computed
  per batch, deterministic a_0 per (seed, epoch, anchor)), `build_datasets`, `make_loader`
- `src/backtest.py`: `History` (market data truncated to bars ≤ t), `Policy` interface, bar-by-bar `run_backtest`
  (validity check, clipping to [-1, 1] and the step limit, fees), `performance` metrics
- `src/policies.py`: `POLICY_NAMES`, `RULE_BASED`, flat / buy & hold / random / MA crossover / momentum,
  `OraclePolicy` (receding-horizon hindsight upper bound; precomputes rest values for all decision bars, then one
  `oracle_step` per bar), `make_policy`, and `readout_allocation` (K sampled trajectories' level probabilities →
  traded allocation, per `inference` config). Model policies will go here too.
- `src/backtest.py` also has `decision_range`: the decision bars of a split, shared by all backtest scripts.
- `src/modules.py` (empty): denoiser (joint-sequence transformer, RoPE, SwiGLU, RMSNorm, two-state (h, ℓ)
  recurrence), ACT head
- still to come as separate modules under `src/`: training loop, sampler (Alg. 2)
- `scripts/`: entry points (`prepare_data.py`, `backtest.py`, `sweep_oracle.py`); `tests/`: pytest suite (synthetic data, no network;
  `conftest.make_bars`, `test_backtest.toy_market`)
- Raw and cached market data goes under `data/`, checkpoints under `checkpoints/`, and run outputs under `runs/` or
  `outputs/`; all of these are git-ignored. Never commit data, checkpoints or run outputs.

## Configs
- `configs/default.yaml` mirrors the dataclass defaults in `src/config.py` (a test enforces equality). Other YAML files
  inherit with `extends: <relative path>` and override only what differs, e.g. `configs/smoke.yaml` for fast local runs.
- Precedence: dataclass defaults < `extends` chain < the YAML file < CLI `key=value` overrides. Unknown keys and wrong
  types are rejected by omegaconf; value checks live in `validate`.
- Every tunable (bar interval, lookback, H, c, k, n, γ, K, σ, model sizes, optimizer, ...) is a config field, never a
  constant. Adding a field means: dataclass field with a comment, same value in `default.yaml`, validation if needed.
- Every script saves its resolved config with `save_config` into its output directory
  (`<output_dir>/<script>/<timestamp>/config.yaml`), so any result can be reproduced from that file alone.

## Invariants: check these in every change touching data, oracle or backtest
1. **No lookahead.** Model inputs at decision bar t use only bars ≤ t. Feature normalization uses trailing statistics
   up to the bar itself (or the sample window), never global, train-set or future statistics.
2. **Timing convention.** Allocation `a_t` is decided at the close of bar t and earns `r_{t+1}`. The oracle, the
   backtest and the baselines must all use the same convention and the same cost formula `c·|a_t − a_{t−1}|`.
3. **Oracle is exact.** DP over (time × level), starting from a continuous `a_0 ∈ [-1, 1]`, no terminal cost, with
   every move (including the first from `a_0`) at most `oracle.max_step`. Verify against brute-force enumeration for
   small H in tests.
4. **Chronological splits with purge gap** ≥ `lookback + H` bars. No shuffling across time before splitting.
5. **Only trajectory tokens are noised**; context and `a_0` tokens are clean conditioning.
6. **Stop-gradient between recurrent steps**; noise `x0`, target `x1` and context are shared across the k steps of a
   rollout, with flow times sorted ascending.
7. Backtests report results **net of fees**, and the baselines use the identical backtest engine and execution rules
   as the models (free execution by default; `<name>_clip` variants are extra comparisons, not replacements).

## Working style
- Build in small, verifiable increments. Each component gets a quick check before moving on: a unit test, a shape
  test, or a plot or summary statistic.
- Tests exist for config loading, oracle DP vs brute force, cleaning, causal features, no-lookahead windows, split
  purge gaps, a_0 reproducibility, and the backtest (hand-computed toy series, engine PnL = `trajectory_pnl`,
  receding oracle with full horizon = DP optimum). Still to write when it exists: sampler with γ=0 matching Euler.
- Before training a model, sanity-check the targets: oracle turnover, holding times, class balance of {-1, 0, 1},
  and oracle PnL at different costs.
- Prefer tiny configs (small model, few bars) for fast local smoke runs; paper-scale settings are for the cluster.
- Be skeptical of good backtest results: first check for leakage, cost errors and timing off-by-ones.
- When a paper recipe doesn't transfer to trading (e.g. exact-match ACT, pseudotargets on multi-solution data),
  point it out and propose an adaptation rather than copying it blindly.
- Match existing code style; type hints on public functions; keep comments sparse and explanatory.
- Git: branch `main`, remote `origin`. Commit and push only when asked.
