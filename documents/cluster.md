# Cluster setup (DTU HPC, LSF)

SSH-only workflow: clone the repo on the cluster, copy the three things git does not carry (market data, `.env`,
and any checkpoints), then submit `train.sh` with `bsub`.

Paths below assume the repo lives at `$HOME/looped-flows` (`/zhome/d3/0/155487/looped-flows` — the same path is
hard-coded in `train.sh`; change both together if it moves).

## 1. Clone and create the environment

```sh
ssh <user>@login.hpc.dtu.dk
cd ~
git clone https://github.com/<user>/looped-flows.git
cd looped-flows
```

`uv` installs Python 3.14 and the dependencies from `uv.lock` on first use. If `uv` is not on the cluster:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # login node has outbound network
```

The lock file pins the CPU torch build used locally. On the cluster install the CUDA build instead, matching the
driver's CUDA version (`nvidia-smi`):

```sh
uv sync
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run that check on a *compute* node (`a100sh` for an interactive GPU shell) — login nodes have no GPU, so
`cuda.is_available()` is `False` there regardless.

## 2. Get the market data across

**The cache is git-ignored** (`data/` in `.gitignore`), and compute nodes have no outbound network, so the parquet
cache must be in place before the job starts. It is small — 39 MB in total, 35 MB of it the 5m bars.

Each interval is two files that must travel together: the `.parquet` with the bars and a `.json` sidecar recording
`fetched_start`, which `src/data.py` uses to decide what is missing. Copy the whole directory from Windows:

```powershell
# from the repo root on the local machine
ssh <user>@login.hpc.dtu.dk "mkdir -p ~/looped-flows/data/raw"
scp data/raw/* <user>@login.hpc.dtu.dk:~/looped-flows/data/raw/
```

Only the 5m bars are needed for the current default config; `futures_BTCUSDT_5m.{parquet,json}` alone is enough if
you want a smaller transfer.

Verify on the cluster, without touching the network:

```sh
uv run python -m scripts.prepare_data --config configs/default.yaml data.update=false
```

Alternative: the login node does have outbound network, so `data.update=true` there will download from Binance
directly and build the cache in place (~20 min for the full 5m history). Never leave `data.update=true` in a
submitted job — the fetch will fail on the compute node.

## 3. Weights & Biases

`.env` holds `WANDB_API_KEY` and is git-ignored — copy it separately and keep it out of commits and logs:

```powershell
scp .env <user>@login.hpc.dtu.dk:~/looped-flows/.env
```

If compute nodes cannot reach wandb.ai, run offline and sync afterwards from the login node:

```sh
# in train.sh, before the uv run line
export WANDB_MODE=offline
# later, on the login node
uv run wandb sync outputs/train/<run>/wandb/offline-run-*
```

Or disable it entirely for a throwaway run with `wandb.enabled=false`.

## 4. Submit

```sh
bsub < train.sh
bstat                      # or: bjobs
tail -f outputs/cluster/<jobid>.out
```

`train.sh` requests one 80 GB A100 for 24 h and runs:

```sh
uv run python -m scripts.train --config configs/default.yaml train.method=looped_flow data.update=false
```

Change `train.method=direct` (or add further `key=value` overrides) to train the baseline. Each run writes
`outputs/train/<timestamp>-<method>/` containing the resolved `config.yaml`, `metrics.jsonl` and `checkpoints/`.

## 5. Get results back

`outputs/` is git-ignored, so checkpoints come back over `scp`. Metrics are already in wandb.

```powershell
scp -r <user>@login.hpc.dtu.dk:~/looped-flows/outputs/train/<run> outputs/train/
```

Checkpoints embed their config, so `scripts/backtest.py` can load one locally with
`backtest.checkpoint=outputs/train/<run>/checkpoints/step_*.pt` — though sampling at paper scale is far too slow
on CPU, so backtests belong on the cluster too.
