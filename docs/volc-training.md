# Volcano Engine (MLP) Training — Console & Env Reference

Focused checklist for submitting a **joint_29d** DDP job on Volcano Engine MLP.  
Full training pipeline (data, local smoke, resume, convert): [training.md](training.md).  
Norm stats: [norm-compute.md](norm-compute.md). Dataset prep: [dataset.md](dataset.md).

---

## 1. What you fill on the platform

| Console field | Typical value | Notes |
|---------------|---------------|--------|
| Nodes | `1` | Or `2` for multi-node |
| GPUs per node | `8` | Maps to `MLP_WORKER_GPU` |
| Image / mount | Must see `/mnt/netdata` | Shared VEFS |
| Custom start command | See §2 | Do not wrap in torchrun yourself |
| Job env / secrets | `WANDB_API_KEY=...` | Never commit the key |

**Platform injects** (do not hardcode in the sourced env file):

| Env | Meaning | torchrun flag |
|-----|---------|----------------|
| `MLP_WORKER_NUM` | Number of nodes | `--nnodes` |
| `MLP_WORKER_GPU` | GPUs per node | `--nproc_per_node` |
| `MLP_ROLE_INDEX` | This node’s rank | `--node_rank` |
| `MLP_WORKER_0_HOST` | Master host | `--master_addr` |
| `MLP_WORKER_0_PORT` | Master port | `--master_port` |

---

## 2. Custom start command (copy-paste)

**Close_Bottle_Cap_v2 (K=50, scheme-A delta, 80k steps):**

```bash
export WANDB_API_KEY="${WANDB_API_KEY}"
set -a
source /mnt/netdata/Team/Personal/congsheng/PI-DEX/configs/volc/joint_29d_close_bottle_cap_v2.k50.delta.8gpu.env
set +a
bash /mnt/netdata/Team/Personal/congsheng/PI-DEX/scripts/volc_ddp_train.sh
```

**Close_Bottle_Cap_v2 (K=50, absolute, 80k steps):** source `configs/volc/joint_29d_close_bottle_cap_v2.8gpu.env` instead.

**Insert_Battery (K=50, scheme-A delta, 80k steps):**

```bash
export WANDB_API_KEY="${WANDB_API_KEY}"
set -a
source /mnt/netdata/Team/Personal/congsheng/PI-DEX/configs/volc/joint_29d_insert_battery.k50.delta.8gpu.env
set +a
bash /mnt/netdata/Team/Personal/congsheng/PI-DEX/scripts/volc_ddp_train.sh
```

**Insert_Battery (K=50, absolute, 80k steps):** source `configs/volc/joint_29d_insert_battery.k50.8gpu.env` instead.

**Insert_Battery K=8 (legacy shorter run):** source `configs/volc/joint_29d_insert_battery.8gpu.env` instead.

`volc_ddp_train.sh` will `conda activate pi-dex`, build argv from env, and launch:

```text
torchrun --nnodes=$MLP_WORKER_NUM --nproc_per_node=$MLP_WORKER_GPU \
  --node_rank=$MLP_ROLE_INDEX \
  --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \
  pi-dex-train-pytorch ... --distributed
```

Disable wandb: set `VOLC_WANDB=0` (and you can omit `WANDB_API_KEY`).

Dry-run (print torchrun only; need stub `MLP_*` locally):

```bash
VOLC_DRY_RUN=1 bash scripts/volc_ddp_train.sh
```

---

## 3. Env file knobs (`configs/volc/*.env`)

Sourced with `set -a` before `volc_ddp_train.sh`. Important keys:

### Paths

| Variable | Role |
|----------|------|
| `PI_DEX_REPO` | Repo root |
| `PI_DEX_ARTIFACTS` | Runs / assets / prepared data |
| `CONDA_ROOT` / `PI_DEX_CONDA_ENV` | Usually `.../miniconda` + `pi-dex` |
| `OPENPI_DATA_HOME` | Tokenizer / OpenPI cache |
| `CONTRACT` | Reviewed observation contract (sets **K** / cameras / split) |
| `CONVERTED_BASE` | Init weights dir (`model.safetensors`) |
| `EXPECTED_BASE_SHA256` | Optional weight digest check |
| `DATASET_ROOT` | Prepared overlay root |
| `ASSETS_DIR` / `ASSET_ID` | Parent of `<asset-id>/norm_stats.json` |
| `ROBOT_ID` | Provenance (e.g. `POC22017`, `POC22005`) |
| `CHECKPOINT_DIR` / `RUN_ID` / `OUTPUT_JSON` | Optional; script defaults if empty |

### Train hyperparameters

| Variable | Role | Example |
|----------|------|---------|
| `MAX_STEPS` | Local batches per rank | `80000` |
| `BATCH_SIZE` | **Per-GPU** local batch | `1` → global = `1 × nnodes × nproc` |
| `LEARNING_RATE` | Peak AdamW LR | `1e-5` |
| `LR_WARMUP_STEPS` | Linear warmup | `1000` |
| `LR_DECAY_STEPS` | Cosine length (incl. warmup) | `80000` |
| `LR_END` | Cosine end LR | `1e-6` |
| `SAVE_INTERVAL` | Checkpoint every N steps | `5000` |
| `LOG_INTERVAL` | Stdout / wandb log cadence | `10` |
| `SEED` | Sample order + dataloader | `0` |
| `DEVICE` / `DTYPE` | Usually `cuda` / `bfloat16` |
| `SPLIT` | Usually `train` |
| `MAX_EPISODES` | Optional debug truncate | empty |
| `RESUME_FROM` | Path to a **step** dir | empty |
| `VOLC_WANDB` | `1` / `0` | `1` |
| `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_NAME` | Optional | `pi-dex` |
| `ACTION_MODE` | `absolute` or scheme-A `delta` | `delta` |
| `COMMAND_SEMANTICS_VERSION` | HDF5 command provenance (still absolute) | `sharpa_sdk_commanded_joint_position_absolute_v1` |

Short runs (`warmup ≥ max_steps`) keep a **constant** peak LR (smoke jobs stay at `LEARNING_RATE`).

### Do not put in the env file

- `WANDB_API_KEY` — job secret only  
- `MLP_*` — platform injects; sourcing stubs would overwrite the real master addr  

---

## 4. Ready-made env files

| File | Task | Notes |
|------|------|--------|
| `configs/volc/joint_29d_close_bottle_cap_v2.k50.delta.8gpu.env` | Close_Bottle_Cap_v2 | **K=50**, scheme-A **delta**, 80k steps |
| `configs/volc/joint_29d_close_bottle_cap_v2.8gpu.env` | Close_Bottle_Cap_v2 | **K=50**, absolute, 80k steps |
| `configs/volc/joint_29d_insert_battery.k50.delta.8gpu.env` | Insert_Battery | **K=50**, scheme-A **delta**, 80k steps |
| `configs/volc/joint_29d_insert_battery.k50.8gpu.env` | Insert_Battery | **K=50**, absolute, 80k steps |
| `configs/volc/joint_29d_insert_battery.8gpu.env` | Insert_Battery | Legacy **K=8** shorter run |
| `configs/volc/joint_29d_ddp.example.env` | Topology stub | Multi-node `MLP_*` example only |
| `configs/volc/opendata_norm.env` | Norm job | Not for DDP train |

Insert_Battery **K=50** env already points `CONTRACT` at  
`configs/site/joint_29d_observation.k50.reviewed.json` and  
`ASSET_ID=sharpa_joint_29d_insert_battery_k50` (norm under `assets-Insert_Battery/`).

---

## 5. Preconditions (before submit)

1. Reviewed contract matches K (`physical_horizon`).  
2. `norm_stats.json` exists at `${ASSETS_DIR}/${ASSET_ID}/`.  
3. `CONVERTED_BASE` exists and SHA matches if set.  
4. Prepared `DATASET_ROOT` is on shared storage.  
5. Checkpoint run root must be empty (or use a new name); resume uses `--resume-from` / `RESUME_FROM` to a **step** subdirectory.

---

## 6. What success looks like in logs

Early (not stuck):

```text
train: discovering episodes ...
build_sample_index: episodes ...
train: loading converted pi05_base ...
train: entering loop planned_batches=...
step=10 loss=... lr=...
```

`OMP_NUM_THREADS=1` right after torchrun is normal; wait for the lines above.

---

## 7. Related scripts

| Script | Role |
|--------|------|
| `scripts/volc_ddp_train.sh` | MLP train entry |
| `scripts/train_ddp.sh` | Generic / local DDP (`USE_VOLC=1` optional) |
| `scripts/volc_compute_norm.sh` | Norm only (single process, no torchrun) |
| `pi-dex-volc-train` | Thin module entry used by scripts |

Overall CLI / modes / checkpoint layout: [training.md](training.md).
