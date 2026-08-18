# PI-DEX Training Guide

本文说明当前 **`joint_29d`** 训练怎么跑、配置项做什么、本地开发机怎么测，以及火山引擎 / torchrun 多节点怎么部署。动作语义与 checkpoint 契约见 [pytorch.md](pytorch.md)；验收留证见 [server-validation.md](server-validation.md)。推理机环境见 [inference-env.md](inference-env.md)。

> **范围：** 本文只覆盖 first-party runner `pi_dex.training_runner`。`cartesian_31d`、FSDP/LoRA/AMP、快系统不在此文档。

---

## 1. 训练方法概览

训练链路固定为：

```text
审阅 observation contract
    → validate-data
    → compute-norm-stats
    → convert / 校验 pi05_base（一次性）
    → train（单机或 DDP）
    → resume（可选）
```

| 步骤 | 作用 |
|------|------|
| **validate-data** | 发现 episode、按 split 过滤、读一个样本并检查 state/image/action shape |
| **compute-norm-stats** | 在 packing / interleave **之前** 算 `state` + 双手 `[29]` quantile stats，写入 assets |
| **train** | 加载 converted `pi05_base` → full fine-tune → 原子发布 checkpoint |
| **resume** | 从已发布目录恢复 model/optimizer/RNG/采样游标，继续写到**新的** checkpoint 目录 |

入口始终是 launcher + runner：

```bash
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode <mode> \
  ...runner 参数...
```

`--` 前是 launcher 参数；`--` 后是 `training_runner` 参数。不得直接调用 `openpi/scripts/train_pytorch.py`（其 loss 对 32D padding 做 mean，不符合 PI-DEX）。

---

## 2. 环境与路径约定

### 2.1 Conda

```bash
source /mnt/netdata/Team/Personal/congsheng/miniconda/etc/profile.d/conda.sh
conda activate pi-dex
# 首次或依赖变更：
# bash scripts/setup_conda_env.sh
# pip install -e .
```

推理机 / 从端环境对齐开发机：见 [inference-env.md](inference-env.md)
（`scripts/setup_inference_env.sh` + `configs/inference/pip-lock.txt`）。

### 2.2 建议环境变量

按本机实际路径改写后导出：

```bash
export PI_DEX_REPO=/mnt/netdata/Team/Personal/congsheng/PI-DEX
export PI_DEX_ARTIFACTS=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts
export OPENPI_DATA_HOME="${PI_DEX_ARTIFACTS}/openpi-data"
export DATASET_ROOT=/mnt/netdata/Team/Academic/Data/North/SharpaOpenData/ClearPlate
export CONTRACT="${PI_DEX_REPO}/configs/site/joint_29d_observation.reviewed.json"
export CONVERTED_BASE="${PI_DEX_ARTIFACTS}/converted/pi05_base-pytorch-bfloat16-K8"
export EXPECTED_BASE_SHA256=2f8539e2308611ea6fff84a5d7774f80d7c177c624769ff842008cf85dea9eeb
export ASSETS_DIR="${PI_DEX_ARTIFACTS}/assets"
export ASSET_ID=sharpa_joint_29d
```

| 变量 / 路径 | 功能 |
|-------------|------|
| `OPENPI_DATA_HOME` | OpenPI 缓存（tokenizer、下载等）；训练/推理进程都应设置 |
| `DATASET_ROOT` / `OPENDATA_ROOT` | SharpaOpenData **根**可覆盖全部任务；单任务子目录仅训该任务。数据准备见 [dataset.md](dataset.md) |

| `CONTRACT` | **已审阅** observation contract；决定 state 列、相机槽、K、split、prompt |
| `CONVERTED_BASE` | JAX→PT 转换后的 `pi05_base` 目录（含 `model.safetensors`） |
| `EXPECTED_BASE_SHA256` | 训练时校验权重字节，防静默换盘 |
| `ASSETS_DIR` / `ASSET_ID` | norm stats 根目录与资产名；正式训练禁止与 Cartesian 共用 |
| `max_token_len` | `create_pi05_model_config` 默认 **448**（全量 OpenData pi0.5 扫描覆盖最长指令；旧 200 会系统性截断） |

### 2.3 制品目录结构（示例）

```text
pi-dex-artifacts/
  openpi-data/                          # OPENPI_DATA_HOME
  converted/pi05_base-pytorch-bfloat16-K8/
    model.safetensors
  assets/sharpa_joint_29d/
    norm_stats.json
  runs/
    formal-joint29d-step2/              # 一次 train 发布目录
      model.safetensors
      optimizer.pt
      rng.pt
      train_state.json
      parameter_manifests.json
      pi_dex.json
      assets/<asset_id>/norm_stats.json
```

---

## 3. 配置信息与功能

### 3.1 Observation contract（数据语义）

文件：`configs/site/joint_29d_observation.reviewed.json`（训练必须用 **reviewed**）。

| 字段 | 功能 |
|------|------|
| `physical_horizon` / `K` | 每手动作窗长度；模型 `action_horizon = 2K` |
| `control_frequency_hz` / `timebase` | 与 HDF5 `aligned_index` 对齐的时间基准 |
| `state_columns` | 拼成 `state` 向量的路径与 slice（当前 65D） |
| `image_slots` | Sharpa 相机 → OpenPI `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` |
| `prompt_policy` | 从 `anno.json` 取 `tags.task_instruction` |
| `split_policy` | `episode_hash_stratified_by_task` 划分 train/validation/test |
| `review_status` | 非 `reviewed` 时正式 `train` 拒绝（除非显式 smoke 开关） |

未审阅模板：`configs/site/joint_29d_observation.unreviewed.json`，仅用于对照/管道烟测。

### 3.2 Runner CLI（`--` 之后）

| 参数 | 默认 | 功能 |
|------|------|------|
| `--mode` | （必填） | `validate-data` / `compute-norm-stats` / `train` / `synthetic-smoke` |
| `--observation-contract` | （必填） | contract JSON 路径 |
| `--dataset-root` | | HDF5 数据集根；除 `synthetic-smoke` 外必填 |
| `--split` | `train` | `train` / `validation` / `test` |
| `--assets-dir` | | norm stats 根；`train` 必填 |
| `--asset-id` | `sharpa_joint_29d` | OpenPI asset 子目录名 |
| `--pytorch-weight-path` | | converted base 目录；`train` 必填，禁止随机初始化 |
| `--expected-base-sha256` | | 可选；加载时校验权重 SHA-256 |
| `--checkpoint-dir` | | **run 根目录**；其下按步写入 `<step>/`；非空目录拒绝覆盖（除非 resume） |
| `--resume-from` | | 已发布的某步目录（如 `run/500`）；恢复后继续写入同一或新的 run 根 |
| `--save-interval` | `500` | 每隔 N step 存盘；`0` 仅存最终步（最终步总会存） |
| `--log-interval` | `10` | 平均 loss 打 stdout / wandb 的间隔 |
| `--wandb` / `--no-wandb` | on | rank0 将 `loss` / `grad_norm` / `lr` 打到 W&B；认证只用环境变量 `WANDB_API_KEY` |
| `--wandb-project` | `pi-dex` | W&B project |
| `--wandb-entity` | | 可选 entity |
| `--wandb-run-name` | `run-id` | W&B run 名 |
| `--run-id` | 自动生成 | 写入 `train_state.json` 的实验 ID |
| `--device` | `cuda` | 单进程设备；DDP 下按 `LOCAL_RANK` 映射到 `cuda:{local_rank}` |
| `--dtype` | `bfloat16` | 与 model dtype 一致；可部署路径用 bfloat16 |
| `--batch-size` | `1` | **每卡 local batch**；global = local × world_size |
| `--max-steps` | 无限制 | 最多训练多少个 **local** batch（每 rank） |
| `--learning-rate` | `1e-5` | AdamW lr |
| `--grad-clip-norm` | `1.0` | 梯度裁剪 |
| `--seed` | `0` | 样本顺序与 dataloader 种子；resume 必须一致 |
| `--distributed` | off | 强制 DDP；`torchrun` 设置了 `RANK`/`WORLD_SIZE` 时也会自动开 |
| `--max-episodes` / `--max-samples` | | 调试截断 |
| `--allow-unreviewed-contract` | | 允许未审阅 contract（数据校验等） |
| `--allow-unreviewed-train-smoke` | | 与上一项同时开才允许未审阅 `train`（仅管道烟测） |
| `--robot-id` 等 | 见 CLI help | 写入 `BimanualActionSpec` 的本体 / 语义版本 / `clock_domain` |
| `--output-json` | | 把本次 run 摘要落到文件 |

### 3.3 Launcher CLI（`--` 之前）

| 参数 | 功能 |
|------|------|
| `--action-representation joint_29d` | 选择 29D 关节表示；Joint **禁止** FK factory |
| `--runner pi_dex.training_runner:run` | first-party 训练入口 |
| `--fk-provider-factory` | 仅 `cartesian_31d` 使用 |

### 3.4 火山 / 分布式环境变量

平台注入（见 `configs/volc/joint_29d_ddp.example.env`）：

| 变量 | 功能 |
|------|------|
| `MLP_WORKER_NUM` | 节点数 → `torchrun --nnodes` |
| `MLP_WORKER_GPU` | 每节点 GPU 数 → `--nproc_per_node` |
| `MLP_ROLE_INDEX` | 当前节点 rank → `--node_rank` |
| `MLP_WORKER_0_HOST` / `PORT` | master 地址端口 |

`torchrun` 再注入标准变量：`RANK`、`WORLD_SIZE`、`LOCAL_RANK`、`MASTER_ADDR`、`MASTER_PORT`。

### 3.5 Checkpoint 内关键配置

| 文件 | 功能 |
|------|------|
| `model.safetensors` | 解包后的 OpenPI 权重（DDP 时 unwrap 再存） |
| `optimizer.pt` / `rng.pt` | 优化器与 RNG |
| `train_state.json` | `global_step`、`run_id`、`sampler_state`、split/dataset manifest、parent base |
| `sampler_state` | `seed`、`order_sha256`、`next_sample_index`、`batch_size`、`world_size`、`global_batch_size` |
| `pi_dex.json` | 动作契约 + norm/weights 指纹 |
| `parameter_manifests.json` | 全参 / 可训参 manifest（full FT 要求集合一致） |
| `assets/<asset_id>/norm_stats.json` | 随 ckpt 复制的归一化资产 |

---

## 4. 本地开发机：训练测试

以下命令在已 `conda activate pi-dex` 且导出第 2 节变量后执行。

### 4.1 无 GPU 管道烟测

```bash
cd "${PI_DEX_REPO}"
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode synthetic-smoke \
  --observation-contract "${CONTRACT}" \
  --output-json "${PI_DEX_ARTIFACTS}/runs/synthetic-smoke.json"
```

### 4.2 校验真实数据

```bash
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode validate-data \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split train \
  --output-json "${PI_DEX_ARTIFACTS}/runs/validate-train.json"
```

### 4.3 计算 normalization（训练前置）

```bash
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode compute-norm-stats \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --output-json "${PI_DEX_ARTIFACTS}/runs/norm.json"
```

产物：`${ASSETS_DIR}/${ASSET_ID}/norm_stats.json`。

### 4.4 单机短训（功能测试）

```bash
CKPT="${PI_DEX_ARTIFACTS}/runs/dev-joint29d-step2"
rm -rf "${CKPT}"   # 仅开发机；正式环境禁止覆盖已有目录

pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --pytorch-weight-path "${CONVERTED_BASE}" \
  --expected-base-sha256 "${EXPECTED_BASE_SHA256}" \
  --checkpoint-dir "${CKPT}" \
  --run-id "dev-joint29d" \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-steps 2 \
  --save-interval 1 \
  --log-interval 1 \
  --no-wandb \
  --learning-rate 1e-5 \
  --seed 0 \
  --output-json "${PI_DEX_ARTIFACTS}/runs/dev-joint29d-step2.json"
```

产物示例：`${CKPT}/1/`、`${CKPT}/2/`（每步一份完整 ckpt）。正式训练默认 `--save-interval 500`，并用 `--wandb`：

```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 可选：WANDB_ENTITY / WANDB_PROJECT 通过脚本或 CLI 传入
```

不要依赖交互式 `wandb login`。冒烟请加 `--no-wandb`。

### 4.5 Resume 测试

```bash
CKPT2="${PI_DEX_ARTIFACTS}/runs/dev-joint29d-resume-step4"
rm -rf "${CKPT2}"

pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --pytorch-weight-path "${CONVERTED_BASE}" \
  --expected-base-sha256 "${EXPECTED_BASE_SHA256}" \
  --resume-from "${CKPT}/2" \
  --checkpoint-dir "${CKPT2}" \
  --run-id "dev-joint29d" \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-steps 2 \
  --save-interval 1 \
  --no-wandb \
  --seed 0 \
  --output-json "${PI_DEX_ARTIFACTS}/runs/dev-joint29d-resume-step4.json"
```

Resume 要求：`--seed`、`--batch-size`、`world_size`、dataset 长度与 `order_sha256` 与旧 ckpt 一致。`--resume-from` 指向**某一步子目录**。
### 4.6 本地单机多卡 DDP 烟测

```bash
torchrun --standalone --nproc-per-node=2 \
  "$(which pi-dex-train-pytorch)" \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --pytorch-weight-path "${CONVERTED_BASE}" \
  --expected-base-sha256 "${EXPECTED_BASE_SHA256}" \
  --checkpoint-dir "${PI_DEX_ARTIFACTS}/runs/dev-ddp-n2" \
  --batch-size 1 \
  --max-steps 2 \
  --seed 0 \
  --distributed
```

说明：

- `--batch-size 1` × 2 GPU → **global batch = 2**
- 只有 rank 0 写 checkpoint
- 也可用火山入口在本地 dry-run：

```bash
export MLP_WORKER_NUM=1 MLP_WORKER_GPU=2 MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=127.0.0.1 MLP_WORKER_0_PORT=29500
pi-dex-volc-train --dry-run -- \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train ...
```

### 4.7 单元测试（不加载大模型）

```bash
cd "${PI_DEX_REPO}"
pytest tests/test_ddp_volc_realtime.py tests/test_openpi_integration.py -q -m "not manual"
```

---

## 5. 多节点训练部署

### 5.0 通用入口（任意节点数 / 卡数）

`scripts/train_ddp.sh` 按 `NNODES × NPROC_PER_NODE` 自动选择单进程或 `torchrun`，
也可 `USE_VOLC=1` 走火山 MLP：

```bash
# 单机：自动检测 GPU 数；world_size==1 时不启 torchrun
bash scripts/train_ddp.sh \
  --dataset-root "${PI_DEX_ARTIFACTS}/prepared/Insert_Battery" \
  --assets-dir "${PI_DEX_ARTIFACTS}/assets-Insert_Battery" \
  --asset-id sharpa_joint_29d_insert_battery \
  --robot-id POC22005 \
  --max-steps 1000 \
  --batch-size 1

# 单机 8 卡
NPROC_PER_NODE=8 bash scripts/train_ddp.sh --dataset-root ... --assets-dir ... --asset-id ...

# 2 节点 × 8 卡（每台机器各跑一条，NODE_RANK 不同）
NNODES=2 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  bash scripts/train_ddp.sh --dataset-root ... --assets-dir ... --asset-id ...

# 火山
USE_VOLC=1 bash scripts/train_ddp.sh --dataset-root ... --assets-dir ... --asset-id ...
```

`--batch-size` 仍是 **per-rank local batch**；`dry-run` 可打印最终命令。

### 5.1 推荐入口（火山引擎 MLP）

**自定义启动命令**（脚本会 `conda activate pi-dex`、校验 `MLP_*` / `WANDB_API_KEY`，并带上
最新的 `save-interval` / `wandb` / Insert_Battery 默认路径）：

```bash
# 任务环境变量里先配好（不要把 key 写进仓库）:
#   WANDB_API_KEY=...
# 可选 source 默认集:
#   set -a; source configs/volc/joint_29d_insert_battery.8gpu.env; set +a

bash /mnt/netdata/Team/Personal/congsheng/PI-DEX/scripts/volc_ddp_train.sh
```

单节点 8 卡：平台侧 `MLP_WORKER_NUM=1`、`MLP_WORKER_GPU=8`。  
两节点 8 卡：`MLP_WORKER_NUM=2`、`MLP_WORKER_GPU=8`（见 `configs/volc/joint_29d_ddp.example.env`）。

常用覆盖：

```bash
MAX_STEPS=5000 SAVE_INTERVAL=500 VOLC_WANDB=1 \
  bash scripts/volc_ddp_train.sh

# 关掉 wandb
VOLC_WANDB=0 bash scripts/volc_ddp_train.sh

# 只打印 torchrun（本地先 export MLP_*）
VOLC_DRY_RUN=1 bash scripts/volc_ddp_train.sh
```

也可继续手写完整 argv（脚本遇到 `--` 后参数则不再套默认）：

```bash
bash scripts/volc_ddp_train.sh -- \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${DATASET_ROOT}" \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --robot-id POC22005 \
  --pytorch-weight-path "${CONVERTED_BASE}" \
  --expected-base-sha256 2f8539e2308611ea6fff84a5d7774f80d7c177c624769ff842008cf85dea9eeb \
  --checkpoint-dir "${PI_DEX_ARTIFACTS}/runs/volc-joint29d-$(date +%Y%m%d-%H%M%S)" \
  --run-id volc-joint29d \
  --device cuda --dtype bfloat16 \
  --batch-size 1 --max-steps 1000 \
  --save-interval 500 --log-interval 10 \
  --wandb --wandb-project pi-dex
```

脚本会组装：

```text
conda activate pi-dex
torchrun --nnodes=$MLP_WORKER_NUM \
         --nproc_per_node=$MLP_WORKER_GPU \
         --node_rank=$MLP_ROLE_INDEX \
         --master_addr=$MLP_WORKER_0_HOST \
         --master_port=$MLP_WORKER_0_PORT \
         pi-dex-train-pytorch ... --distributed
```

DDP 已默认 `find_unused_parameters=True`（pi0.5 有未参与 loss 的参数）。

### 5.2 非火山：手动 torchrun 多机

节点 0：

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=<node0_ip> --master_port=29500 \
  "$(which pi-dex-train-pytorch)" \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train ... --distributed
```

节点 1：同样命令，仅 `--node_rank=1`。

### 5.3 多机部署检查清单

1. **共享存储**：dataset、converted base、assets、checkpoint 目录对所有节点可见且一致。
2. **同一代码与环境**：各节点同一 commit、同一 conda env、`OPENPI_DATA_HOME` 一致。
3. **batch 语义**：`--batch-size` 为 per-GPU；有效 global batch = `batch_size × nnodes × nproc_per_node`。
4. **显存**：DDP 复制整模，**不降低**单卡显存；不够用时减 local batch / 分辨率，不要指望 DDP 当 FSDP。
5. **checkpoint**：rank 0 写入 `checkpoint_dir/<step>/`；run 根目录非空则拒绝覆盖。
6. **resume 多机**：`world_size` 与上次完全相同，否则拒绝；`--resume-from` 指向某一步子目录。
7. **网络**：`MASTER_PORT` 互通；NCCL 超时按集群调 `NCCL_DEBUG` / `NCCL_SOCKET_IFNAME`（站点运维约定）。
8. **环境**：入口脚本必须 `conda activate pi-dex`（或 `VOLC_SKIP_CONDA=1` 且 PATH 已正确）；`OPENPI_DATA_HOME` 与 `WANDB_API_KEY` 写入任务环境。

### 5.4 示例拓扑

| 拓扑 | `MLP_WORKER_NUM` | `MLP_WORKER_GPU` | global batch (`--batch-size 1`) |
|------|------------------|------------------|----------------------------------|
| 单节点 8 卡 | 1 | 8 | 8 |
| 两节点 8 卡 | 2 | 8 | 16 |

Insert_Battery 默认 env：`configs/volc/joint_29d_insert_battery.8gpu.env`。

---

## 6. Converted base（一次性准备）

若 `CONVERTED_BASE` 尚不存在，需先做受控转换与 parity（细节见验证清单 A.2）：

```bash
python -m pi_dex.convert_pi05 ...   # 写出 model.safetensors
python -m pi_dex.parity_pi05 ...    # JAX↔PT sample_actions 对比
```

训练侧只消费转换目录 + `--expected-base-sha256`；转换产物**不是**可部署 PI-DEX checkpoint（缺 Sharpa norm / `pi_dex.json`）。

---

## 7. 常见失败与含义

| 现象 | 含义 |
|------|------|
| `checkpoint_dir already exists` | 拒绝覆盖非空 run 根；换新目录或先处理旧 run |
| `review_status` / unreviewed | 正式 train 需要 reviewed contract |
| `expected ... sha256` 不匹配 | 权重文件被换或路径指错 |
| `resume ... world_size conflicts` | 续训卡数/节点数与上次不一致 |
| `resume ... order_sha256 mismatch` | seed 或数据集长度变了 |
| `shuffle must be False under DDP` | DDP 必须关 shuffle 以保持游标语义 |
| `distributed=False` 与已初始化 PG 冲突 | 不要在 torchrun 下强制关 DDP |
| CUDA OOM | 降 `--batch-size` 或减数据分辨率；DDP 不省显存 |
| `Expected to have finished reduction` / unused params | DDP 需 `find_unused_parameters=True`（已默认开启；与 OpenPI 一致） |
| wandb 未配置 | 设置 `export WANDB_API_KEY=...`，或冒烟时加 `--no-wandb` |

---

## 8. 相关入口速查

| 入口 | 用途 |
|------|------|
| `pi-dex-train-pytorch` | 单机 / 被 torchrun 拉起的训练 |
| `scripts/train_ddp.sh` | 通用 1..N 节点 × 1..M 卡启动 |
| `pi-dex-volc-train` / `scripts/volc_ddp_train.sh` | 火山 MLP → torchrun |
| `scripts/prepare_task_dataset.sh` / `pi-dex-prepare-dataset` | 任务数据 overlay + inventory + norm |
| `pi-dex-realtime-infer` | 真机推理（非训练；见 realtime 模块） |
| `pi-dex-serve` / `scripts/serve_joint29d.sh` | joint_29d WebSocket model server |
| `pi-dex-serve-probe` | 环回打一次 metadata + infer |
| `pi-dex-robot-client` / `scripts/robot_client_joint29d.sh` | 从端 Zenoh 桥（配 start.sh + F6 推理） |
| `scripts/setup_inference_env.sh` | 推理机环境（详见 [inference-env.md](inference-env.md)） |
| `scripts/export_inference_lock.sh` | 从当前开发 env 导出 pip-lock |
| `configs/site/joint_29d_observation.reviewed.json` | 数据契约 |
| `configs/volc/joint_29d_ddp.example.env` | 多机环境变量样例 |

正式大规模放行前，仍建议按 [server-validation.md](server-validation.md) 阶段 B.1/B.2/D 留存命令、日志与制品哈希；本文命令可复现实验，本身不等于验收 PASS。
