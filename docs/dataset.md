# PI-DEX Dataset Preparation Guide

本文说明如何把 **SharpaOpenData 全部任务**接到 `joint_29d` 训练管线，以及如何计算
normalization stats。训练命令细节见 [training.md](training.md)；单 episode 磁盘格式见
[data/schema.json](../data/schema.json)。

---

## 1. 结论：如何覆盖全部任务

**不需要改代码。** 把 `--dataset-root` 指到 OpenData **根目录**（所有任务的父目录），而不是某个任务子目录：

```bash
# 错误（只有 ClearPlate）
--dataset-root /mnt/netdata/Team/Academic/Data/North/SharpaOpenData/ClearPlate

# 正确（全部任务）
--dataset-root /mnt/netdata/Team/Academic/Data/North/SharpaOpenData
```

`pi_dex.sharpa_dataset.discover_episodes` 会对 root 做 `rglob("anno.json")`，因此：

| 布局 | 是否覆盖 |
|------|----------|
| `<root>/<task>/season_*/<episode>/` | 是 |
| `<root>/easy10/<subtask>/season_*/<episode>/` | 是（嵌套子任务） |
| 缺 HDF5 / 同一目录多个 HDF5 | **跳过**该 episode |

本机实测（2026-08-15）：

| 指标 | 数值 |
|------|------|
| 任务目录 | 51（全部有可发现 episode） |
| 发现 episode | **34381** |
| 仅 ClearPlate | ~375 |

> 正式训练请使用 **新的** `--asset-id` / assets 目录。不要复用只在 ClearPlate 冒烟上算过的
> `sharpa_joint_29d` stats 去训全量库。

---

## 2. 数据在盘上的样子

```text
SharpaOpenData/                          ← 建议的 dataset-root
  ClearPlate/
    season_POC22027_..._train/
      POC22027_..._train/
        anno.json
        train_0.3.0@....hdf5
        clip_tags.json
        ...
  easy10/
    stack_blocks/
      season_.../
        ...
  ...
```

每个可用 episode 至少需要：

1. `anno.json`（prompt：`tags.task_instruction`）
2. 恰好一个匹配 `train_*.hdf5` 的文件

OpenData 根下的 `MISSING_HDF5_*.md` / `qc_failed_*.jsonl` 是质检报告，**不是**训练输入；
discover 不会把它们当成 episode。

---

## 3. 准备流程总览

```text
① 锁定 reviewed observation contract
② inventory：确认任务数 / episode 数 / split 计数
③ validate-data（建议先小规模，再全量）
④ compute-norm-stats（仅 train split；写入独立 asset_id）
⑤ train（同一 dataset-root + 同一 assets）
```

### 3.1 Foundation_Model 任务（空 `task_instruction`）

Academic `Foundation_Model/*` 常见全部 episode `tags.task_instruction` 为空，inventory
会全部 reject。不要改原盘：用 overlay（HDF5 软链 + 重写 `anno.json`）：

```bash
bash scripts/prepare_task_dataset.sh \
  --source-root /mnt/netdata/Team/Academic/Data/Foundation_Model/Insert_Battery \
  --task-name Insert_Battery \
  --default-prompt "Pick up the large battery with the right hand and insert it into the large battery compartment. Then pick up the small battery with the right hand and insert it into the small battery compartment."
```

产出：

- `pi-dex-artifacts/prepared/Insert_Battery/`（训练用 `dataset-root`）
- `pi-dex-artifacts/assets-Insert_Battery/sharpa_joint_29d_insert_battery/norm_stats.json`
- inventory / prepare meta 在 `pi-dex-artifacts/dataset/`

随后用 `scripts/train_ddp.sh --dataset-root .../prepared/Insert_Battery --robot-id POC22005 ...`。

`Close_Bottle_Cap_v2` 同样走 overlay（原盘 251 episode 的 `task_instruction` 全空；prompt
对齐 OpenData `Closethebottlecap` / Tighten Bottle Cap）。火山训练 env：
`configs/volc/joint_29d_close_bottle_cap_v2.k50.delta.8gpu.env`（**K=50** scheme-A
delta，asset `sharpa_joint_29d_close_bottle_cap_v2_k50_delta`）或
`configs/volc/joint_29d_close_bottle_cap_v2.8gpu.env`（absolute，asset
`sharpa_joint_29d_close_bottle_cap_v2_k50`）。

**Cadence：** Sharpa 原始控制约 59.4 Hz。Insert_Battery 与 OpenData 都会出现一次漏拍
（相邻 dt 误差 **16.835 ms**），从未观察到两次漏拍（>20 ms）。reviewed contract 因此把
`max_control_period_error_ms` / `max_alignment_timestamp_error_ms` /
`max_group_timestamp_skew_ms` 设为 **20.0**。8 ms 会把 Insert_Battery 也拒掉。

推荐环境变量：

```bash
source /mnt/netdata/Team/Personal/congsheng/miniconda/etc/profile.d/conda.sh
conda activate pi-dex
# pip install -e .   # 若尚未安装最新脚本入口

export PI_DEX_REPO=/mnt/netdata/Team/Personal/congsheng/PI-DEX
export PI_DEX_ARTIFACTS=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts
export OPENPI_DATA_HOME="${PI_DEX_ARTIFACTS}/openpi-data"
export OPENDATA_ROOT=/mnt/netdata/Team/Academic/Data/North/SharpaOpenData
export CONTRACT="${PI_DEX_REPO}/configs/site/joint_29d_observation.reviewed.json"
# 全量库专用 asset，勿与 ClearPlate 冒烟共用
export ASSET_ID=sharpa_joint_29d_opendata_v0
export ASSETS_DIR="${PI_DEX_ARTIFACTS}/assets-opendata"
```

---

## 4. 步骤 ① Observation contract

训练必须用 reviewed 文件：

```text
configs/site/joint_29d_observation.reviewed.json
```

它决定：

- state 列（当前 65D）与相机槽
- `physical_horizon` K=8、`raw_control_60_hz`
- prompt 来源与缺失策略
- **split**：`episode_hash_stratified_by_task`，约 train 0.8 / val 0.1 / test 0.1

按 **task_instruction 文本**分层，再在层内用 `hash(seed:episode_id)` 分桶；换 seed /
换 contract fractions 会改变划分，必须重算 stats 并新开 run。

---

## 5. 步骤 ② Inventory（推荐先做）

不打开每个 HDF5 做 horizon 探测，只统计 discover + split：

```bash
cd "${PI_DEX_REPO}"
pi-dex-dataset-inventory \
  --dataset-root "${OPENDATA_ROOT}" \
  --observation-contract "${CONTRACT}" \
  --output-json "${PI_DEX_ARTIFACTS}/dataset/opendata_inventory.json"
```

输出：

- `opendata_inventory.json`：任务列表、episode 总数、split 计数（摘要）
- `opendata_inventory.full.json`：含逐 episode 的 split_manifest（体积大）

通过标准（本机参考）：

- `task_count == 51`
- `episode_count` 约 3.4e4（随磁盘清理可能略变）
- `split_counts.train` ≈ 0.8 × episode_count

也可用 Python：

```python
from pi_dex.dataset_inventory import inventory_dataset
payload = inventory_dataset(
    dataset_root="/mnt/netdata/Team/Academic/Data/North/SharpaOpenData",
    observation_contract="configs/site/joint_29d_observation.reviewed.json",
)
print(payload["episode_count"], payload["task_count"], payload["split_counts"])
```

---

## 6. 步骤 ③ Validate-data

### 6.1 小规模冒烟（建议）

```bash
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode validate-data \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${OPENDATA_ROOT}" \
  --split train \
  --max-episodes 8 \
  --output-json "${PI_DEX_ARTIFACTS}/dataset/validate_smoke.json"
```

### 6.2 全量 train split 索引构建

`validate-data` / `train` 在过滤 split 后会 `build_sample_index`：对每个 episode
**打开一次 HDF5**，并按 cadence 探测能装下 K 步的 start。全量约 3 万+ episode，
墙钟可能很长，请用 `nohup` / `tmux` 并写日志。

`compute-norm-stats` **不再走这条慢路径**：按 episode 向量化抽 `state` / 双手
动作（不解码图像），并用 `--norm-workers` 多进程分片。见 §7。

```bash
mkdir -p "${PI_DEX_ARTIFACTS}/logs" "${PI_DEX_ARTIFACTS}/dataset"

nohup pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode validate-data \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${OPENDATA_ROOT}" \
  --split train \
  --output-json "${PI_DEX_ARTIFACTS}/dataset/validate_train_full.json" \
  > "${PI_DEX_ARTIFACTS}/logs/validate_opendata_train.log" 2>&1 &
```

`output-json` 里的 `manifest.episode_count` / `sample_count` 是 **通过 horizon 探测的**
可用样本规模；小于 inventory 的 episode 数是正常的（尾段不够 K、缺字段等会被丢掉）。

---

## 7. 步骤 ④ 计算 norm stats（全量准备的核心）

只在 **train split** 上统计；val/test 不得泄漏进 stats。实现原理、本机吞吐和与旧
Insert_Battery 文件的对照见 [norm-compute.md](norm-compute.md)。

### 7.1 火山引擎（推荐）

norm 是**单 Python 进程**（不要用 `volc_ddp_train.sh` / torchrun），但进程内会按
episode 开 CPU worker 并行读 HDF5。平台侧 **1 节点 × 1 进程**，优先高核 CPU；
GPU 闲置也没关系。

默认 `NORM_WORKERS` 为空时，Python 使用 `min(cpu_count, 64)`。脚本把
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` 设为 1，避免
每个 worker 再开 BLAS 线程把核打满。需要覆盖时：

```bash
export NORM_WORKERS=16          # 本机 VEFS 实测约 16 饱和；或省略自动选
export NORM_STRIDE=1            # 正式统计必须为 1
```

MLP **自定义启动命令**：

```bash
bash /mnt/netdata/Team/Personal/congsheng/PI-DEX/scripts/volc_compute_norm.sh
```

可选：把 `configs/volc/opendata_norm.env` 配进任务环境。多 worker 时只有 `MLP_ROLE_INDEX=0` 会跑，其余直接退出。

本地 dry-run：

```bash
set -a; source configs/volc/opendata_norm.env; set +a
VOLC_DRY_RUN=1 bash scripts/volc_compute_norm.sh
```

无 MLP 变量时：`VOLC_SKIP_MLP=1 bash scripts/volc_compute_norm.sh`。

产物：

```text
${ASSETS_DIR}/${ASSET_ID}/norm_stats.json
${PI_DEX_ARTIFACTS}/dataset/norm_opendata_full.json
```

已存在则拒绝覆盖，除非 `NORM_FORCE=1`。

### 7.2 本机 nohup（不推荐长时间占开发机）

```bash
source /mnt/netdata/Team/Personal/congsheng/miniconda/etc/profile.d/conda.sh
conda activate pi-dex
cd /mnt/netdata/Team/Personal/congsheng/PI-DEX

nohup bash scripts/prepare_opendata_full.sh \
  > /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/logs/prepare_opendata_full.nohup.log 2>&1 &
echo "PID=$!"
```

手动等价命令：

```bash
mkdir -p "${ASSETS_DIR}"

nohup pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode compute-norm-stats \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${OPENDATA_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --output-json "${PI_DEX_ARTIFACTS}/dataset/norm_opendata_full.json" \
  > "${PI_DEX_ARTIFACTS}/logs/norm_opendata_full.log" 2>&1 &
```

内容包含 OpenPI quantile 字段：`state`、`left_actions`、`right_actions`（各宽 29 的
action stats；padding 维不参与）。

### 7.3 调试：限制样本数

```bash
# 快路径只扫前 N 个有效 window（不建全库 index），非正式
pi-dex-train-pytorch ... --mode compute-norm-stats \
  --max-samples 4096 \
  --norm-workers 8 \
  --asset-id sharpa_joint_29d_opendata_debug4096 \
  ...
```

或 `--max-episodes 64` 只取 discover 顺序下的前 64 个 train episode（有偏，仅管道测试）。

### 7.4 为何必须换 asset_id

| 场景 | asset_id 建议 |
|------|----------------|
| ClearPlate 冒烟 | `sharpa_joint_29d`（历史） |
| OpenData 全量 | `sharpa_joint_29d_opendata_v0` |
| 以后换 split seed / contract | 递增版本，禁止原地覆盖混用 |

Joint 与 Cartesian 也禁止共用同一 `asset_id`。

---

## 8. 步骤 ⑤ 用全量数据训练（衔接）

```bash
export CONVERTED_BASE="${PI_DEX_ARTIFACTS}/converted/pi05_base-pytorch-bfloat16-K8"
export EXPECTED_BASE_SHA256=2f8539e2308611ea6fff84a5d7774f80d7c177c624769ff842008cf85dea9eeb
export CKPT="${PI_DEX_ARTIFACTS}/runs/opendata-joint29d-$(date +%Y%m%d)"

pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${OPENDATA_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --pytorch-weight-path "${CONVERTED_BASE}" \
  --expected-base-sha256 "${EXPECTED_BASE_SHA256}" \
  --checkpoint-dir "${CKPT}" \
  --dtype bfloat16 \
  --device cuda \
  --batch-size 1 \
  --seed 0 \
  --output-json "${CKPT}.json"
```

多机时同样使用 `${OPENDATA_ROOT}` + 同一 `${ASSETS_DIR}`，见 [training.md](training.md)
第 5 节。

---

## 9. Split 与泄漏注意

- 默认 `--split train`：只训练 / 只算 stats 用 train 桶。
- 评估时另开：`--split validation` 或 `test`（**不要**在这些 split 上重算并覆盖正式
  `asset_id`）。
- `dedupe_by_episode_id=true`：同一 `episode_id` 只保留一次。
- 分层键是 **自然语言 task_instruction**，不是磁盘上的任务文件夹名；同一文件夹下若
  prompt 不同会进不同 stratum。

---

## 10. 常见问题

| 现象 | 原因 / 处理 |
|------|-------------|
| 只有几百 episode | `dataset-root` 仍指向单个任务（如 ClearPlate） |
| inventory 报 empty task_instruction | 已改为跳过并计入 `rejected_episode_count`；这类 episode 不进任何 split |
| inventory 很多，validate 更少 | 部分 episode horizon / 字段不过关，被 sample index 丢弃 |

| discover 很慢 | 全库 `rglob` + 海量小文件；属正常，可先 inventory |
| norm 极慢 | 先建全量 index 再遍历 sample；用 nohup；调试用 `--max-samples` |
| 缺任务目录 | 本机 51/51 均可发现；若某目录 0 episode，查是否只有报告无 HDF5 |
| 与旧 ckpt 混用 | 全量 stats ≠ ClearPlate stats；必须新 assets + 新 checkpoint 目录 |

---

## 11. 相关入口

| 入口 | 用途 |
|------|------|
| `pi-dex-dataset-inventory` | 全库任务 / episode / split 盘点 |
| `pi-dex-prepare-dataset` / `scripts/prepare_task_dataset.sh` | 空 prompt overlay + inventory + norm |
| `pi-dex-train-pytorch … --mode validate-data` | 建 sample index + shape 烟测 |
| `pi-dex-train-pytorch … --mode compute-norm-stats` | 写 `norm_stats.json` |
| `scripts/train_ddp.sh` | 任意节点/卡数训练启动 |
| `configs/site/joint_29d_observation.reviewed.json` | 数据语义契约 |
| `data/schema.json` | HDF5 / sidecar 字段说明 |

准备完成后，用 [training.md](training.md) 做单机或火山多节点训练即可。

## 12. Prompt token 长度（`max_token_len`）

全量可用 episode 扫描见
`pi-dex-artifacts/dataset/token_length_scan.json`。pi0.5 格式含 65D discrete state，
仅 state 开销约 271 tokens；实测最大约 **425**。PI-DEX 默认
`create_pi05_model_config(..., max_token_len=448)`。继续使用 200 会截断全部样本的语言条件。
