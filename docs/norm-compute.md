# joint_29d compute-norm-stats 优化

只在 **train split** 上统计 `state`、`left_actions`、`right_actions` 的 OpenPI quantile
stats，写入 `assets/<asset_id>/norm_stats.json`。val/test 不得泄漏。本文说明旧路径为
什么慢、快路径改了什么、语义是否对齐、以及怎么跑。

相关操作步骤仍见 [dataset.md](dataset.md) §7；训练侧 CLI 见 [training.md](training.md)。

---

## 1. 旧路径为什么慢

`training_runner` 的 `compute-norm-stats` 以前和 `validate-data` / `train` 共用一条线：

1. `discover_episodes` + `filter_episodes_for_split`
2. `build_sample_index(..., provenance=...)`：对每个 aligned start 调
   `_load_action_chunk` → 从 HDF5 **拷贝整段** `joint_angle` / `time`，再跑 cadence
3. 构造 `SharpaJoint29dDataset`，对每个窗走 `__getitem__`（含 **三路 JPEG 解码**）
4. `compute_bimanual_normalization_stats` 一次更新一个 `[K,29]` / `[D]`

Stats 只用 `state` 和双手动作。图像和逐窗 HDF5 拷贝全部浪费。规模差一个数量级：

| 集合 | train episode | 有效窗（约） |
|------|----------------|--------------|
| Insert_Battery overlay | 152 | 88 306 |
| SharpaOpenData 全量 | 26 991 | ~1.5×10⁷（重叠窗） |

旧路径在 OpenData 上相当于「建完 1500 万窗 index，再解 1500 万次三路 JPEG」。不要用
torchrun / DDP 包这步：每个 rank 会写同一 `assets/`。

---

## 2. 快路径做什么

实现：`src/pi_dex/norm_compute.py`，由 `--mode compute-norm-stats` 默认走（无
`--norm-legacy`）。

| 步骤 | 行为 |
|------|------|
| 打开 HDF5 | 每个 episode **一次** |
| 读数组 | 只读动作四组和 contract 的 state 列；**不读图像** |
| 合法 start | 向量化：`mode/sub_state==1`、horizon、raw 59.4 Hz cadence、组间 skew、canonical alignment（与 `derive_bimanual_logical_action_chunk` 同一套阈值） |
| 收集 | `state [S,D]`、`left/right [S,K,29]` |
| 并行 | 按 episode `fork` + `imap`（保序）；worker 只回传该集数组 |
| 统计 | **父进程**独占一份 OpenPI `RunningStats`（不跨进程合并直方图） |

训练语义保持：

- 重叠长度 `K=8` 的窗；OpenPI `update` 把 `[K,29]` reshape 成 `-1, 29`，每个物理步算一条向量
- 同一 raw 行会被约 `K` 个窗重复计入（与旧路径一致）
- 正式统计 `--norm-stride` 必须为 `1`

默认 worker 数：CLI `--norm-workers`，否则环境变量 `NORM_WORKERS`，否则
`min(cpu_count, 64)`。`workers==1` 为进程内串行，避免无意义的 IPC。

调试：`--norm-legacy` 仍走 Dataset + JPEG（给对照，不要用于全量）。

---

## 3. 本机实测（2026-08-18）

机器：Xeon Platinum 8457C；标称 180 逻辑核，**在线 43**（`os.cpu_count()=43`）；内存 440 GiB。
数据在 VEFS（`/mnt/netdata`）。当时 load ~11。

Discover + 读 `anno.json` 做 split：全库约 **1–2 分钟**。

抽取吞吐（Closethebottlecap train 160 episode，约 18 万窗）：

| `NORM_WORKERS` | episode/s | 外推 26 991 条纯扫描 |
|----------------|-----------|----------------------|
| 1 | 17.1 | ~26 min |
| 8 | 23.4 | ~19 min |
| 16 | 24.6 | ~18 min |
| 32 | 23.8 | 略慢于 16 |

16 已经饱和。再加核主要增加 VEFS 打开 HDF5 和父进程直方图的争用。本机全量建议：

```bash
export NORM_WORKERS=16
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

全库冷读跨很多任务目录，比单任务微基准慢。本机 16 worker 全量作业（2026-08-18）在扫过约 1800 episode 后约 **7–10 episode/s**（含 discover），墙钟预期 **约 45–70 分钟**，不是隔夜。单任务热目录的 24 episode/s 不能直接当全库速度。

---

## 4. 与旧 Insert_Battery 结果对照

同一 `prepared/Insert_Battery`、同一 reviewed contract、同一 train split。旧文件：

`pi-dex-artifacts/assets-Insert_Battery/sharpa_joint_29d_insert_battery/norm_stats.json`

（与 `runs/sharpa_joint_29d_insert_battery-20260817-083004/20000/assets/...` 字节相同。）

| 项 | 旧路径 | 快路径（workers=1） |
|----|--------|---------------------|
| 窗数 | 88 306 | **88 306** |
| JSON 是否逐字节相同 | — | 否 |

窗集合对得上。JSON 不同来自 OpenPI `RunningStats` 的累加粒度，不是选错帧。

对照用的「精确矩」是：快路径抽出的 `[88306,65]` / 展平后的双手 `[706448,29]`，在
**float64** 上算 `mean` / 总体 `std`（`ddof=0`）/ `np.quantile`。它衡量 RunningStats
离这批数组真实矩有多远，**不是**再扫一遍旧 Dataset 得到的第二套官方文件。

相对这批精确矩：

- 快路径 mean 误差 ~`1e-6`；旧文件 ~`1e-4`（个别维）
- 低方差维上旧路径更差：例如 state dim 59 的 std，精确约 `0.00166`，旧文件写成 `0.0`
- q01/q99 是 5000-bin 直方图近似；按 episode 大批更新比「一窗一次」更接近排序分位数

**已用旧 stats 训完的 Insert_Battery ckpt 不要换文件。** 新的 OpenData asset 用快路径即可。

---

## 5. 怎么跑

### 5.1 本机（无 MLP 变量）

```bash
source /mnt/netdata/Team/Personal/congsheng/miniconda/etc/profile.d/conda.sh
conda activate pi-dex
cd /mnt/netdata/Team/Personal/congsheng/PI-DEX

export NORM_WORKERS=16
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VOLC_SKIP_MLP=1

mkdir -p /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/logs
nohup bash scripts/volc_compute_norm.sh \
  > /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/logs/norm_opendata_full.log 2>&1 &
```

或 `bash scripts/prepare_opendata_full.sh`（inventory 已在则会跳过 inventory）。

### 5.2 火山 MLP

平台 **1 节点 × 1 进程**，选高核 CPU，GPU 闲置没关系。不要 `volc_ddp_train.sh`。

```bash
export NORM_WORKERS=16   # VEFS 上不必拉满核
bash /mnt/netdata/Team/Personal/congsheng/PI-DEX/scripts/volc_compute_norm.sh
```

环境样例：`configs/volc/opendata_norm.env`。多 worker 时只有 `MLP_ROLE_INDEX=0` 跑。

### 5.3 产物

```text
${PI_DEX_ARTIFACTS}/assets-opendata/sharpa_joint_29d_opendata_v0/norm_stats.json
${PI_DEX_ARTIFACTS}/dataset/norm_opendata_full.json
```

已存在则拒绝覆盖，除非 `NORM_FORCE=1`。`asset_id` 不要和 Insert_Battery / ClearPlate 混用。

日志里会看到：

```text
compute-norm-stats: vectorized workers=16 stride=1 episodes=26991 ...
compute-norm-stats: episodes 50/26991 windows=... used=... skipped=...
```

`path` 字段为 `vectorized_multiprocess`。

### 5.4 CLI 要点

| 参数 / 环境变量 | 含义 |
|-----------------|------|
| `--norm-workers` / `NORM_WORKERS` | CPU 进程数；本机 VEFS 建议 16 |
| `--norm-stride` | 每隔 N 个合法 start 取一个；正式必须 1 |
| `--norm-legacy` | 旧 Dataset+JPEG |
| `--max-episodes` / `--max-samples` | 调试截断；快路径不再为 max-samples 先建全库 index |
| `NORM_MP_START` | 默认 `fork`；异常时可试 `spawn` |
| `OMP_NUM_THREADS=1` 等 | 避免每个 worker 再开 BLAS 线程 |

---

## 6. 代码入口

| 路径 | 角色 |
|------|------|
| `src/pi_dex/norm_compute.py` | 向量化抽取 + 多进程 + 父进程 RunningStats |
| `src/pi_dex/training_runner.py` | `compute-norm-stats` 默认快路径；`--norm-legacy` 旧路径 |
| `src/pi_dex/openpi_integration.py` `compute_bimanual_normalization_stats` | 旧路径 / 假 dataset 单测，保留 |
| `scripts/volc_compute_norm.sh` | 火山 / 本机入口 |
| `tests/test_norm_compute.py` | 与 `derive_bimanual_*` / `_load_state` 对齐；含真 episode |

不要把 norm 做成 DDP。直方图在 worker 里算再合并会改变 OpenPI 的分箱语义，当前明确不做。
