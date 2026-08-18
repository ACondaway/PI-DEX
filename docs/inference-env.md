# 推理机环境配置（对齐开发机）

本文说明如何在 **GPU 推理机** / **从端 NUC** 上安装与当前开发机一致的 `pi-dex` 环境，并完成
`pi-dex-serve` + Zenoh 桥的受控联调。动作语义与 wire 契约见 [pytorch.md](pytorch.md)；
训练见 [training.md](training.md)。

> **安全：** 当前链路缺少 lease / e-stop / watchdog（`BimanualController`）。仅适合人在旁边盯着的短时试跑，不能无人值守。

---

## 1. 目标一致性

对齐内容：

| 项 | 值 / 来源 |
|----|-----------|
| Conda env 名 | `pi-dex` |
| Python | 3.11 |
| Torch | `2.7.1` + **CUDA 12.6**（`cu126`，与开发机一致） |
| 第三方包 | `configs/inference/pip-lock.txt`（开发机导出） |
| 本仓库包 | editable：`openpi-client`、`openpi`、`pi-dex` |
| OpenPI overlay | `openpi/.../transformers_replace` 拷入 site-packages |

脚本：

| 脚本 | 作用 |
|------|------|
| `scripts/export_inference_lock.sh` | **在开发机**从当前 env 导出 lock |
| `scripts/setup_inference_env.sh` | **在推理机**按 lock + 本仓库装齐环境 |

---

## 2. 开发机：导出 lock

依赖变更或确认环境可用后：

```bash
source /mnt/netdata/Team/Personal/congsheng/miniconda/etc/profile.d/conda.sh
conda activate pi-dex
cd /path/to/PI-DEX
bash scripts/export_inference_lock.sh
# → configs/inference/pip-lock.txt
```

把该文件与仓库一起同步到推理机（同一 commit）。

---

## 3. 推理机：一键安装

先同步仓库到与开发机相同的 commit，再：

```bash
cd /path/to/PI-DEX

# 无 Miniconda 时自动安装到 ~/miniconda3
bash scripts/setup_inference_env.sh --install-miniconda

# 已有 conda：
# PI_DEX_CONDA_ROOT=~/miniconda3 bash scripts/setup_inference_env.sh
# 或：
# bash scripts/setup_inference_env.sh --conda-root /path/to/miniconda3

source ~/miniconda3/etc/profile.d/conda.sh   # 换成实际 CONDA_ROOT
conda activate pi-dex
bash scripts/setup_inference_env.sh --verify-only
```

常用选项：

```bash
# 预览步骤
bash scripts/setup_inference_env.sh --dry-run --install-miniconda

# 打印 activate 片段
bash scripts/setup_inference_env.sh --print-activate --conda-root ~/miniconda3

# 强制装 eclipse-zenoh（GPU 机若也跑 robot-client）
bash scripts/setup_inference_env.sh --with-zenoh

# 跳过完整 lock（更快，但与开发机不完全一致）
bash scripts/setup_inference_env.sh --skip-lock

# CUDA 变体（默认 cu126）
bash scripts/setup_inference_env.sh --torch-cuda cu124
bash scripts/setup_inference_env.sh --torch-cuda cpu
```

环境变量：`PI_DEX_CONDA_ROOT`、`PI_DEX_CONDA_ENV`、`PIP_INDEX_URL`、`TORCH_CUDA`、
`INSTALL_MINICONDA`、`SKIP_LOCK`、`WITH_ZENOH`、`PI_DEX_ARTIFACTS`。

---

## 4. Profile：GPU serve vs 从端桥

| Profile | 命令 | 说明 |
|---------|------|------|
| `gpu-serve`（默认） | `bash scripts/setup_inference_env.sh` | 装完整 lock + CUDA torch，跑 `pi-dex-serve` |
| `robot-client` | `bash scripts/setup_inference_env.sh --profile robot-client` | 默认 CPU torch + `eclipse-zenoh`，跑 `pi-dex-robot-client` |
| `full` | `--profile full` | 与 gpu-serve 相同安装面，可再加 `--with-zenoh` |

从端机器人栈本身仍用 Sharpa 的 `start.sh` / `start-nuc.sh` / `start-remote-orin.sh`；
本环境只服务 PI-DEX 桥与 GPU 推理，**不替代**机器人启动脚本。

---

## 5. 部署拓扑与启动顺序

```text
从端 NUC (+ Orin)          GPU 推理机              Pendant
─────────────────          ───────────              ───────
bash start.sh              pi-dex-serve
pi-dex-robot-client  ←WS→  (checkpoint)
       │ Zenoh
       ▼
north_observation → inference/action
                                              F6 推理模式
                                              F2 moving
```

1. 从端：`bash start.sh`（或拆开的 nuc / orin 脚本）
2. GPU：环境就绪后启动 serve
3. 从端：启动 Zenoh 桥（需与机器人同 Zenoh 域；已装 `eclipse-zenoh`）
4. Pendant：`F6` → 推理，`F2` → moving（人盯急停）

### 5.0 从训练制品导出推理包（开发机）

训练 step 目录里有 `optimizer.pt`（约 13G）等 serve 用不到的文件。在开发机抽出最小集合：

```bash
bash scripts/export_inference_bundle.sh \
  --run-dir "${PI_DEX_ARTIFACTS}/runs/sharpa_joint_29d_insert_battery-20260817-083004"
# 或显式步目录：
# bash scripts/export_inference_bundle.sh --checkpoint-dir .../20000
```

默认写到 `${PI_DEX_ARTIFACTS}/exports/<run>-step<N>/`（同盘硬链 `model.safetensors`，几乎不占额外空间）。
把**整个** export 目录 rsync/scp 到推理机。布局与启动命令见该目录 `README.md`。

不要拷：`optimizer.pt`、HDF5、converted `pi05_base`。必须带：`ckpt/model.safetensors`、
`ckpt/pi_dex.json`、`ckpt/assets/<asset_id>/norm_stats.json`、
`openpi-data/big_vision/paligemma_tokenizer.model`、reviewed contract。

### 5.1 GPU：model server

```bash
conda activate pi-dex
export OPENPI_DATA_HOME="${PI_DEX_ARTIFACTS}/openpi-data"   # 按本机改

bash scripts/serve_joint29d.sh \
  --checkpoint-dir /path/to/run/10000 \
  --assets-dir /path/to/assets-Insert_Battery \
  --asset-id sharpa_joint_29d_insert_battery \
  --robot-id POC22005 \
  --host 0.0.0.0 \
  --port 8000

# 环回探针（本机或经 SSH 隧道）
pi-dex-serve-probe --host 127.0.0.1 --port 8000
```

跨机时注意防火墙与可选 `--api-key`；生产前仍需按 [server-validation.md](server-validation.md) 补 TLS/认证评审。

### 5.2 从端：Zenoh 桥

```bash
conda activate pi-dex

# 离线编解码自检（无需 Zenoh / GPU）
pi-dex-robot-client --mode codec-smoke

bash scripts/robot_client_joint29d.sh \
  --serve-host <GPU_IP> \
  --serve-port 8000 \
  --prompt "insert the battery"
```

默认 topic：`north_observation` → `inference/action`（与 `examples/sharpa_north_sdk.py` 一致）。
编解码：`pi_dex.north_codec` / `pi_dex.north_pb2`（schema：`examples/north.proto`）。

---

## 6. 校验清单

在推理机执行：

```bash
bash scripts/setup_inference_env.sh --verify-only
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pi-dex-robot-client --mode codec-smoke
# serve 起来后：
pi-dex-serve-probe --host 127.0.0.1 --port 8000
```

期望：`VERIFY OK`、Torch 带 CUDA（GPU 机）、codec-smoke 输出 `ok: true`。

---

## 7. 相关文件

| 路径 | 说明 |
|------|------|
| `configs/inference/pip-lock.txt` | 冻结第三方依赖 |
| `scripts/setup_inference_env.sh` | 推理机安装入口 |
| `scripts/export_inference_lock.sh` | 开发机导出 lock |
| `scripts/export_inference_bundle.sh` | 从训练 step 抽出推理包（权重 + tokenizer + contract） |
| `scripts/serve_joint29d.sh` | GPU serve 包装 |
| `scripts/robot_client_joint29d.sh` | 从端桥包装 |
| [pytorch.md](pytorch.md) §6 | 协议与 wire 细节 |
| [training.md](training.md) | 训练与 checkpoint |
| [server-validation.md](server-validation.md) | 验收留证 |
