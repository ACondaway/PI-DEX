# 服务器、GPU 与硬件验证清单

本文档只定义必须在目标服务器或机器人现场执行的验证流程，覆盖锁定环境、vendored OpenPI、
`pi05_base` 转换、PyTorch 训练、GPU、分布式边界、WebSocket 服务和硬件接入。它不是已完成
验证的证据，也不能替代站点安全规程、机器人厂商说明或控制器验收。

> **当前状态：BLOCKED / NOT RUN。** 本文中的命令未在编写文档的本机执行；当前工作树还没有根 `uv.lock`，因此阶段 A 已按本文规则阻塞。未验证 CUDA、GPU 显存、真实 checkpoint、网络服务、控制器或机器人。任何交付说明都不得据此宣称多 GPU、WebSocket 生产服务或真实硬件能力已经可用。

## 1. 结论边界

- 单元测试通过只能证明被测试的软件边界，不能证明目标 GPU、驱动、checkpoint、网络或机器人安全。
- GPU smoke test 通过不能证明完整模型能在目标显存内加载或稳定推理。
- policy 能产生动作不能证明动作的单位、坐标系、关节顺序、限位或硬件执行正确。
- 多机 DDP 已实现（torchrun / 火山 MLP）；正式放行仍须按阶段 D 留证。FSDP / LoRA / AMP
  仍不在范围内。
- 仓库提供 `pi-dex-train-pytorch` launcher 与 first-party `pi_dex.training.training_runner` /
  Sharpa `joint_29d` dataset；完整站点 WebSocket launcher、Sharpa SDK controller 租约和
  硬件急停仍未闭合。服务器和硬件测试必须使用经评审、纳入版本控制的接入程序。
- 当前没有受控的 `pi05_base` JAX/Orbax→PyTorch converter wrapper/parity harness；原始上游
  converter 仅退出码为零或生成 `model.safetensors` 不能作为初始化权重通过证据。
- 当前 PyTorch 路径只支持 full bfloat16/full float32，不支持 LoRA、FSDP、EMA 或
  mixed precision/AMP；DDP 可用但不能用来规避单卡显存不足。资源不足不能通过静默切换
  未支持路径规避。
- 任一阶段为 `FAIL` 或 `BLOCKED` 时不得继续到依赖它的阶段。硬件阶段还必须满足现场负责人的独立放行条件。

阶段依赖固定为：`A → A.1 → A.2`；基础 GPU 阶段 B 可在 A 后并行执行；B.1 还依赖 A.1、
A.2 以及已审阅的 dataset/runner；B.2 依赖 B.1；C 依赖 B.2；E 依赖 C 和 D；F 依赖 E；G
依赖 F。A.3 是不加载数据/CUDA 的 launcher contract 验证，可在 A 后执行；D 是分布式 DDP
留证阶段，在 E 前必须完成。每阶段只要求其输入制品，不能提前要求后续阶段尚未产出的
checkpoint。

## 2. 前置条件

### 2.1 软件与制品

- 使用一个明确的 Git commit；工作树无未记录改动，或所有改动都有单独补丁和审阅记录。
- 根目录和 `openpi/` 的锁文件必须已提交。若根目录没有 `uv.lock`，服务器验证判为 `BLOCKED`；
  不得在验证 run 内临时生成锁文件后声称结果可复现。服务器 coding agent 可以在验证前的开发
  commit 中生成 lock，但必须审阅、提交，然后从该固定 clean commit 开启新的验证记录。
- 进入阶段 C 及 E–G 的训练后 checkpoint 必须来自受控位置，并至少包含：

  ```text
  model.safetensors
  pi_dex.json
  assets/<asset_id>/norm_stats.json
  ```

- 进入阶段 C 及 E–G 时，测试所用动作表示、`BimanualActionSpec`、OpenPI model config、normalization
  `asset_id` 和 checkpoint 必须来自同一实验记录；不得根据 action 宽度反推表示。
- 双模式的阶段 C 及后续验收必须分别准备一个 v3 checkpoint/normalization 资产。Cartesian 资产的 action
  stats 宽度为 31，Joint 为 29；两者不可共用、改名或跨模式恢复。
- 训练初始化的默认源必须是
  `gs://openpi-assets/checkpoints/pi05_base`。源 JAX checkpoint、converted PyTorch base 和
  训练后的 PI-DEX checkpoint 是三种不同制品；只有最后一种包含完整 Sharpa stats 和
  `pi_dex.json`，能够进入部署验收。
- OpenPI tokenizer/model 所需的远端资源必须已固定并缓存，或目标服务器具备经过批准的网络访问。应记录实际缓存对象的来源和哈希；仅记录配置字段不能证明远端 tokenizer 字节未变化。
- 服务器验证前必须登记 PaliGemma tokenizer 的源 URI、隔离可写 acquisition cache、校验后
  immutable snapshot 路径、发布方给出的预期 SHA-256 和现场实际 SHA-256；两者不一致时为
  `BLOCKED`。获准联网或记录一个下载后的新哈希本身不能证明无漂移。
- 所有外部 training runner/FK factory、站点 smoke、distributed、WebSocket 和硬件 harness
  必须来自版本控制项目，并记录 `module:callable`、commit、源码哈希及其依赖 lock 哈希。
  若只使用 OpenPI 已锁定环境而没有额外依赖，也必须记录脚本 SHA-256。
- 服务器不得在日志中输出访问令牌、私有数据样本或完整原始 observation。

### 2.2 服务器与 GPU

- 目标系统、CPU 架构、内核、NVIDIA driver、CUDA runtime 和 GPU 型号已列入部署清单。
- GPU 支持目标 checkpoint 所要求的精度；当前 PI-DEX PyTorch policy 部署只接受声明为 `bfloat16` 的模型配置。
- 磁盘空间足以容纳环境、checkpoint、只读制品副本和测试日志；GPU 显存预算包含模型、KV cache、输入、临时张量和安全余量。
- 上游对 full fine-tuning 的粗略估算为单卡显存 `>70 GB`（A100 80GB/H100 级别）；它不是
  PI-DEX 的实测通过阈值。转换还可能同时持有 Orbax、NumPy、Torch 和输出副本，必须分别预留
  cache、临时目录、主机 RAM/swap 和磁盘空间。
- 测试进程使用专用用户和明确的 GPU 可见范围，不与未知训练或推理任务共享显存。
- 服务器与机器人控制时钟已使用站点认可的方式同步，并能记录时钟源、偏差和采样时刻。

### 2.3 WebSocket 与网络

- 先只绑定 `127.0.0.1`；环回测试通过并完成网络审阅后，才允许绑定受控接口。
- 端口、防火墙、身份认证、TLS 或可信隔离网络策略由站点负责人书面确认。仓库示例本身不构成认证或加密方案。
- 客户端和服务端必须使用相同的 action metadata、wire format、execution horizon 和 clock domain。
- 已定义连接中断、超时、响应丢失、server restart、chunk ID 跳号以及客户端重连后的安全 hold/recovery 流程。

### 2.4 真实硬件

- 仿真或无动力 dry-run 已通过；首轮服务器验证不得直接连接可运动机器人。
- 现场具有可触达且经过单独验证的硬件急停、驱动使能隔离和控制权归属机制。
- 已划定隔离区，并安排有权停止测试的现场操作员和安全观察员。
- 两种模式都必须记录 robot/embodiment、左右手 arm/hand joint order、单位、映射和已标定的
  absolute commanded-position 语义。当前数据路径不支持 delta/residual；相关能力实现前不得用
  验收配置绕过该限制。Cartesian 还必须记录 FK 标定、wrist link、坐标系及 rotation-6D 约定；
  Joint 必须证明没有构造/调用 FK，并按 7D arm + 22D hand 的关节限幅验收。
- 控制器侧独立实施位置、速度、加速度、力矩、工作空间、碰撞和超时限制；具体阈值由机器人/站点负责人批准，不从模型输出推断。
- 已定义 observation 过期、时钟不一致、NaN/Inf、越界动作、网络断连、GPU 异常和进程崩溃时的 hold 或安全停机行为。

## 3. 固定测试身份与记录方式

先由操作员填写站点实际路径；示例值不得原样用于生产：

```bash
export PI_DEX_REPO=/srv/pi-dex
export PI_DEX_CHECKPOINT=/srv/checkpoints/REPLACE_WITH_CHECKPOINT
export PI_DEX_ASSET_ID=REPLACE_WITH_ASSET_ID
export PI_DEX_DEVICE=cuda:0
export UV_CACHE_DIR=/srv/pi-dex-cache/uv/REPLACE_WITH_VALIDATION_ID
export UV_LINK_MODE=copy
export OPENPI_DATA_HOME=/srv/pi-dex-cache/openpi
export PI_DEX_PI05_BASE_ACQUIRED="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base"
export PI_DEX_PI05_BASE_JAX=/srv/pi-dex-artifacts/pi05_base-source/REPLACE_WITH_MANIFEST_HASH
export PI_DEX_PI05_BASE_PT=/srv/pi-dex-artifacts/REPLACE_WITH_CONVERTED_BASE
export PI_DEX_CONVERSION_CONFIG_ID=REPLACE_WITH_REVIEWED_CONFIG_ID
export PI_DEX_EXPERIMENT_CONFIG=/srv/pi-dex-site/REPLACE_WITH_EXPERIMENT_CONFIG
export PI_DEX_RECORD_DIR=/srv/pi-dex-records/REPLACE_WITH_VALIDATION_ID
export PI_DEX_BIND_HOST=127.0.0.1
export PI_DEX_PORT=8000
export PI_DEX_TOKENIZER_ACQUIRED="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
export PI_DEX_TOKENIZER_MODEL=/srv/pi-dex-artifacts/tokenizer/REPLACE_WITH_HASH/paligemma_tokenizer.model
export PI_DEX_TOKENIZER_EXPECTED_SHA256=REPLACE_WITH_RELEASE_SHA256
# 以下值必须替换成已审阅项目里的真实 Python module:callable；仓库不提供默认 runner/FK。
export PI_DEX_TRAINING_RUNNER=REPLACE_WITH_REVIEWED_MODULE_CALLABLE
export PI_DEX_FK_PROVIDER_FACTORY=REPLACE_WITH_REVIEWED_MODULE_CALLABLE
cd "$PI_DEX_REPO"
```

`PI_DEX_EXPERIMENT_CONFIG` 必须显式包含 representation-scoped spec 和 model/runner 参数并记录
文件哈希；不得从 checkpoint 宽度反推。所有 runner/harness 要么是本仓库的 first-party 入口，
要么由带哈希 wheel/锁定独立环境以可复现方式安装；仅记录外部 lock、再从当前 env 直接运行
任意 `/srv` 脚本不算受控。`PI_DEX_RECORD_DIR` 由版本控制 command recorder 创建并用于保存每条
命令、时间、退出码和 stdout/stderr；若 recorder 尚未实现，执行记录保持 `BLOCKED`。

PaliGemma tokenizer 获取阶段的实际路径必须由
`maybe_download("gs://big_vision/paligemma_tokenizer.model")` 在同一 `OPENPI_DATA_HOME` 下解析并
与 `PI_DEX_TOKENIZER_ACQUIRED` 比对；验真后复制到 hash-pinned staging 并原子发布 immutable
`PI_DEX_TOKENIZER_MODEL`。Vendored downloader 会创建 lock/目录、修改权限并可能刷新过期条目，
因此不能把 `OPENPI_DATA_HOME` 指向典型只读挂载。训练/policy harness 必须通过 first-party
local-path resolver 证明 tokenizer 实际读取 `PI_DEX_TOKENIZER_MODEL`，且运行前后 snapshot
manifest 不变；不能再次走 GCS downloader，也不能只哈希一个 runtime 未使用的同名副本。

一次验证记录只能声明一个明确的 `PI_DEX_ACTION_REPRESENTATION`：`cartesian_31d` 或
`joint_29d`。完整双模式验收应复制整份记录，各自绑定 checkpoint、asset ID、spec 和日志，
不能在同一行结果中混写两个模式。

Joint 验证所用 spec 还必须把 `coordinate_frame`、`rotation_6d_convention`、
`kinematics_calibration_version` 和左右 `wrist_link` 设为 `None`，并确认 sidecar/wire 中对应
字段为 JSON `null`；出现任何 Cartesian 占位字符串都应 fail closed。

每个命令都要记录：完整命令、开始/结束时间、退出码、stdout/stderr 日志位置及日志 SHA-256。日志路径不得位于 checkpoint 目录内，checkpoint 在整个验证期间保持只读。失败后不得只重跑成功片段；必须保留首次失败记录，并说明修复和完整重跑范围。

记录源码与制品身份：

```bash
git rev-parse HEAD
git rev-parse HEAD:openpi
git status --short
uv --version
test -f uv.lock
test -f openpi/uv.lock
sha256sum uv.lock openpi/uv.lock
```

第二行记录当前 commit 中 vendored `openpi/` 目录的 Git tree object；根仓库没有把它登记为
submodule，因此空的 `git submodule status` 不能证明 OpenPI 源码身份。上述命令是阶段 A 的
最小身份记录。源/converted base 由 A.2 生成清单；完整训练 checkpoint 由 B.2 生成清单；最终
三件套和 tokenizer 在进入 C 前校验。不得在前一阶段尚未生成制品时用全局存在性检查制造循环
阻塞，也不得用重新下载或重建后的成功记录覆盖首次失败。

## 4. 分阶段验证

### 阶段 A：锁定环境与软件基线

根环境只能使用已提交的锁定解析：

```bash
cd "$PI_DEX_REPO"
uv sync --locked --extra pytorch
uv run --locked --extra pytorch ruff check src scripts tests
uv run --locked --extra pytorch ruff format --check src scripts tests
uv run --locked --extra pytorch pytest tests -m "not manual"
```

vendored OpenPI 环境同样从其锁文件同步，并以 editable root package 做联调：

```bash
cd "$PI_DEX_REPO/openpi"
uv sync --locked
uv run --locked --with-editable .. python -c "import openpi; import pi_dex"
uv run --locked --with-editable .. pytest \
  ../tests/test_openpi_integration.py \
  ../tests/test_openpi_normalization_roundtrip.py \
  -m "not manual"
```

上述第一次 sync 开始就必须沿用第 3 节的专用 `UV_CACHE_DIR`、`UV_LINK_MODE=copy` 和同一个
可丢弃环境；所有后续 OpenPI `uv run` 也使用它。任何 `uv sync`、Transformers 重装或环境重建
都使 A.1 补丁证据失效，必须重新应用补丁并逐文件验真后才能继续。

这里的 `--locked` 只锁定 `openpi/` 基础项目；`--with-editable ..` 是由当前 Git commit
固定源码、但未写入 `openpi/uv.lock` 的本地 overlay。报告必须保留这一边界，不得把组合环境
描述成“由单一 lock 完整锁定”。若站点要求单一锁定解析，必须先通过经评审的 workspace 或
带哈希 wheel 发布流程把根包纳入共同依赖图，再更新本文命令；不得在验证现场临时改造。

通过标准：

- 两次环境同步未改写锁文件；执行后 `git status --short` 与执行前一致。
- Ruff、格式检查、根测试及 OpenPI 联调测试退出码均为零。根环境允许仅因未安装 OpenPI 而跳过 `test_openpi_normalization_roundtrip.py`；随后 OpenPI 环境中的两个定向联调文件不得 skip。
- 测试日志记录通过、失败、跳过和 deselected 的准确数量；不得将依赖缺失导致的 skip 记为通过。

### 阶段 A.1：PyTorch Transformers 补丁完整性

`uv sync --locked` 只证明依赖解析完成，不足以证明 PyTorch π0.5 runtime 正确。当前 vendored
OpenPI 要求 `transformers==4.53.2`，并要求
`openpi/src/openpi/models_pytorch/transformers_replace/` 中的替换文件已安装到实际加载的
Transformers package。

补丁必须在专用、可丢弃的环境中通过版本控制 provisioning 入口应用，使用独立 uv cache 和
copy link mode。不得直接覆盖共享环境，或覆盖可能以 hardlink 指向共享 uv cache 的文件；补丁
前已经导入 Transformers/OpenPI 的 Python 进程必须退出。服务器至少记录：

1. Python、uv、PyTorch 和 Transformers 的实际版本与加载路径；Transformers 必须精确为
   `4.53.2`。
2. 补丁源目录全部文件的相对路径和 SHA-256，以及安装后对应目标文件的 SHA-256；每一对必须
   逐字节一致。
3. 专用环境、uv cache 和 link mode，证明其他项目/cache 未被污染。
4. 在全新 Python 进程中构造 `PI0Pytorch`，使其内置 patch 检查真实执行；仅 import module
   不足以触发该检查。模型 config 的 `openpi_model_contract_metadata` 字段必须与 A.2 精确相同，
   包括 dtype；只允许把不进入语义契约的 `pytorch_compile_mode` 显式覆盖为 `None`。CPU 结构
   smoke 不得被描述成 GPU 或目标精度算子验证。

版本不匹配、目标文件缺失/哈希不同、环境 provenance 不明、patch 落入共享 hardlink cache，或
模型构造的自检失败时，本阶段为 `BLOCKED`。

### 阶段 A.2：`pi05_base` 获取、转换 provenance 与跨框架 parity

默认初始权重只接受官方 `gs://openpi-assets/checkpoints/pi05_base`。`pi05_base` 是 checkpoint
制品名，不是 OpenPI `config_name`；不得以 `pi05_droid`、随机初始化或同名的其他框架制品
替代。converter 的 `--checkpoint_dir` 接收 checkpoint 根目录并自行读取 `<root>/params/`，
不得传 `params/` 本身。

下载/转换前必须冻结并记录：官方 URI、受控本地只读副本、发布方批准的预期 manifest/hash、
现场按相对路径排序生成的 size/SHA-256 manifest、PI-DEX 根 commit、`HEAD:openpi` tree OID、
vendored converter 文件 SHA-256、外部 wrapper 的项目 commit/lock/source hash、
完整命令和 precision。仅为下载结果新算一个 hash 不能独立证明来源未漂移；若没有预期
manifest，必须由实验发布记录明确批准该制品，否则为 `BLOCKED`。

官方源先进入隔离可写 acquisition cache；只有 manifest 验真后才能原子发布为上句的受控
immutable snapshot。Vendored `maybe_download()` 不得在转换/训练/推理阶段直接操作该 snapshot，
也不得在其中创建 lock 或修改权限。

当前没有已注册的 PI-DEX `TrainConfig`，所以经审阅的 wrapper 必须从
`PI_DEX_EXPERIMENT_CONFIG` 解析 spec，调用 `create_pi05_model_config(spec)` 构造模型配置，再直接
调用/修补转换函数；不得把 representation 值当成 `--config_name`，也不得执行原始 CLI 占位
命令。`PI_DEX_CONVERSION_CONFIG_ID` 标识序列化配置记录，不是 registry 名。转换语义指纹固定为
`openpi_model_contract_metadata`：`pi05=True`、`action_dim=32`、`action_horizon=2*K`、
`discrete_state_input=True`、dtype、Gemma variants 和 `max_token_len`；compile mode 另行记录但不
决定 base 共享。当前 mapper 硬编码 `gemma_2b` PaliGemma 结构和 `gemma_300m` expert，因此 wrapper
必须强制这两个默认 variant，或先实现并验证通用 mapper。两种 representation 只有语义指纹
相同时才可共用 converted base；stats、checkpoint 和 sidecar 仍必须分开。

当前原始 converter 根据路径是否包含 `pi05` 选择 AdaRMS、忽略 `strict=False` 的
missing/unexpected keys，并从 `checkpoint_dir.parent/assets` 尝试复制资产。经审阅的
wrapper/harness 必须按 config 选择 π0.5 分支、要求未批准的 missing/unexpected keys 为空、
拒绝随机保留参数并独立处理 assets。简化的 converter `config.json` 不能替代完整 config 记录
或 `pi_dex.json`。未经包装的 converter 仅退出码为零或生成 safetensors 不得记为 `PASS`。

转换必须写入同一文件系统中的全新 staging 目录；完成源覆盖率、strict load、parity、完整
逐文件 size/SHA-256 manifest 及 provenance record 后，才原子 rename 到不存在的
`PI_DEX_PI05_BASE_PT`。失败时清理 staging，目标已存在时 fail closed；不得覆盖发布制品或把
旧 `config.json`/assets 混入新权重。

转换后至少验证：

1. 输出 `model.safetensors` 由同一 PI-DEX config 严格加载；全部 key、shape、dtype、有限性和
   文件 SHA-256 符合预期。该检查不能替代 converter 内部的源参数覆盖率检查。
2. 用固定且有哈希的 post-transform images/masks/tokens/state、actions、noise 和 flow timestep，
   在 eval/no-augmentation 条件下比较 JAX 源模型和 PyTorch converted 模型的 action vector
   field。`atol/rtol`、最大/平均误差和非有限值标准在执行前登记；必要时顺序运行以控制显存。
3. 源/输出目录在验证期间只读且前后 manifests 相同；converter 复制的任何上游 assets 都不得
   冒充 Sharpa 31D/29D stats。
4. 转换前记录 RAM/swap/磁盘，转换时记录 peak RSS 和临时/最终占用；空间不足为 `BLOCKED`，
   OOM、swap thrashing 或磁盘耗尽为 `FAIL`。

本阶段产物只供 runner 的 `pytorch_weight_path`/严格初始化加载使用，不是可 resume/serve 的
PI-DEX checkpoint。完成标准是 provenance 完整、参数覆盖无未批准遗漏、严格加载与 parity 均
满足预登记标准。

### 阶段 A.3：双模式 launcher 与训练接缝

本阶段只验证 launcher 的参数、动态导入时点、spec/FK completion handshake 和退出码边界，
不得访问训练数据、下载/加载 base、初始化 CUDA 或写 checkpoint。真正的 runner 资源与训练
验收属于 B.1/B.2。使用仓库已有的定向测试：

```bash
cd "$PI_DEX_REPO/openpi"
uv run --locked --with-editable .. pytest ../tests/test_training_launcher.py -m "not manual"
```

通过标准：Joint 禁止 FK、Cartesian 缺 FK、语法错误均在 dynamic import 前拒绝；成功返回必须
绑定 representation 匹配的 spec，Cartesian 还必须通过 context 获取且至多创建一次 provider；
普通返回与 `SystemExit` 的成功路径完成同一握手，非法退出码不会经 shell 8-bit 折返伪装成功。
这些测试退出码为零只能记为“launcher contract 通过”，不能记为 dataset、FK、训练或 checkpoint
闭环通过。

### 阶段 B：GPU 与 bfloat16 smoke test

先记录驱动和 GPU 拓扑：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv
nvidia-smi topo -m
```

再在根锁定环境中验证指定设备和基本 bfloat16 运算：

```bash
cd "$PI_DEX_REPO"
uv run --locked --extra pytorch python - <<'PY'
import os
import torch

device = torch.device(os.environ["PI_DEX_DEVICE"])
assert torch.cuda.is_available(), "CUDA is not available"
assert device.type == "cuda", f"expected a CUDA device, got {device}"
torch.cuda.set_device(device)
assert torch.cuda.is_bf16_supported(), "selected runtime does not report bfloat16 support"
properties = torch.cuda.get_device_properties(device)
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "device": str(device),
    "name": properties.name,
    "total_memory": properties.total_memory,
    "bf16_supported": torch.cuda.is_bf16_supported(),
})
x = torch.ones((1024, 1024), dtype=torch.bfloat16, device=device)
y = x @ x
torch.cuda.synchronize(device)
assert y.dtype == torch.bfloat16
assert torch.isfinite(y).all().item()
print({"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device)})
PY
```

通过标准：指定 GPU 可见、bfloat16 smoke test 退出码为零、无 CUDA error/ECC/Xid/OOM，并记录执行前后的显存占用。此阶段只证明基本 runtime，不能替代完整模型测试。

### 阶段 B.1：完整微调资源门控

上游给出的 PyTorch full fine-tuning 粗略估算为 `>70 GB` GPU 显存；它不是 PI-DEX 的实测
阈值。当前 PyTorch 路径不能用 mixed precision、FSDP、LoRA 或 EMA 降低资源需求。已支持的
DDP 会复制模型，不能代替 FSDP 降低单卡显存；多卡主要用于增大 global batch / 缩短墙钟时间。

资源 probe 必须使用与正式实验相同的 model config、converted weights、optimizer、compile
mode、最大 image/state/prompt shape、`2*K` horizon 和计划 batch size，至少完成一次完整
forward→backward→optimizer step，使梯度和 optimizer state 峰值真实出现。当前 vendored
`torch.compile` 只包装 `sample_actions`，训练 forward 不会触发该编译；若正式部署启用 compile，
必须另以与 C 相同的 policy inference 路径测量首次/稳态编译时间和 inference 峰值，不能从训练
step 推断。
记录 GPU allocated/reserved/总显存峰值、主机 peak RSS/swap、data worker 内存、临时磁盘、
吞吐和首步/稳态耗时。显存、主机内存、磁盘安全余量和最大 compile 时间必须事先登记；OOM、
ECC/Xid、swap thrashing、磁盘耗尽或超出阈值均为 `FAIL`。改变 batch、horizon、precision 或
compile 设置后必须建立新的验证记录。

full fine-tuning 还必须生成按参数名排序的 all-parameter 与 `requires_grad` name/shape/numel
manifests 及 SHA-256，二者必须完全一致；optimizer 必须恰好覆盖所有 trainable parameters。
出现冻结参数时，本实验不得记录为 full fine-tuning，除非另有已审阅的能力名称和标准。
Runner 还必须在模型构造前 fail closed 要求
`TrainConfig.pytorch_training_precision == train_config.model.dtype`，并用实际参数 dtype policy/
manifest 验真；部署训练两者均为 `bfloat16`，float32 诊断两者均为 `float32`，不得启动后静默
改写任一字段。

面向阶段 C 的训练必须使用可部署的 full-bfloat16 配置。Full-float32 可以是单独诊断实验，
但其 checkpoint 当前不能进入 PI-DEX 部署验收；bfloat16 loss 可能高于 float32，因此也不能
套用未经验证的跨 precision 阈值。

### 阶段 B.2：双模式训练闭环与质量验收

Joint 29D 和 Cartesian 31D 必须分别执行。若仍缺 first-party HDF5 runner 或完整 checkpoint
manager，本阶段保持 `BLOCKED`；launcher 参数检查、单次 backward 或 loss 下降不能替代闭环。

每个模式启动前必须预登记 dataset manifest、episode 级 train/validation/test split、去重/
泄漏报告、仅从训练集计算的 stats、seed、initial converted-base hash、optimizer、scheduler、
batch、step 数、保存周期和质量阈值。31D/29D stats 与指标不可共用；不得只比较两种模式的 raw
aggregate loss 来判定优劣。首个闭环固定 `num_workers=0`；多 worker 只有在 HDF5 per-worker
open/close、seed、清理和确定性验收完成后才允许。

至少完成：

1. step 0 证明模型实际来自 A.2 的 base，而非随机初始化；除登记的 padding neutralization
   外，加载前后参数 provenance 符合预期。
2. 单 batch 和小样本 overfit smoke 中 loss、gradient norm、参数更新均有限且非零；padding
   不进入 stats/noise/loss，左右手 sentinel 不交换。Overfit 只证明链路，不证明泛化。
3. 正式训练持续记录 train/held-out loss、gradient norm、学习率、吞吐、显存、坏样本拒绝/
   跳过计数和 checkpoint hash；不得静默丢弃坏 batch。
4. 在固定 held-out episodes 和预登记 seeds 上反归一化动作。Cartesian 分手报告 wrist
   translation、rotation geodesic 和 hand-joint 指标；Joint 分手报告 arm/hand-joint 指标；
   两者都报告同步 chunk、时序连续性、hand swap 和单侧退化。
5. 专项验证 `[L0,R0,...]`：A.2 parity 只证明转换，不证明把 `2K` positions 解释成 `K` 个
   同步双手步与预训练时序语义等价。必须报告收敛、左右对称/耦合和 deinterleave 轨迹质量。
6. 比较连续 `N+M` steps 与 `N` steps 保存→全新进程 resume→`M` steps；恢复 model、
   optimizer、scheduler、global step、Python/NumPy/Torch/CUDA RNG 和 data cursor，并按预登记
   标准检查下一 batch、学习率、loss 和最终状态。
7. 原子发布最终 checkpoint，保存父 base、数据/config/tokenizer lineage，并通过 cross-mode
   rejection、resume 和内部 strict-load smoke，使其具备进入阶段 C 独立验收的资格。B.2 的
   `PASS` 不依赖阶段 C 的结果。

训练 checkpoint 必须对 model、optimizer、scheduler、step、各 RNG、data cursor、config、
dataset/split、父 base 和 tokenizer 等完整 resume 制品生成逐文件 size/SHA-256 manifest；不能
只哈希部署三件套。Cartesian 还必须绑定 URDF/MJCF、标定文件、FK provider commit/source hash、
golden-pair fixture、预登记容差和结果哈希。发布目标必须 representation/run-scoped、原子写入且
拒绝跨模式目录或 `asset_id` 碰撞。

完成标准是训练闭环与预登记离线质量指标均通过。该结论不等于真实任务成功、网络可靠或硬件
安全；真机指标只能在 F/G 阶段报告。

### 阶段 C：真实 OpenPI checkpoint 与 policy

此阶段必须使用经评审的站点 smoke harness；仓库目前没有可直接冒充生产 launcher 的命令。harness 应调用 PI-DEX 的 checkpoint validator 和 `create_bimanual_trained_policy`，并显式传入 checkpoint、asset ID、spec、model config 和 `pytorch_device`。

进入本阶段前才执行最终制品门控：检查 `model.safetensors`、`pi_dex.json`、
`assets/<asset_id>/norm_stats.json` 和 tokenizer 文件均存在；比较 B.2 完整 manifest 与实验发布
记录，并验证 tokenizer 实际解析路径及预期/实际 SHA-256。A/A.2 不要求这些尚未产出的文件。

至少验证：

1. 在加载模型前，`pi_dex.json`、模型配置、weights 字节、normalization asset 字节和解析后统计指纹全部匹配。
2. checkpoint 的 `state_dim` 与 inference observation 的一维 state 宽度一致；短 state、长 state 和额外时间轴均 fail closed。
3. policy 实际位于 `PI_DEX_DEVICE`，模型配置为 pi05、action width 32、action horizon `2*K` 且 dtype contract 为 `bfloat16`。
4. serving-time dense action input/output projection 已执行：Cartesian 前 31 个语义维参数
   不变、1 个 padding 输入列/输出行/bias 为零；Joint 前 29 个语义维参数不变、3 个
   padding 输入列/输出行/bias 全为零。
5. 对每种表示使用固定 observation 和预登记的 PyTorch 随机种子做至少两次公开 policy
   推理；输出经解交错、去 padding 和反归一化后分别为 Cartesian
   `left/right[E,31]` 或 Joint `left/right[E,29]` float32，值有限，metadata 与 spec 一致。
   GPU 非确定性未被另行消除时不得要求两次数值逐位相等。
6. 分别记录冷启动时间、首个请求延迟、稳态延迟分位数、峰值显存、每秒请求数和测试时的 `num_steps`。性能阈值必须来自部署 SLO，而不是在结果出来后补写。
7. checkpoint 路径在验证期间保持只读；加载前后哈希一致。

站点命令模板如下。该入口必须是已提交并审阅的文件；若不存在，此阶段为 `BLOCKED`：

```bash
export PI_DEX_POLICY_SMOKE=/srv/pi-dex-site/policy_smoke.py
test -f "$PI_DEX_POLICY_SMOKE"
cd "$PI_DEX_REPO/openpi"
uv run --locked --with-editable .. python "$PI_DEX_POLICY_SMOKE" \
  --checkpoint "$PI_DEX_CHECKPOINT" \
  --asset-id "$PI_DEX_ASSET_ID" \
  --experiment-config "$PI_DEX_EXPERIMENT_CONFIG" \
  --device "$PI_DEX_DEVICE"
```

两种模式都必须额外拒绝另一模式的 spec、normalization 资产、checkpoint metadata 和 v2
sidecar；不得通过 shape 猜测、截断或补 stats 来兼容。通过标准是全部契约和负向用例按
预期通过或拒绝，推理输出、延迟和显存满足预登记阈值，且无权重、normalization 或
tokenizer 来源漂移。只完成模型加载但没有真实推理不得记为通过。

### 阶段 D：分布式 DDP

验收目标是证明多进程 DDP 路径可用且边界清晰。最小探针：

```bash
# 单机双进程（CPU gloo 也可用于 loader/cursor 探针）
torchrun --standalone --nproc-per-node=2 \
  -m pi_dex.training.training_runner  # 或站点 distributed_probe / 短 max-steps train
```

火山引擎 MLP 入口（平台注入 ``MLP_*``）：

```bash
bash scripts/volc_ddp_train.sh -- \
  --action-representation joint_29d \
  --runner pi_dex.training.training_runner:run -- \
  --mode train ...
# 或: pi-dex-volc-train -- --action-representation joint_29d ...
```

通过标准：

- ``DistributedSampler(shuffle=False, drop_last=True)`` + local ``batch_size``；``sampler_state`` 记录 ``world_size`` / ``global_batch_size``。
- 仅 rank-0 写入原子 checkpoint；其它 rank barrier 后退出且无部分文件。
- resume 时 ``world_size`` / ``batch_size`` / ``order_sha256`` 必须一致。
- ``distributed=False`` 与已初始化 process group 冲突时报错；DDP 下 ``shuffle=True`` 被拒绝。
- 报告可写 “DDP supported under documented constraints”，不得宣称 FSDP / AMP / LoRA。

### 阶段 E：WebSocket 环回与受控网络

服务端和探针入口由站点接入层提供并纳入版本控制。先启动环回服务：

> **放行门槛：** vendored stock server 只能用于环回诊断。阶段 E–G 在站点入口实现并验证服务端认证或等效可信隔离、TLS（若跨越非可信边界）、有限消息大小、安全错误封装、连接/接收/推理 deadline，以及独立 controller watchdog 之前均为 `BLOCKED`。watchdog 必须在线程永久阻塞和 TCP 半开时仍能在预登记时限内进入 hold。

```bash
export PI_DEX_SERVER_ENTRYPOINT=/srv/pi-dex-site/server.py
export PI_DEX_WEBSOCKET_PROBE=/srv/pi-dex-site/websocket_probe.py
test -f "$PI_DEX_SERVER_ENTRYPOINT"
test -f "$PI_DEX_WEBSOCKET_PROBE"
cd "$PI_DEX_REPO/openpi"
uv run --locked --with-editable .. python "$PI_DEX_SERVER_ENTRYPOINT" \
  --checkpoint "$PI_DEX_CHECKPOINT" \
  --asset-id "$PI_DEX_ASSET_ID" \
  --experiment-config "$PI_DEX_EXPERIMENT_CONFIG" \
  --device "$PI_DEX_DEVICE" \
  --host "$PI_DEX_BIND_HOST" \
  --port "$PI_DEX_PORT"
```

在独立终端确认只监听预期地址，然后运行探针：

```bash
ss -ltnp
cd "$PI_DEX_REPO/openpi"
uv run --locked --with-editable .. python "$PI_DEX_WEBSOCKET_PROBE" \
  --host "$PI_DEX_BIND_HOST" \
  --port "$PI_DEX_PORT"
```

探针必须覆盖：

- metadata 握手完整且与 checkpoint/spec/action representation/execution horizon 一致；
  `layout_version=3`、`metadata_schema_version=3`、wire v3 和 `logical_action_dim` 必须精确
  匹配；`session_id` 是 32 位小写十六进制，每个响应精确回显，server restart 后生成新值，
  旧值、mismatch、未知字段或 v2 wire/metadata 均被拒绝。
- 合法 observation 分别返回有限的 Cartesian `left/right[E,31]` 或 Joint
  `left/right[E,29]` float32、正确 clock domain、源时间戳和单调 chunk sequence ID。
- state 宽度错误、dtype/shape 错误、NaN/Inf、过期 observation、错误 clock domain 和非法时间戳均被拒绝，且没有产生可下发动作。
- 并发请求不会交叉污染 observation 或 chunk；响应超时和 backpressure 行为符合预先定义的 SLO。
- 主动断网、半开连接、永不返回的 inference、进程退出、GPU 异常、响应丢失和 server restart 后，独立 watchdog 在预登记 deadline 内使控制器进入 hold；旧 broker/session 不会静默续跑，必须重新获取并验证 metadata 后显式恢复。
- 超大消息在反序列化或分配无界内存前被有限的 message-size policy 拒绝；服务端错误响应不包含 traceback、绝对路径、凭据或内部配置。
- 日志不包含凭据或完整原始 observation；服务关闭后端口和 GPU 资源被释放。

环回全部通过后，才可在受控接口重复同一套探针，并记录网络路径、TLS/认证状态、延迟、丢包和防火墙规则。通过标准是所有正向、负向及断线恢复用例满足预登记标准；仅能建立 WebSocket 连接不算通过。

### 阶段 F：无动力硬件 dry-run

保持机器人驱动禁用或使用厂商认可的 simulation/dry-run 模式。服务输出只进入一个只记录、不执行的 controller adapter。

逐项确认：

- metadata 中 robot ID、embodiment、左右手 joint order、单位和 action mode 与现场配置逐字匹配；
  Cartesian 的标定、wrist link 和坐标系也必须逐字匹配，Joint 的对应字段必须为
  `null`，position/rotation 单位必须为 `not_applicable`。
- 分别使用与 representation 匹配的 controller dry-run adapter 和逐维限幅：Cartesian 接收
  31D wrist pose + hand，验证 rotation-6D 和 FK/坐标系；Joint 接收 29D arm + hand，验证
  7 个 arm joint 与 22 个 hand joint 的顺序/单位，并确认没有执行 FK/rotation 校验。
- execution horizon、控制频率、timestamp/clock domain 和允许延迟与控制器配置一致。
- 每个动作在 controller 限位检查之前和之后均有审计记录，但日志不泄露不必要的原始传感器数据。
- 左右手使用可区分的测试模式验证映射，不允许依赖对称零动作掩盖交换错误。
- stale、NaN/Inf、越界、序列跳号、断网和 server restart 均导致 hold/reject，且不能通过普通重连自行解除。
- 硬件急停和软件 stop/hold 是独立路径；软件成功不能替代硬件急停检查。

通过标准：没有 actuator 被使能，没有动作到达真实执行接口；所有映射、限位和故障注入结果由控制软件负责人和机器人负责人共同签字。

### 阶段 G：受控有动力硬件

只有 A–F 全部通过、现场风险评估获批并完成独立急停验证后才能执行。仓库不提供通用运动命令；具体步骤必须来自站点 runbook。

推荐按以下门控逐级放行，每一级都需要重新确认急停、隔离区和控制权：

1. 驱动使能但保持安全姿态，不发送模型动作。
2. controller 接收单个经过限幅的动作块，但只执行站点批准的最小运动范围。
3. 使用可区分的左右手低风险动作确认方向、单位、关节映射和时序。
4. 在书面批准的速度、力矩、工作空间和持续时间内进行短时闭环测试。
5. 通过短时测试后才执行有限 soak test，并持续监测延迟、序列、GPU、网络、controller fault 和温度。

每一级的通过标准必须在执行前填写，包括最大位置/速度/力矩、允许 tracking error、请求超时、最大连续 chunk 数和停止条件。任一异常、人工疑虑或指标超限都立即进入站点定义的安全状态，并将本级记为 `FAIL`；不得通过放宽阈值继续。

## 5. 结果记录模板

### 5.1 执行身份

```text
验证编号：
日期、时区：
操作员：
独立复核人：
现场安全负责人（硬件阶段）：
Git commit：
canonical origin / remote：
vendored openpi tree OID（HEAD:openpi）：
工作树状态：
根 uv.lock SHA-256：
openpi/uv.lock SHA-256：
站点 harness 项目 commit / lock SHA-256 / 脚本 SHA-256：
专用 OpenPI env / uv cache / link mode：
Transformers version / import path：
transformers_replace source manifest / installed-target manifest SHA-256：
主机名、OS、kernel、CPU：
uv / Python / PyTorch 版本：
GPU index、UUID、型号、driver、CUDA、显存：
pi05_base source URI / local root / expected manifest / actual manifest：
vendored converter SHA-256 / wrapper project commit-lock-source SHA-256 / exact command / precision：
conversion config ID / openpi_model_contract_metadata / full config SHA-256：
conversion missing keys / unexpected keys：
converted artifact full manifest / provenance-record SHA-256：
JAX↔PyTorch fixture SHA-256 / atol / rtol / max-mean error / result：
conversion peak RSS / swap / temporary-final disk usage：
training checkpoint full manifest / deployment-subset SHA-256：
tokenizer URI / resolved runtime path / expected SHA-256 / actual SHA-256：
asset_id / state_dim：
action representation / logical action dim / padding dims：
training runner module:callable / commit / hash：
all-parameter / trainable-parameter name-shape-numel manifest SHA-256：
FK provider project commit / source SHA-256（Joint 填 N/A）：
URDF-MJCF / calibration / golden fixture / tolerance / result SHA-256（Joint 填 N/A）：
model config 摘要：
model.dtype / pytorch_training_precision / actual parameter dtype policy：
experiment config / BimanualActionSpec SHA-256：
dataset manifest / split / leakage report / normalization hash：
initial weight hash / optimizer / scheduler / seed / steps：
resource probe config / peak GPU allocated-reserved / host RSS / headroom：
train-validation / per-hand physical metrics：
resume continuity result / final checkpoint lineage：
robot / embodiment / controller / firmware：
标定与 joint mapping 版本：
clock source 与实测偏差：
WebSocket bind、端口、TLS/认证/隔离策略：
预先批准的性能与硬件安全阈值：
```

### 5.2 命令与阶段结果

下表是复制后填写的空白模板，`NOT RUN` 只是未执行占位，不代表本文当前事实。当前仓库因缺少
根 `uv.lock`，阶段 A 是 `BLOCKED`；按第 1 节的依赖门控，后续阶段在其前置阶段解除前也不能
执行或标记通过。

| 阶段 | 测试 ID / 完整命令 | 开始/结束时间 | 退出码 | 状态 | 日志路径与 SHA-256 | 备注 |
|---|---|---|---:|---|---|---|
| A 环境 |  |  |  | NOT RUN |  |  |
| A.1 Transformers patch |  |  |  | NOT RUN |  |  |
| A.2 pi05_base conversion/parity |  |  |  | NOT RUN |  |  |
| A.3 launcher contract |  |  |  | NOT RUN |  |  |
| B GPU |  |  |  | NOT RUN |  |  |
| B.1 full-finetune resources |  |  |  | NOT RUN |  |  |
| B.2 Joint 29D training/quality/resume |  |  |  | NOT RUN |  |  |
| B.2 Cartesian 31D training/quality/resume |  |  |  | NOT RUN |  |  |
| C Joint 29D OpenPI policy |  |  |  | NOT RUN |  |  |
| C Cartesian 31D OpenPI policy |  |  |  | NOT RUN |  |  |
| D 分布式拒绝 |  |  |  | NOT RUN |  |  |
| E Joint 29D WebSocket |  |  |  | NOT RUN |  |  |
| E Cartesian 31D WebSocket |  |  |  | NOT RUN |  |  |
| F Joint 29D 无动力 dry-run |  |  |  | NOT RUN |  |  |
| F Cartesian 31D 无动力 dry-run |  |  |  | NOT RUN |  |  |
| G 有动力硬件 |  |  |  | NOT RUN |  |  |

状态只允许：`PASS`、`FAIL`、`BLOCKED`、`NOT RUN`。预期失败用例只有在 harness 确认失败类型、发生时点和无副作用后才能记为 `PASS`。

### 5.3 性能、故障与偏差

```text
冷启动时间：
首请求延迟：
稳态 p50 / p95 / p99：
吞吐量与并发数：
峰值 GPU allocated / reserved：
GPU ECC / Xid / OOM：
网络 RTT / 丢包 / 重连结果：
故障注入项目及安全状态：
硬件 tracking error 与 controller fault：
未执行项目及原因：
与计划的偏差、批准人和风险：
遗留问题：
```

### 5.4 最终结论

最终结论必须限定到实际通过的最高层级，例如：

- “仅锁定软件测试通过；GPU、网络和硬件未运行。”
- “指定 GPU 上 checkpoint 推理 smoke test 通过；未验证多 GPU或 WebSocket。”
- “WebSocket 环回通过；未绑定外部接口，未连接机器人。”
- “无动力 dry-run 通过；未执行有动力运动。”
- “仅在所记录 robot/controller/阈值/runbook 下完成受控运动测试；不代表其他硬件或生产环境。”

不得使用笼统的“服务器验证完成”“支持真实机器人”或“production ready”。操作员、独立复核人及现场安全负责人分别签字，并附上所有日志哈希和未完成项。

## 6. 相关文档

- [项目概览](../README.md)
- [服务器 Coding Agent 接管说明](server-handoff.md)
- [PyTorch 训练与部署契约](pytorch.md)
- [开发规范](agent.md)
- [推理机环境](inference-env.md)
- [文档索引](README.md)
