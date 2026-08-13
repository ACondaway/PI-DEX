# 服务器、GPU 与硬件验证清单

本文档只定义必须在目标服务器或机器人现场执行的验证流程，覆盖 GPU、vendored OpenPI、分布式边界、WebSocket 服务和硬件接入。它不是已完成验证的证据，也不能替代站点安全规程、机器人厂商说明或控制器验收。

> **当前状态：BLOCKED / NOT RUN。** 本文中的命令未在编写文档的本机执行；当前工作树还没有根 `uv.lock`，因此阶段 A 已按本文规则阻塞。未验证 CUDA、GPU 显存、真实 checkpoint、网络服务、控制器或机器人。任何交付说明都不得据此宣称多 GPU、WebSocket 生产服务或真实硬件能力已经可用。

## 1. 结论边界

- 单元测试通过只能证明被测试的软件边界，不能证明目标 GPU、驱动、checkpoint、网络或机器人安全。
- GPU smoke test 通过不能证明完整模型能在目标显存内加载或稳定推理。
- policy 能产生动作不能证明动作的单位、坐标系、关节顺序、限位或硬件执行正确。
- 当前 PI-DEX 自定义数据 loader 在 `torch.distributed` 已初始化时应立即拒绝运行；在另有实现和测试前，不得宣称支持 DDP 或多机训练。
- 仓库当前没有可直接用于生产的训练 CLI、完整 checkpoint manager、站点 WebSocket launcher、Sharpa SDK controller 或硬件急停实现。服务器和硬件测试必须使用经评审、纳入版本控制的站点接入程序，不能用临时脚本补齐后宣称能力已完成。
- 任一阶段为 `FAIL` 或 `BLOCKED` 时不得继续到依赖它的阶段。硬件阶段还必须满足现场负责人的独立放行条件。

## 2. 前置条件

### 2.1 软件与制品

- 使用一个明确的 Git commit；工作树无未记录改动，或所有改动都有单独补丁和审阅记录。
- 根目录和 `openpi/` 的锁文件必须已提交。若根目录没有 `uv.lock`，服务器验证判为 `BLOCKED`；不得在服务器临时生成锁文件后声称结果可复现。
- checkpoint 必须来自受控位置，并至少包含：

  ```text
  model.safetensors
  pi_dex.json
  assets/<asset_id>/norm_stats.json
  ```

- 测试所用 `BimanualActionSpec`、OpenPI model config、normalization `asset_id` 和 checkpoint 必须来自同一实验记录。
- OpenPI tokenizer/model 所需的远端资源必须已固定并缓存，或目标服务器具备经过批准的网络访问。应记录实际缓存对象的来源和哈希；仅记录配置字段不能证明远端 tokenizer 字节未变化。
- 服务器验证前必须登记 PaliGemma tokenizer 的源 URI、受控只读 cache 路径、发布方给出的预期 SHA-256 和现场实际 SHA-256；两者不一致时为 `BLOCKED`。获准联网或记录一个下载后的新哈希本身不能证明无漂移。
- 所有站点 smoke、distributed、WebSocket 和硬件 harness 必须来自单独的版本控制项目，并记录 commit、源码哈希及其依赖 lock 哈希。若它只使用 OpenPI 已锁定环境而没有额外依赖，也必须记录脚本 SHA-256。
- 服务器不得在日志中输出访问令牌、私有数据样本或完整原始 observation。

### 2.2 服务器与 GPU

- 目标系统、CPU 架构、内核、NVIDIA driver、CUDA runtime 和 GPU 型号已列入部署清单。
- GPU 支持目标 checkpoint 所要求的精度；当前 PI-DEX PyTorch policy 部署只接受声明为 `bfloat16` 的模型配置。
- 磁盘空间足以容纳环境、checkpoint、只读制品副本和测试日志；GPU 显存预算包含模型、KV cache、输入、临时张量和安全余量。
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
- robot/embodiment、FK 标定、左右手关节顺序、wrist link、坐标系、单位、绝对或 residual 语义均有可追溯版本。
- 控制器侧独立实施位置、速度、加速度、力矩、工作空间、碰撞和超时限制；具体阈值由机器人/站点负责人批准，不从模型输出推断。
- 已定义 observation 过期、时钟不一致、NaN/Inf、越界动作、网络断连、GPU 异常和进程崩溃时的 hold 或安全停机行为。

## 3. 固定测试身份与记录方式

先由操作员填写站点实际路径；示例值不得原样用于生产：

```bash
export PI_DEX_REPO=/srv/pi-dex
export PI_DEX_CHECKPOINT=/srv/checkpoints/REPLACE_WITH_CHECKPOINT
export PI_DEX_ASSET_ID=REPLACE_WITH_ASSET_ID
export PI_DEX_DEVICE=cuda:0
export PI_DEX_BIND_HOST=127.0.0.1
export PI_DEX_PORT=8000
export PI_DEX_TOKENIZER_MODEL=/srv/pi-dex-cache/paligemma_tokenizer.model
export PI_DEX_TOKENIZER_EXPECTED_SHA256=REPLACE_WITH_RELEASE_SHA256
cd "$PI_DEX_REPO"
```

每个命令都要记录：完整命令、开始/结束时间、退出码、stdout/stderr 日志位置及日志 SHA-256。日志路径不得位于 checkpoint 目录内，checkpoint 在整个验证期间保持只读。失败后不得只重跑成功片段；必须保留首次失败记录，并说明修复和完整重跑范围。

记录源码与制品身份：

```bash
git rev-parse HEAD
git status --short
git submodule status
uv --version
test -f uv.lock
test -f openpi/uv.lock
test -f "$PI_DEX_CHECKPOINT/model.safetensors"
test -f "$PI_DEX_CHECKPOINT/pi_dex.json"
test -f "$PI_DEX_CHECKPOINT/assets/$PI_DEX_ASSET_ID/norm_stats.json"
test -f "$PI_DEX_TOKENIZER_MODEL"
sha256sum uv.lock openpi/uv.lock
sha256sum \
  "$PI_DEX_CHECKPOINT/model.safetensors" \
  "$PI_DEX_CHECKPOINT/pi_dex.json" \
  "$PI_DEX_CHECKPOINT/assets/$PI_DEX_ASSET_ID/norm_stats.json"
printf '%s  %s\n' "$PI_DEX_TOKENIZER_EXPECTED_SHA256" "$PI_DEX_TOKENIZER_MODEL" \
  | sha256sum -c -
```

通过标准：commit、工作树状态、两个锁文件和三个 checkpoint 制品均有记录；实际哈希与实验发布记录一致。任一文件缺失或哈希不一致均为 `BLOCKED`，不能用重新下载或重建制品后的结果覆盖原记录。

## 4. 分阶段验证

### 阶段 A：锁定环境与软件基线

根环境只能使用已提交的锁定解析：

```bash
cd "$PI_DEX_REPO"
uv sync --locked --extra pytorch
uv run --locked --extra pytorch ruff check src tests
uv run --locked --extra pytorch ruff format --check src tests
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

这里的 `--locked` 只锁定 `openpi/` 基础项目；`--with-editable ..` 是由当前 Git commit
固定源码、但未写入 `openpi/uv.lock` 的本地 overlay。报告必须保留这一边界，不得把组合环境
描述成“由单一 lock 完整锁定”。若站点要求单一锁定解析，必须先通过经评审的 workspace 或
带哈希 wheel 发布流程把根包纳入共同依赖图，再更新本文命令；不得在验证现场临时改造。

通过标准：

- 两次环境同步未改写锁文件；执行后 `git status --short` 与执行前一致。
- Ruff、格式检查、根测试及 OpenPI 联调测试退出码均为零。根环境允许仅因未安装 OpenPI 而跳过 `test_openpi_normalization_roundtrip.py`；随后 OpenPI 环境中的两个定向联调文件不得 skip。
- 测试日志记录通过、失败、跳过和 deselected 的准确数量；不得将依赖缺失导致的 skip 记为通过。

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

### 阶段 C：真实 OpenPI checkpoint 与 policy

此阶段必须使用经评审的站点 smoke harness；仓库目前没有可直接冒充生产 launcher 的命令。harness 应调用 PI-DEX 的 checkpoint validator 和 `create_bimanual_trained_policy`，并显式传入 checkpoint、asset ID、spec、model config 和 `pytorch_device`。

至少验证：

1. 在加载模型前，`pi_dex.json`、模型配置、weights 字节、normalization asset 字节和解析后统计指纹全部匹配。
2. checkpoint 的 `state_dim` 与 inference observation 的一维 state 宽度一致；短 state、长 state 和额外时间轴均 fail closed。
3. policy 实际位于 `PI_DEX_DEVICE`，模型配置为 pi05、action width 32、action horizon `2*K` 且 dtype contract 为 `bfloat16`。
4. serving-time dense action input/output projection 已执行；31 个语义维参数不变，第 32 维对应的输入列、输出行和输出 bias 为零。
5. 使用固定 observation 和预先登记的 PyTorch 随机种子做至少两次公开 policy 推理；输出经解交错、去 padding 和反归一化后为 `left/right[E,31]` float32，值有限，metadata 与 spec 一致。GPU 非确定性未被另行消除时不得要求两次数值逐位相等。
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
  --device "$PI_DEX_DEVICE"
```

通过标准：全部契约和负向用例按预期通过或拒绝；推理输出、延迟和显存满足预先登记的阈值；无权重、normalization 或 tokenizer 来源漂移。只完成模型加载但没有真实推理不得记为通过。

### 阶段 D：分布式边界

当前验收目标是证明不受支持的分布式 loader 会安全拒绝，而不是证明 DDP 可用。使用经审阅的 harness 初始化 `torch.distributed` 后调用 PI-DEX 自定义 loader：

```bash
export PI_DEX_DISTRIBUTED_PROBE=/srv/pi-dex-site/distributed_probe.py
test -f "$PI_DEX_DISTRIBUTED_PROBE"
cd "$PI_DEX_REPO/openpi"
uv run --locked --with-editable .. torchrun \
  --standalone \
  --nproc-per-node=2 \
  "$PI_DEX_DISTRIBUTED_PROBE"
```

通过标准：

- 每个 rank 都在读取训练样本、执行 forward/backward 或写 checkpoint 之前观察到预期的 `NotImplementedError`。
- 进程组被正常回收，没有残留 worker、hang、重复样本消费或部分 checkpoint。
- harness 将“预期拒绝”转换为自身退出码零；未捕获异常导致的任意非零退出不能自动记为通过。
- 报告结论必须写成“分布式路径按设计 fail closed”，不得写成“DDP/multi-GPU supported”。

未来若实现 DDP，必须另行增加样本分片、global batch、loss/gradient parity、all-reduce、rank-0 原子 checkpoint、恢复一致性和故障注入测试；本文当前标准不能用于放行该能力。

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

- metadata 握手完整且与 checkpoint/spec/execution horizon 一致；`session_id` 是 32 位小写十六进制，每个响应精确回显该值，server restart 后生成新值，旧值或 mismatch 被拒绝；未知字段或旧 wire version 被拒绝。
- 合法 observation 返回有限的 `left/right[E,31]` float32、正确 clock domain、源时间戳和单调 chunk sequence ID。
- state 宽度错误、dtype/shape 错误、NaN/Inf、过期 observation、错误 clock domain 和非法时间戳均被拒绝，且没有产生可下发动作。
- 并发请求不会交叉污染 observation 或 chunk；响应超时和 backpressure 行为符合预先定义的 SLO。
- 主动断网、半开连接、永不返回的 inference、进程退出、GPU 异常、响应丢失和 server restart 后，独立 watchdog 在预登记 deadline 内使控制器进入 hold；旧 broker/session 不会静默续跑，必须重新获取并验证 metadata 后显式恢复。
- 超大消息在反序列化或分配无界内存前被有限的 message-size policy 拒绝；服务端错误响应不包含 traceback、绝对路径、凭据或内部配置。
- 日志不包含凭据或完整原始 observation；服务关闭后端口和 GPU 资源被释放。

环回全部通过后，才可在受控接口重复同一套探针，并记录网络路径、TLS/认证状态、延迟、丢包和防火墙规则。通过标准是所有正向、负向及断线恢复用例满足预登记标准；仅能建立 WebSocket 连接不算通过。

### 阶段 F：无动力硬件 dry-run

保持机器人驱动禁用或使用厂商认可的 simulation/dry-run 模式。服务输出只进入一个只记录、不执行的 controller adapter。

逐项确认：

- metadata 中 robot ID、embodiment、标定、左右手 joint order、wrist link、坐标系、单位和 action mode 与现场配置逐字匹配。
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
工作树状态：
根 uv.lock SHA-256：
openpi/uv.lock SHA-256：
站点 harness 项目 commit / lock SHA-256 / 脚本 SHA-256：
主机名、OS、kernel、CPU：
uv / Python / PyTorch 版本：
GPU index、UUID、型号、driver、CUDA、显存：
checkpoint 路径与三个制品 SHA-256：
tokenizer URI / cache path / expected SHA-256 / actual SHA-256：
asset_id / state_dim：
model config 摘要：
robot / embodiment / controller / firmware：
标定与 joint mapping 版本：
clock source 与实测偏差：
WebSocket bind、端口、TLS/认证/隔离策略：
预先批准的性能与硬件安全阈值：
```

### 5.2 命令与阶段结果

下表是复制后填写的空白模板，`NOT RUN` 只是未执行占位，不代表本文当前事实。当前仓库因缺少
根 `uv.lock`，阶段 A 是 `BLOCKED`；按第 1 节的依赖门控，阶段 B–G 在 A 解除前也不能执行。

| 阶段 | 测试 ID / 完整命令 | 开始/结束时间 | 退出码 | 状态 | 日志路径与 SHA-256 | 备注 |
|---|---|---|---:|---|---|---|
| A 环境 |  |  |  | NOT RUN |  |  |
| B GPU |  |  |  | NOT RUN |  |  |
| C OpenPI policy |  |  |  | NOT RUN |  |  |
| D 分布式拒绝 |  |  |  | NOT RUN |  |  |
| E WebSocket |  |  |  | NOT RUN |  |  |
| F 无动力 dry-run |  |  |  | NOT RUN |  |  |
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
- [PyTorch 训练与部署契约](pytorch.md)
- [开发规范](../AGENT.md)
