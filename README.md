# PI-DEX

PI-DEX 是面向 Sharpa North 双手灵巧操作的研究代码，基于 OpenPI 的 π0.5（`pi05`）。当前首个开发里程碑聚焦 PyTorch 训练和部署的动作边界，尚未宣称具备真实机器人端到端能力。

服务器 Coding Agent 接管时应先阅读
[服务器接管说明](docs/server-handoff.md)，再按
[PyTorch 训练与部署契约](docs/pytorch.md) 实现，并使用
[服务器、GPU 与硬件验证清单](docs/server-validation.md) 留存证据。三份文档分别回答“下一步做什么”、
“实现必须满足什么语义”和“如何证明已经完成”，不能互相替代。完整文档列表见
[docs/README.md](docs/README.md)。

当前版本为 `0.1.0` 研究预览；`pi_dex` 公共 Python API 均为实验性接口。动作数值布局、metadata schema、checkpoint 和 wire 语义通过独立版本及严格字段校验拒绝不兼容输入，但数据集 factory、FK provider 和 controller lease 协议预期在 Sharpa SDK 与真实数据接入时替换或收窄；升级时必须按文档迁移，不应依赖未记录的实现细节。

当前已经实现：

源码按职责分包（`src/pi_dex/`）：

| 包 | 内容 |
|----|------|
| `core` | action / spec / normalization / training_contract |
| `data` | observation contract、Sharpa HDF5、episode split、norm compute |
| `training` | OpenPI 训练、checkpoints、DDP / Volcano launcher |
| `weights` | π0.5 convert / parity |
| `serve` | WebSocket policy server、deployment wire |
| `robot` | 真机 OpenPI Runtime + harobotsDL NorthZmqEnv |

- 两种显式单手动作表示及其统一 32D OpenPI 编解码：`cartesian_31d` 补 1 维，
  `joint_29d` 补 3 维，并按 `L, R` 双手交错/解交错；
- 显式的 horizon、时间基准、坐标系、动作模式、rotation-6D、手臂/手部关节顺序与镜像映射、标定版本和延迟契约；
- 基于各 HDF5 group 自身 `aligned_index` 的 30/60 Hz 动作窗口选择；
- `cartesian_31d = wrist position(3) + rotation-6D(6) + hand joints(22)` 必须注入
  标定 FK provider；`joint_29d = arm joints(7) + hand joints(22)` 保留原始 commanded
  joints 并禁止注入 FK；Joint spec 的 Cartesian/FK-only 字段必须为 `None`；
- 按所选有效宽度 31/29 计算的逐手/共享归一化统计，以及 normalization asset 指纹；
- 保持 OpenPI 原始 `PI0Pytorch` 与 checkpoint shape 不变的 padding-neutral 训练核心；
- `pi-dex-train-pytorch` 动作表示选择与外部 `module:callable` runner 接缝；
- first-party Sharpa `joint_29d` HDF5 dataset（JPEG EOI、state、双手 `[K,29]` actions）与
  `pi_dex.training.training_runner`（validate-data / compute-norm-stats / train / synthetic-smoke）；
- 多节点 DDP（``DistributedSampler``、rank-0 checkpoint）与火山引擎入口
  （``pi-dex-volc-train`` / ``scripts/volc_ddp_train.sh``）；
- 真机推理（``pi-dex-robot-client``：OpenPI ``Runtime`` + ``ActionChunkBroker`` +
  harobotsDL ``NorthZmqEnv``；订阅 ``north_observation``、发布 ``inference/action``）；
- WebSocket model server（``pi-dex-serve``：GPU 侧收 obs / 回 action）；
- 离线 convert/infer smoke（``pi-dex-realtime-infer``）；
- 受控 convert wrapper（`python -m pi_dex.weights.convert_pi05`）、strict weight load
  （`load_verified_pi05_base`）与最小训练 checkpoint manager；
- checkpoint 对 action、OpenPI 模型/tokenizer 配置、真实权重文件与 normalization asset 的指纹绑定，以及 PyTorch policy 加载；其中 tokenizer 仅绑定配置，实际 model 文件字节仍属下述外部边界；
- 只发布已反归一化物理动作的服务 adapter；
- 带 `peek/commit` 确认、客户端 observation 快照、chunk 序列/控制周期、可信 controller 时钟、唯一 lease、抗前跳/回拨的原子时间窗、不可变限幅，以及 recovery epoch 故障锁定的双手 dispatch 协议。

当前明确未实现或不在范围内：

- FSDP / LoRA / AMP（DDP 已支持：torchrun + ``pi-dex-volc-train``）；
- 完整 ``BimanualController`` 租约 / 急停 / 硬件验收（SDK↔policy 推理入口、
  ``pi-dex-serve`` 与 Zenoh 桥已有；lease / e-stop / watchdog 仍缺）；
- `pi05_base` 发布方预期 manifest 批准与验证清单 A.2 正式留证；
- Sharpa North 的 URDF/MJCF、FK 和机器人标定（cartesian_31d）；
- B.1/B.2 完整质量验收阈值与 held-out 指标闭环；
- 服务、控制器与硬件（阶段 6）；
- delta/residual action 的数据派生和训练；当前 commanded-joint 数据路径只接受
  `ActionMode.ABSOLUTE`；
- 快系统和外部触觉编码器接入。

Checkpoint 起点固定为官方 `gs://openpi-assets/checkpoints/pi05_base`。`pi05_droid` 只允许作为
有明确实验假设的对照，`pi05_libero` 不作为 Sharpa 的初始化起点。Cartesian 31D 与 Joint
29D 可以共享经过严格验证的 base 权重字节，但必须分别计算 normalization stats，并生成不可
混用的训练 checkpoint、assets 和 `pi_dex.json`；转换后的 base 本身不是可部署的 PI-DEX
checkpoint。

部署还有几项必须由外部基础设施闭合的边界：PaliGemma tokenizer 的实际文件字节尚未随
checkpoint 固化；同步推理调用必须由 transport deadline 和 controller watchdog 提供
有界失败；vendored OpenPI WebSocket server 本身没有认证/TLS、消息大小限制，还会把
traceback 返回客户端，因此只能用于受控环回开发，不能直接作为生产入口。服务器/GPU/
OpenPI/WebSocket/硬件验证清单见
[docs/server-validation.md](docs/server-validation.md)。

启动接缝显式选择一种表示：

```bash
# 开发环境（miniconda，见 environment.yml）
source /mnt/netdata/Team/Personal/congsheng/miniconda/etc/profile.d/conda.sh
conda activate pi-dex
# 首次或依赖变更：bash scripts/setup_conda_env.sh

# 推理机环境（对齐开发机）：见 docs/inference-env.md
# bash scripts/setup_inference_env.sh --install-miniconda
```

启动接缝显式选择一种表示：

```bash
# 直接关节动作：禁止 FK
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner your_project.training:run -- --config /server/path/train.yaml

# Cartesian 动作：必须提供经标定的零参数 FK factory
pi-dex-train-pytorch \
  --action-representation cartesian_31d \
  --runner your_project.training:run \
  --fk-provider-factory your_project.kinematics:create_fk \
  -- --config /server/path/train.yaml
```

根包日常验证：

```bash
conda activate pi-dex
ruff check src scripts tests
ruff format --check src scripts tests
pytest tests -m "not manual"
```

vendored `openpi/` 仍保留其上游 uv 工具链；根目录 PI-DEX 开发环境改用 miniconda（`environment.yml`），不再使用根 `uv.lock`。
`--` 后的参数原样交给外部 runner。runner 必须先用
`context.bind_action_spec(spec)` 绑定实际贯穿数据、OpenPI、loss、checkpoint 和部署的同一份
spec；Cartesian 还必须通过 `context.create_kinematics()` 获取 FK。若成功返回前未完成这些
握手，launcher 会拒绝退出为成功。示例中的 module、callable 和路径都是占位符；仓库
尚未提供可替换它们的 first-party HDF5 runner。这些缺口不会使用 `state/*/tcp_pose`、假定
`2*k` 对齐、伪造 FK、跳过反归一化或顺序写左右手来掩盖。PyTorch 集成说明见
[docs/pytorch.md](docs/pytorch.md)。项目约束以 [docs/agent.md](docs/agent.md) 为准。
训练 / 数据 / 推理环境分别见 [docs/training.md](docs/training.md)、
[docs/dataset.md](docs/dataset.md)、[docs/inference-env.md](docs/inference-env.md)。
文档索引：[docs/README.md](docs/README.md)。

当前 action `layout_version=3`、`metadata_schema_version=3`，部署 wire 为 v3。v3 在固定
32D 模型投影内绑定所选 31D/29D 表示及有效 mask；旧 v2 metadata/checkpoint/wire 必须
显式离线迁移和重新验证，不能原样加载。
