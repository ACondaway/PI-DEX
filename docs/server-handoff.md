# 服务器 Coding Agent 接管说明

本文是服务器端后续开发的执行入口，记录剩余工作、实施顺序、固定决策和完成标准。它不是测试
报告；任何 `PASS` 必须按 [服务器验证清单](server-validation.md) 在目标服务器留下命令、日志和
制品哈希。实现语义以 [PyTorch 训练与部署契约](pytorch.md) 和
[开发规范](agent.md) 为准。

## 1. 接管时的仓库状态

当前仓库已经实现双动作表示、动作派生/编解码、padding-neutral PyTorch loss、normalization
契约、checkpoint sidecar 校验、策略适配、launcher 接缝和安全 dispatch 协议，但尚未完成真实
训练闭环。接管 agent 必须先重新核对实际 Git commit 和工作树，不能把本文编写时的本地状态
当成服务器上的事实。

当前明确的剩余项如下：

- **大规模多节点 / DDP**：已提供 `DistributedSampler` + DDP wrap + rank-0 checkpoint，以及
  火山引擎 MLP 入口（`pi-dex-volc-train` / `scripts/volc_ddp_train.sh`）。首个闭环仍可单机
  运行；多机正式放行需按验证清单阶段 D（分片、global batch、rank-0 ckpt、resume）留证。
- observation contract 已审阅：`configs/site/joint_29d_observation.reviewed.json`
  （`reviewed_by=congsheng`）。未审阅模板仍保留供对比测试。
- `pi05_base` 转换 + `parity_pi05` 轨迹对比已有本地证据；验证清单 A.2 仍缺发布方**预期**
  source manifest 批准与正式留证流程。
- 真机推理入口已落地（`pi-dex-realtime-infer` / `pi_dex.realtime_*`）；完整
  `BimanualController` 租约 / 急停 / 硬件 harness（阶段 4–6）仍未完成。
- Cartesian / FK 未开始。
- B.1/B.2 完整质量验收（held-out 指标、overfit 阈值登记）尚未按验证清单闭环记 PASS。

## 2. 已固定的技术决策

后续实现不得在没有新的设计评审和迁移说明时改变以下决定：

1. 优先打通 PyTorch π0.5 full fine-tuning；JAX 只用于源 checkpoint 和转换 parity 参考。
2. 两种 Sharpa 表示都默认从官方
   `gs://openpi-assets/checkpoints/pi05_base` 初始化。`pi05_droid` 仅作为受控消融实验，
   `pi05_libero` 不作为 Sharpa 起点；不得默认随机初始化。
3. 先完成 `joint_29d`，再完成 `cartesian_31d`。Joint 不依赖 FK，能够先隔离数据、训练和
   checkpoint 问题；Cartesian 只有在机器人描述、标定与 FK golden pairs 可用后才能放行。
4. 两种模式必须复用一套 first-party dataset/runner/checkpoint/service 框架，通过
   `--action-representation` 切换；不得维护两套逐渐漂移的训练实现。
5. OpenPI 模型侧始终保持 `action_dim=32`、`pi05=True`、
   `discrete_state_input=True` 和 `action_horizon=2*K`。单手有效维为 Cartesian 31 或 Joint
   29，模型序列严格为 `L0,R0,L1,R1,...`。
   `joint_29d`/`cartesian_31d` 是 launcher 的 representation 值，不是 OpenPI config 名；仓库
   当前还没有 first-party、已注册的 PI-DEX `TrainConfig` 名。除站点审阅的 `K` 外，当前
   `create_pi05_model_config` 默认固定 `bfloat16`、`gemma_2b`、`gemma_300m` 和
   `max_token_len=448`（按全量 SharpaOpenData pi0.5 tokenize 扫描覆盖最长指令；旧实验若
   仍用 200 须在 config 中显式写出）。两种模式只有这些语义字段和 `K` 完全一致时才能共享
   converted base；
   compile mode 虽不进入语义指纹，也必须作为运行参数记录。
   Runner 必须在构造模型前要求 `TrainConfig.pytorch_training_precision == model.dtype`：可部署
   训练两者均为 `bfloat16`，独立 float32 诊断两者均为 `float32`；不得像 stock runner 一样在
   启动后静默改写其中一个值。
6. 当前数据派生只支持标定后的 absolute commanded joint positions。配置能够枚举 delta 或
   residual 不代表数据和训练路径已经支持它们。
7. 两种模式可以在 OpenPI model config 指纹完全相同时共享只读的 converted base 权重；它们
   不得共享 normalization asset、训练 checkpoint、`asset_id` 或 `pi_dex.json`。
8. 当前可部署 checkpoint 必须声明 `bfloat16`。float32 可用于独立诊断，但不能进入当前部署
   验收。
9. 单机单进程仍是默认路径；多机 DDP 通过 torchrun / 火山 MLP 入口启用（``--distributed`` 或
   ``RANK``/``WORLD_SIZE``）。``--batch-size`` 为 **local** batch；global batch =
   local × world_size。仅 rank-0 原子发布 checkpoint。上游 PyTorch 路径仍不支持 LoRA、
   FSDP、EMA 或 mixed-precision/AMP 训练。

## 3. 接管顺序

### 阶段 0：冻结源码和可复现环境

- 确认本次接管的 commit、remote、vendored `openpi/` tree、工作树状态和所有新增文件均已
  纳入版本控制。
- 生成并审阅根 `uv.lock`；环境建立后不得出现未记录的 lock 改写。
- 在隔离 uv cache 和 copy link mode 下建立 OpenPI 环境，验证精确的
  `transformers==4.53.2`，安装并逐文件验真 `transformers_replace`。不能在共享 hardlink
  cache 上直接覆盖 Transformers。
- 记录 Python、uv、PyTorch、CUDA、Transformers、driver、GPU 和补丁文件的加载路径及哈希。

完成标准：目标服务器能够从已提交的两个 lock 和受控补丁重建环境；全新 Python 进程中的补丁
自检通过，工作树和共享 cache 未被隐式改写；验证阶段 B 的基础 GPU/bfloat16 smoke 通过。
根 lock 的生成属于验证前开发：必须审阅、提交，然后从新的固定 clean commit 开启 A/A.1/B
验证，不能在同一次验证 run 中临时生成后宣称可复现。

### 阶段 1：固定数据语义并实现 first-party dataset

在写 HDF5 loader 前，必须将以下站点决定写进版本化配置并由数据负责人确认：

- `K`、30/60 Hz timebase、实际控制频率和 episode 尾部窗口策略；
- `state[state_dim]` 的逐列来源、顺序、单位、absolute/delta 语义和缺失策略；
- 三个 OpenPI 图像槽的 HDF5 映射、RGB 通道顺序、layout、dtype/range、resize/crop 和 mask
  语义。`head/stereo/lefteye → base_0_rgb`、左右 wrist fisheye → 对应 wrist 槽只能作为待审阅
  候选；第四个 head view 如何处理不得由 loader 猜测；
- prompt 来源、annotation 区间规则、无 prompt 样本策略和文本规范化；
- robot/embodiment、左右 arm/hand joint order、镜像映射和 command semantics 版本；
- episode 级 train/validation/test split、去重/泄漏规则以及坏样本报告策略。

Dataset 必须：

- 按 JPEG EOI 解码 padded 图像，显式产出 OpenPI 所需三个 image key 和对应 bool mask；
- 使用每个 HDF5 group 自身的 `aligned_index`，不使用 `2*k` 假设；
- 每个样本产出一维 `state` 和每手 `[K,D]` absolute commanded actions；
- 在进入 OpenPI 前校验图像 layout/dtype/range，不依赖上游用 `shape[1] == 3` 猜测
  BCHW/BHWC 的启发式；
- 首个闭环固定 `num_workers=0`。只有实现 h5py per-worker 打开/关闭、worker seed、异常清理和
  确定性测试后才能启用多 worker，不能继承父进程 HDF5 handle；
- 输出 dataset manifest、拒绝/跳过统计和 episode 级 split manifest。

完成标准：可在不初始化 CUDA 的情况下，对小型受控数据集完成 schema、窗口、图像、prompt、
state、左右手 sentinel 和异常样本验证；Joint 路径明确未创建或调用 FK。

### 阶段 2：建立受控 `pi05_base` PyTorch 初始化制品

必须提供纳入版本控制的 converter wrapper/harness，完成：

1. 先下载到隔离且可写的 acquisition cache，验证来源与逐文件 manifest 后复制/原子发布为
   hash-pinned immutable snapshot；converter 输入是该 snapshot 中的
   `.../pi05_base` 根目录，不是 `.../pi05_base/params`。
2. 记录官方 URI、预期 manifest、现场逐文件 size/SHA-256 manifest、PI-DEX root commit、
   `HEAD:openpi` tree OID、vendored converter 文件哈希、wrapper commit/lock/source hash、完整模型
   配置、precision、命令、峰值主机内存和磁盘占用。
3. 用 `create_pi05_model_config(spec)` 的同构配置转换并加载：`pi05=True`、32D、`2*K`、
   discrete state、Gemma variants、token limit 和 dtype 必须固定。上游没有名为
   `pi05_base` 的训练 config；不得借用 horizon/state 语义不同的 LIBERO/DROID config 来掩盖
   这一缺口。
4. 修补或包裹上游 converter 的风险：按配置而非路径字符串选择 π0.5 AdaRMS；检查
   `strict=False` 返回的 missing/unexpected keys；拒绝随机保留参数；不信任 converter 的
   assets 复制或简化 `config.json`。当前 mapper 硬编码 `gemma_2b` PaliGemma 结构和
   `gemma_300m` expert；在实现通用 mapper 前 wrapper 必须拒绝其他 variants。
5. 严格加载生成的 `model.safetensors`，并使用固定 post-transform observation、actions、
   noise 和 flow timestep 做 JAX↔PyTorch vector-field parity。容差必须在运行前登记。

Vendored `maybe_download()` 会创建目录/锁、调整 cache 权限并可能刷新过期条目，不能直接指向
典型只读挂载。服务器必须把“可写 acquisition cache”和“校验后的 immutable snapshot”分开；
训练/推理的 tokenizer 还要实现 first-party local-path resolver，直接读取 snapshot 中已验哈希
的文件，不能在运行时再次调用硬编码 GCS downloader。对 checkpoint 的 wrapper 同样不得让
runtime 在 immutable snapshot 内写 lock 或变更权限。

转换必须写入全新 staging 目录，在覆盖率、strict load、parity、完整逐文件 manifest 和
provenance record 全部通过后，才原子发布到不存在的目标；失败清理 staging，已存在目标必须
fail closed。转换产物只是只读 initialization artifact。不得给它伪造 Sharpa normalization 或
`pi_dex.json`，也不得直接交给 PI-DEX policy/server 部署。

完成标准：源和输出 provenance 完整，参数映射无未批准遗漏，目标 PI-DEX model config 严格加载
成功，parity 满足预登记阈值。仅生成 `model.safetensors` 或 converter 退出码为零不算完成。

### 阶段 3：打通 `joint_29d` 单机训练闭环

实现 first-party runner，且不得直接调用 stock `openpi/scripts/train_pytorch.py`：上游脚本对
完整 32D elementwise loss 做 `mean()`，不满足 PI-DEX 的 29D/31D padding-neutral loss。
Runner 必须使用 `PiDexPytorchTrainer`，并负责：

- 从阶段 2 的 converted `pi05_base` 实际加载权重，在创建 optimizer 前验证全部 key/shape；
- 绑定 launcher 返回的同一 `BimanualActionSpec`，构造 dataset、normalization、OpenPI
  transforms、model、optimizer、scheduler 和训练日志；
- 启动时记录按参数名排序的全部参数与 `requires_grad` 参数的 name/shape/numel manifest 和
  SHA-256，并证明两者集合一致；若冻结任何模型参数，该实验不得称为 full fine-tuning，除非
  另有明确设计评审和能力命名；
- 用每手 `[K,29]` actions 和 `[29]` stats 训练，证明 3 个 padding 维不参与
  stats/noise/loss；
- 保存并恢复 model、optimizer、scheduler、global step、Python/NumPy/Torch/CUDA RNG、数据
  shuffle/sampler 或明确的数据游标、训练配置、dataset/split manifest 和父 base provenance；
- 在临时目录写完整制品后原子发布，定义失败清理、保留策略和 resume 行为。
- PI-DEX experiment/run config ID、`asset_id` 和 checkpoint 根目录必须包含 representation 与
  稳定 run ID；这里不是尚不存在的 OpenPI TrainConfig registry 名。manager
  必须拒绝已存在的发布目标、同一 `asset_id` 的跨模式复用，以及 Joint/Cartesian 写入同一
  checkpoint 根目录；不能靠覆盖旧目录实现重跑。

在启动真实训练前，先通过服务器验证 B.1 的相同配置资源门控。完成标准：受控小数据集完成
`dataset → normalization → load base → forward/backward → save → new-process resume → internal strict-load smoke`；
连续训练与保存恢复训练满足预登记连续性标准；跨模式 spec/stats/checkpoint 均 fail closed。

### 阶段 4：打通 `cartesian_31d`

- 提供版本化 Sharpa URDF/MJCF、标定、base frame、左右 wrist link、rotation-6D 约定和
  first-party `ForwardKinematicsProvider`。
- 使用可信 joint→pose golden pairs 验证两侧位置、旋转、单位和映射；不能用数据中的可选
  `state/*/tcp_pose` 替代 commanded-action FK。
- 将 URDF/MJCF、标定文件、FK provider 源码 commit/hash、golden-pair manifest、预登记容差和
  结果哈希写入实验/checkpoint lineage；spec 中的版本字符串不能代替这些制品级 provenance。
- 复用阶段 1–3 的 dataset、runner 和 checkpoint manager，仅通过 representation 和 FK
  factory 切换；重新计算独立 `[31]` stats。

在启动真实训练前同样通过 B.1。完成标准：完成与 Joint 相同的训练/resume/internal-load/cross-mode
矩阵；缺少 FK/标定时在数据读取或
CUDA 初始化前失败，Joint 传入 FK 时也失败。

### 阶段 5：训练质量验收

链路可运行不等于策略有效。每个模式必须独立预登记并报告：

- episode 级 held-out split、seed、训练步数和接受阈值；
- train/held-out loss、gradient norm、学习率、吞吐、显存和坏样本计数；
- 小样本 overfit smoke；
- 分手、分动作子空间的物理单位指标：Cartesian 的 wrist translation、rotation geodesic、hand
  joints；Joint 的 arm/hand joints；
- `[L0,R0,...]` 表示的左右手交换、单侧退化、同步性、耦合和 deinterleave 后轨迹连续性。

`2*K` 交错只保证 shape 可兼容 OpenPI；模型会把它们视为不同 horizon positions，尚未证明其
等价于预训练时序语义。必须通过真实双手数据验证收敛和行为，不能从 checkpoint 能加载推断成功。

完成标准：分别完成服务器验证 B.2，离线质量达到预登记阈值，并清楚标注它只证明所用数据与
指标，不证明阶段 C policy、真机任务成功或硬件安全。

### 阶段 6：服务、控制器和硬件

必须完整执行 `server-validation.md` 的 C–G，而不是只实现一个 WebSocket launcher。至少交付
first-party policy/server entrypoint、带认证/隔离与消息大小/deadline/安全错误边界的 transport
wrapper、分布式 fail-closed probe、独立 controller watchdog、Sharpa SDK controller adapter、
reset/recovery 状态机、无动力 harness 和现场 runbook；随后依次完成真实 checkpoint policy smoke、受控 WebSocket、
无动力 dry-run，最后才是经现场批准的最小范围有动力测试。不得直接发布 vendored stock
WebSocket server，也不得跳过独立硬件急停与 controller 限位。

### 阶段 7：快系统与触觉

快系统和外部触觉编码器在两个慢系统训练闭环后接入。开始前必须单独冻结特征 schema、频率、
延迟预算、训练目标、residual/absolute 语义、触觉缺失 fallback 和安全限幅。它们当前属于
deferred milestone，不阻塞阶段 0–6，也不得在完成前宣传为已有能力。

### 阶段与服务器验证映射

接管阶段与 [服务器验证清单](server-validation.md) 的唯一映射是：阶段 0 对应 A/A.1/B；阶段 1
是 A.2 前的数据/spec 前置；阶段 2 对应 A.2；A.3 独立验证仓库 launcher 合约；阶段 3/4 在每个
真实配置先过 B.1 后产出训练闭环；阶段 5 完成两种模式的 B.2；阶段 6 完整执行 C–G，其中 D
是 E 前必须通过的分布式 fail-closed 边界。前置阶段未通过时不得跳到后续验收。

## 4. Checkpoint manager 最低制品集合

训练 checkpoint 与仅供推理的发布 checkpoint 可以有不同的内部文件，但 manager 至少必须使
以下信息可恢复和可审计：

```text
model.safetensors
pi_dex.json
assets/<asset_id>/norm_stats.json
optimizer state
scheduler state or exact reconstructable schedule config
global step
Python / NumPy / Torch CPU / each CUDA RNG state
data shuffle/sampler state or exact next-sample cursor
training config and BimanualActionSpec
representation-scoped config name / asset_id / checkpoint root / run ID
all-parameter and trainable-parameter name-shape-numel manifests and hashes
dataset/split manifest hashes
parent pi05_base source and converted artifact hashes
tokenizer URI and file hash
```

Cartesian 还必须记录 URDF/MJCF 与标定文件哈希、FK provider commit/hash、golden-pair manifest、
容差和结果哈希；Joint 对这些字段必须明确为不适用且不得加载对应 provider。

当前 `save_training_contract` 只负责验证并原子写 `pi_dex.json`，不会替 runner 保存上述训练状态。
若 manager 只恢复 weights，不得把它描述为可复现 resume。

## 5. 接管时不得猜测的事项

以下字段缺少经审阅的真实值时，相关阶段应标为 `BLOCKED`，不能用占位字符串、shape 或单个
样本推断：

- robot/embodiment ID、firmware、joint order、左右镜像和 command semantics version；
- `K`、timebase、control frequency、state 列布局和单位；
- 三个模型相机槽、crop/resize、image mask 与 prompt 规则；
- Cartesian URDF/MJCF、标定、base frame、wrist links 与 rotation-6D convention；
- normalization dataset、episode split、数据质量过滤和训练/部署接受阈值；
- checkpoint、tokenizer 和转换制品的预期发布 manifest/hash；
- controller 限位、deadline、watchdog、hold/recovery 和现场安全阈值。

## 6. 最终交付标准

服务器 agent 每次交付必须报告：完成到上述哪个阶段、修改的 commit、执行过的命令和精确结果、
未执行项目、制品与日志哈希、已知风险和下一阻断项。`joint_29d` 完成不代表
`cartesian_31d` 完成；训练闭环完成不代表质量、网络或真机安全已经通过。不得使用笼统的
“训练已打通”“服务器验证完成”或 “production ready”。
