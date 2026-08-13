# PyTorch 训练与部署契约

本文记录 PI-DEX 当前可用的 PyTorch/OpenPI 集成边界。代码已经覆盖动作编码、训练 loss、归一化资产、checkpoint 契约和安全 dispatch；Sharpa dataset、可信 FK、训练 CLI 与控制 SDK 仍需由后续接入提供。

## 1. 统一动作规格

训练、checkpoint、策略服务和机器人进程必须共享同一个 `BimanualActionSpec`。下面的标识仅用于展示接口；接入真实数据前必须替换为经过验证的机器人、标定、时钟和控制语义。

```python
from pi_dex.spec import ActionMode
from pi_dex.spec import ActionTimebase
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization
from pi_dex.spec import Rotation6DConvention

spec = BimanualActionSpec(
    physical_horizon=8,
    timebase=ActionTimebase.RAW_CONTROL_60_HZ,
    control_frequency_hz=59.4,
    robot_id="<validated-station-id>",
    embodiment_version="<validated-embodiment-version>",
    coordinate_frame="<validated-base-frame>",
    action_mode=ActionMode.ABSOLUTE,
    hand_normalization=HandNormalization.PER_HAND,
    rotation_6d_convention=(
        Rotation6DConvention.MATRIX_FIRST_TWO_COLUMNS_COLUMN_MAJOR
    ),
    kinematics_calibration_version="<validated-calibration-version>",
    command_semantics_version="<validated-absolute-command-version>",
    left_arm_joint_order=tuple(f"left_arm_j{i}" for i in range(7)),
    right_arm_joint_order=tuple(f"right_arm_j{i}" for i in range(7)),
    left_hand_joint_order=tuple(f"left_hand_j{i}" for i in range(22)),
    right_hand_joint_order=tuple(f"right_hand_j{i}" for i in range(22)),
    hand_mapping_version="<validated-hand-column-and-mirror-version>",
    left_wrist_link="<validated-left-wrist-link>",
    right_wrist_link="<validated-right-wrist-link>",
    clock_domain="<shared-runtime-clock-domain>",
    max_group_timestamp_skew_ms=2.0,
    max_alignment_timestamp_error_ms=2.0,
    max_control_period_error_ms=8.0,
    max_observation_age_ms=50.0,
    max_command_lead_ms=25.0,
)
```

该规格表示 `K=8` 个物理双手控制步。单手逻辑动作是 `31D = wrist position(3) + rotation-6D(6) + hand joints(22)`；模型动作末尾补一个无语义零，因此 OpenPI shape 为 `[B, 2K, 32] = [B, 16, 32]`，序列严格按 `L0,R0,L1,R1,...` 排列。

metadata 中的 `layout_version=2` 同时版本化 31D/32D 数值布局及每个维度的语义归属；v2 在保持宽度不变的同时新增左右手关节列顺序和 `hand_mapping_version`，因此必须让仍只理解 v1 的 reader 双向 fail closed。`metadata_schema_version=2` 独立版本化必填字段集合。checkpoint 的 `pi_dex` mapping 不接受未知扩展；部署握手只额外允许并验证 `wire_format`、`execution_horizon` 与每次服务启动唯一的 `session_id`。旧的 `layout_version=1` checkpoint/服务必须经显式离线迁移并重新验证列映射，不能原样加载。

## 2. Sharpa 数据与 FK 边界

`data/schema.json` 提供每侧 7D 手臂和 22D 手部 commanded joint angles，但仓库没有可信的机器人描述或标定，不能自行推导 wrist pose。数据适配器必须构造四个 `CommandedJointGroup`、canonical `AlignedTimeline`、episode provenance，并注入实现 `ForwardKinematicsProvider` 的标定 FK：

```python
from pi_dex.sharpa_data import derive_bimanual_logical_action_chunk

chunk = derive_bimanual_logical_action_chunk(
    aligned_timeline=canonical_head_camera_timeline,
    provenance=verified_episode_provenance,
    left_arm=left_arm_command_group,
    left_hand=left_hand_command_group,
    right_arm=right_arm_command_group,
    right_hand=right_hand_command_group,
    start_aligned_frame=frame_index,
    spec=spec,
    kinematics=validated_fk_provider,
)
```

边界会验证 canonical HDF5 path、各 group 的 `N`、时间戳、采样周期、robot/embodiment/calibration、手臂与手部关节顺序、左右手列映射/镜像版本、wrist link 和 rotation-6D 正交性。构造 `CommandedJointGroup` 时必须传入数据源的实际 `joint_order`；`EpisodeActionProvenance.hand_mapping_version` 必须来自可追溯的数据转换/录制配置，不能根据 22 列数值猜测。每个 group 都使用自己的 `aligned_index`；实现不会用 `2 * frame_index`，也不会回退到 measured `state/<side>_arm/tcp_pose`。episode 级时间轴与关节数组保存在 immutable bytes backing 上，无法通过重新开启 NumPy `writeable` 标志绕过首次校验。

供训练的随机访问 dataset 在 repack/data transforms 后必须为每个样本产生：

```text
state         floating [D]
left_actions  floating [K, 31]
right_actions floating [K, 31]
```

图像、prompt、state 和 HDF5 I/O 的具体映射仍由后续 Sharpa observation dataset 负责。这里的 `state` 是一个未 batch 的一维向量；`[T,D]` 等额外时间轴不能静默展平，因为 π0.5 会把 state 编码进离散 prompt。

## 3. 归一化与 OpenPI loader

归一化统计必须在 padding 和交错之前计算。π0.5 路径严格要求 `state`、`left_actions`、`right_actions` 三个统计项，每项都有一维 `mean/std/q01/q99`；两侧动作统计 shape 都是 `[31]`。

```python
import dataclasses

from openpi.shared import normalize
from pi_dex.openpi_integration import compute_bimanual_normalization_stats
from pi_dex.openpi_integration import configure_bimanual_train_config
from pi_dex.openpi_integration import create_pi05_model_config

# base_train_config 必须由 Sharpa 接入层提供 DataConfigFactory、asset_id、
# checkpoint/assets 路径和图像/state/prompt transforms。
model_config = create_pi05_model_config(spec)
train_config = configure_bimanual_train_config(
    dataclasses.replace(base_train_config, model=model_config),
    spec,
)
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

norm_stats = compute_bimanual_normalization_stats(dataset, data_config, spec)
asset_id = data_config.asset_id
if asset_id is None:
    raise ValueError("Sharpa DataConfig must declare a normalization asset_id")
normalize.save(train_config.assets_dirs / asset_id, norm_stats)
data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
```

训练允许显式选择 `bfloat16` 或 `float32` 模型配置；当前 vendored OpenPI policy loader 会把选定参数转为 bfloat16，因此 PI-DEX 部署入口只接受训练时即声明为 `bfloat16` 的 checkpoint。若未来上游提供保持全 float32 的加载路径，必须先增加对应契约与集成测试再放开。

`HandNormalization.PER_HAND` 独立统计左右手；`SHARED` 在交错前池化两侧样本并要求保存的左右统计完全相同。

π0.5 loader 必须设置 `use_quantile_norm=True`。同时不能把 `repo_id` 设成 OpenPI 的特殊哨兵值 `"fake"`：上游会在该值下无条件跳过 normalization；PI-DEX 会在构建训练 loader 时显式拒绝这种配置。

标准 OpenPI LeRobot loader 会把 `action_horizon=2K` 解释为 `2K` 个连续物理时刻，因此 PI-DEX 必须传入已经按 `K` 取窗的自有 dataset：

```python
from pi_dex.openpi_integration import create_pytorch_data_loader_from_dataset

loader = create_pytorch_data_loader_from_dataset(
    dataset,
    train_config.data,
    train_config.assets_dirs,
    train_config.model,
    spec,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    seed=seed,
)
```

当前 loader 明确只支持单进程训练；若 `torch.distributed` 已初始化会立即报 `NotImplementedError`，避免上游无限 loader 与 `DistributedSampler` 组合后静默重复 epoch 或空转。`data/repack` output transforms 必须为空；model output 在首次配置前也必须为空，重复配置时则只允许已存在唯一的 `UnpackBimanualActions`。这样 PI-DEX 会在 inverse normalization 前把 `actions` 解成两侧字段，且不会和其他 model output transform 重排语义。

变换顺序固定为：

```text
repack/data inputs
→ Validate state[D] and optional left/right targets[K,31]
→ Normalize(state, left_actions, right_actions)
→ prompt/image/state model inputs
→ each hand 31D→32D
→ interleave L/R as [2K,32]
```

推理按逆序先解交错、丢弃第 32 维，再分别反归一化两手。部署入口还会在加载后的
`action_in_proj` 第 32 输入列及 `action_out_proj` 第 32 输出行/偏置上执行确定性零投影，
使 stock OpenPI denoising 中的随机 padding 不再反馈污染 31 个语义维；该策略写入
checkpoint 的 `padding_inference_policy` 契约。应始终通过
`create_bimanual_trained_policy` 加载，不要绕过该投影直接发布原始 OpenPI policy。

## 4. PyTorch 模型与训练 step

`action_dim` 必须保持 32，才能加载 π0.5 的 action projection 权重。下面是外层训练程序的集成骨架，并非完整 CLI：

```python
import jax
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from pi_dex.pytorch_training import PiDexPytorchTrainer

device = torch.device(device_name_from_runtime_config)
model = PI0Pytorch(train_config.model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
trainer = PiDexPytorchTrainer(
    model,
    optimizer,
    spec,
    gradient_clip_norm=1.0,
)

for observation, actions in loader:
    observation = jax.tree.map(lambda value: value.to(device), observation)
    actions = actions.to(device=device, dtype=torch.float32)
    result = trainer.train_step(observation, actions)
    detached_loss = result.loss
```

训练核心会把 target 与 flow-matching noise 的第 32 维置零，并只在 31 个有效维上求平均。trainer 要求每个可训练参数恰好属于一个 optimizer parameter group，并拒绝遗漏、重复或不属于 model 的参数。loss 在 backward 前必须为有限标量；backward 后无论是否启用裁剪都会检查总 gradient norm，非有限值不会进入 `optimizer.step()`。这些保护会产生必要的设备同步。trainer 不是 `nn.Module` wrapper；保存原始 model（DDP 场景则保存 unwrapped module）时，`state_dict` key 与 OpenPI 保持一致。返回的 loss 和 gradient norm 已 detach，不会因日志累计而保留 autograd graph。

## 5. Checkpoint 与 normalization 资产绑定

在临时 checkpoint 目录中保存模型、optimizer 和 normalization assets 后，由主进程写入 PI-DEX sidecar，再原子发布整个目录：

```python
from openpi.shared import normalize
from openpi.training import checkpoints as openpi_checkpoints
from pi_dex.checkpoints import load_and_validate_training_contract
from pi_dex.checkpoints import save_training_contract

# 必须先在同一个待原子发布的目录中保存真实权重和 stats：
# temporary_checkpoint_dir/model.safetensors
# temporary_checkpoint_dir/assets/<asset_id>/norm_stats.json
normalize.save(temporary_checkpoint_dir / "assets" / asset_id, norm_stats)

save_training_contract(
    temporary_checkpoint_dir,
    spec,
    model_config=train_config.model,
    norm_stats=norm_stats,
    asset_id=asset_id,
)

checkpoint_norm_stats = openpi_checkpoints.load_norm_stats(
    checkpoint_dir / "assets",
    asset_id,
)
load_and_validate_training_contract(
    checkpoint_dir,
    spec,
    model_config=train_config.model,
    norm_stats=checkpoint_norm_stats,
    asset_id=asset_id,
)
```

`pi_dex.json` 同时绑定 action/training 契约、影响模型结构或 tokenizer/state 输入语义的 OpenPI 配置、`model.safetensors` 字节、normalization asset 路径与序列化文件字节，并记录已解析统计内容的稳定 SHA-256 指纹与精确 state 宽度。保存 sidecar 前若上述实际文件不存在会立即失败；resume 和部署也必须从 checkpoint 读取 stats 并完成全部比对，不提供 metadata-only 的成功路径。部署入口会把所用的三类 artifact 复制到私有临时快照，随后对同一份副本完成验证和模型加载，避免路径在校验与使用之间被替换。公开的 `load_and_validate_training_contract` 只验证调用瞬间的路径内容；resume 管理器若在返回后再从可变原路径加载，仍须自行使用私有快照或原子发布目录，不能把一次校验当作永久授权。compile mode 属于运行时优化，不纳入模型语义契约。

当前契约绑定了 `max_token_len`、Gemma variants 与离散 state 模式，但 vendored OpenPI
仍从独立 GCS/cache 读取 PaliGemma tokenizer model，实际 tokenizer 文件字节尚未随
checkpoint 固化。服务器必须使用受控只读 cache 并记录该文件 SHA-256；未来把 tokenizer
资产纳入 snapshot 和 sidecar 前，不能把 checkpoint 描述为完全自包含。

## 6. 策略服务

服务端 adapter 只接受 OpenPI 已经完成 inverse normalization 的 `left_actions/right_actions[K,31]`。它不会把 raw `[2K,32]` 模型输出冒充米/弧度物理命令。

下面的 stock OpenPI server 片段只用于 `127.0.0.1` 上的受控开发联调。它没有服务端认证或
TLS、将消息大小设为无限、没有 inference/receive deadline，并会把异常 traceback 返回给
客户端。生产或机器人链路必须先由经评审的 transport wrapper 补齐这些边界，并由独立的
controller watchdog 在推理线程阻塞或连接半开时限时进入安全态。

```python
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from pi_dex.openpi_integration import create_bimanual_trained_policy

policy = create_bimanual_trained_policy(
    # 必须是第 3 节由 configure_bimanual_train_config 返回的同一语义配置。
    train_config,
    checkpoint_dir,
    spec,
    execution_horizon=execution_horizon,
    pytorch_device=device_name_from_runtime_config,
)
WebsocketPolicyServer(
    policy,
    host="127.0.0.1",
    port=listen_port,
    metadata=policy.metadata,
).serve_forever()
```

客户端 observation 必须额外携带：

```text
observation_timestamp_ns positive int, in spec.clock_domain
clock_domain             exact spec.clock_domain string
```

adapter 会在调用 OpenPI 前递归快照 dict/list/tuple、标量和数值 NumPy array，并从快照移除这两个 transport-only 字段；object/complex array、循环容器和自定义 leaf 会被拒绝。该快照可防止完成拷贝后的外部修改，但无法使正在被另一线程写入的多个 buffer 原子化；采集端必须在快照期间交出整棵 observation 的独占所有权或使用其自身锁。wire response 为：

```text
actions.left       float32[E,31]
actions.right      float32[E,31]
source_timestamp_ns positive int
clock_domain        spec.clock_domain
chunk_sequence_id   positive monotonic int (policy adapter instance scope)
session_id          32 lowercase hexadecimal digits, adapter instance scope
```

## 7. 机器人客户端与安全 dispatch

客户端启动时先验证完整 metadata，并从服务端声明的 execution horizon 构造 broker：

```python
from pi_dex.deployment import BimanualActionChunkBroker
from pi_dex.deployment import BimanualCommandDispatcher
from pi_dex.deployment import BimanualSafetyLimits

server_metadata = client.get_server_metadata()
broker = BimanualActionChunkBroker.from_metadata(
    client,
    server_metadata,
    spec,
    expected_execution_horizon=execution_horizon,
)

limits = BimanualSafetyLimits(
    spec,
    verified_left_min,
    verified_left_max,
    verified_right_min,
    verified_right_max,
)
dispatcher = BimanualCommandDispatcher.from_metadata(
    controller,
    server_metadata,
    spec,
    limits,
    expected_execution_horizon=broker.execution_horizon,
    clock_domain=spec.clock_domain,
)

observation["observation_timestamp_ns"] = observation_timestamp_ns
observation["clock_domain"] = spec.clock_domain
broker.dispatch_next(
    observation,
    dispatcher,
    target_timestamp_ns=target_timestamp_ns,
)
```

broker 使用 `peek → dispatch → commit` 语义：控制器成功前不会消费动作；dispatch 失败会锁定 broker，必须显式 `reset()` 后才能继续。broker 与 dispatcher 都应由同一份 metadata 构造，并绑定 policy 自身暴露的同一 `session_id`；每个 response 也必须回显它。换用另一 policy/server session、只复用旧 metadata 或直接拼接 response 都会 fail closed。只有 `from_metadata(...)` 构造的 broker 才能调用硬件 dispatch；它会保存完整握手 action spec，并在推理前要求 dispatcher 的 spec、session 与 horizon 都精确相等。直接构造的 broker 仅可用于 `peek/commit`，不能绕过握手下发硬件。`reset()` 要求 policy 提供可调用的同名 hook；上游 OpenPI `BasePolicy` 已提供默认 no-op，自行引入 episode-local 状态的 policy 则必须覆写并成功完成状态复位。broker 会串行化完整事务，并在调用远端 policy/WebSocket client 前递归快照 observation；采集端仍须在该短暂快照期间冻结整棵输入，快照完成后的修改不会影响序列化内容。dispatcher 也会串行化“动作快照→校验→apply→状态提交”，避免并发重放。若绕过 broker 直接调用公开 `dispatcher.dispatch(result, ...)`，调用方必须独占完整 result tree 直至同步调用返回，才能保证左右手数组形成同一时刻的快照。

每个缓存步都保留 source timestamp、clock domain、独立的 `chunk_sequence_id` 和 `chunk_step_index`。dispatcher 绑定服务端 metadata 验证出的 execution horizon，并验证 chunk 身份、严格连续的 chunk ID、连续且不越界的 step、同 chunk source、一致的跨 chunk 控制周期及递增 target；重放、跳号、乱序或退化/近似平行的 rotation-6D 向量都会 fail closed。它还会在单次 `apply_bimanual_action` 前完成双手 shape/dtype/finite/限幅、观测新鲜度、未来 target 和最大 lead 校验；任一失败都会请求经 controller 状态确认的 `hold` 并永久锁存该 dispatcher 实例。

具体 Sharpa controller 必须公开与 dispatcher 完全相等且运行期间不可变的 `action_spec`、`safety_faulted` 与单调 `recovery_epoch`，并在后端原子维护每个 recovery epoch 唯一的 dispatch lease。`acquire_dispatch_lease()` 必须绑定 spec、clock domain 和 epoch；若抛错或返回无效 token，不得遗留 active lease。`read_clock_ns()` 从该 lease 的可信 controller 时钟读数，调用方不再提供可用于准入判断的 current time。dispatcher 在获得自身锁后及 apply 前各重读一次时钟，并把第二次读数作为 `not_before_timestamp_ns`；controller 必须在其内部同一临界区验证 lease、epoch 和 `not_before_timestamp_ns <= now <= not_after_timestamp_ns < target_timestamp_ns`，然后才原子下发双手（或针对同一 target 先 stage 两侧再统一 commit）。上下界同时关闭 read→apply 期间的时钟前跳与回拨窗口。

`hold()` 必须原子锁存安全状态、使当前 lease 失效，并返回 `BimanualHoldReceipt(safety_faulted=True, recovery_epoch=<same epoch>)`。外部硬件安全恢复必须先原子进入已验证安全态并撤销旧 lease，成功后才可清除 fault、递增 epoch 并允许新 lease；随后调用方必须 reset broker 并重建 dispatcher。若旧 dispatcher 发现 epoch 已变化，它的 stale token 不得再 hold 新 epoch，因而只能本地锁存并报告 hold acknowledgement 失败；此时硬件安全性必须由已经完成的 recovery 事务保证。仓库尚无控制 SDK，不能声称已经实现硬件同步、急停或 safe-hold。当前 WebSocket reset 不会传播到服务端；未来有状态 policy 需新增 reset RPC 或在客户端显式管理。

wire v2 假定一个 policy adapter 实例服务一个机器人控制 session。adapter 或所在 server process 重启后，chunk ID 会重新起始；响应在运输中丢失时，后续 ID 也会在客户端观测为跳号。单纯重建 WebSocket 连接不会自动重置服务端 adapter 计数器，也不是安全恢复手段；客户端必须将任何连接丢失本身视为 session 失效，而不能假定连续 ID 足以识别一次无丢包的重连。旧 dispatcher 必须将连接失效、ID 重置或跳号视为安全故障并进入 hold。恢复流程是重新获取并校验 metadata、完成 controller 的硬件安全恢复，再新建 broker 与 dispatcher，不能在旧对象上自动续跑。wire 字段仍为 v2；机器人端旧的 caller-supplied `current_timestamp_ns`/controller-without-lease API 已被替换，且 atomic apply 新增 `not_before_timestamp_ns`，接入方必须迁移到上述 controller contract。v1 客户端缺少 `chunk_sequence_id` 和 controller recovery handshake，与 v2 不兼容，必须同步升级并重新握手。

## 8. 依赖与当前边界

根包以 Python 3.11 为基线。`torch==2.7.1` 位于可选 `pytorch` extra，与 vendored OpenPI 精确版本一致，避免同时安装不兼容的 PyTorch；PyTorch 使用 BSD-3-Clause 许可证，其 CPU/CUDA/MPS wheel 和 GPU backend 可用性由部署平台决定。核心动作、schema 与 normalization 模块仍只依赖 NumPy。根包的 `pytorch` extra 不是完整 OpenPI 集成环境；训练和 policy 示例必须在 `openpi/` 的锁定环境中以 editable 方式安装根包。Sharpa data factory 还必须显式配置不含 `/`、`..` 或路径分隔符的单层 `assets.asset_id`，不能依赖可能含组织前缀的 `repo_id` 回退值。

当前没有实现训练 CLI、DDP lifecycle、完整 checkpoint manager、Sharpa HDF5 observation dataset、可信 FK、控制 SDK、急停或真实硬件测试。同步 inference API 本身也没有 deadline 参数；stock WebSocket 还缺少服务端认证/TLS、消息大小上限和安全错误封装。生产 transport 必须提供有界超时，并由 controller 独立 watchdog 在调用线程阻塞或连接半开时进入 hold。根依赖锁文件也必须在获准建立环境后由 `uv lock` 生成，不能手工伪造。需要在服务器执行的验证步骤和结果记录模板见 [server-validation.md](server-validation.md)。
