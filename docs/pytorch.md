# PyTorch 训练与部署契约

本文记录 PI-DEX 当前可用的 PyTorch/OpenPI 集成边界。代码已经覆盖双动作表示、训练
loss、归一化资产、checkpoint 契约、安全 dispatch 和一个外部 training runner 的 CLI
接缝；first-party Sharpa HDF5 dataset/完整 runner、可信 FK 实现与控制 SDK 仍需后续接入。

## 1. 统一动作规格

训练、checkpoint、策略服务和机器人进程必须共享同一个 `BimanualActionSpec`。下面的标识仅用于展示接口；接入真实数据前必须替换为经过验证的机器人、标定、时钟和控制语义。

```python
from pi_dex.actions import ActionRepresentation
from pi_dex.spec import ActionMode
from pi_dex.spec import ActionTimebase
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization
from pi_dex.spec import Rotation6DConvention

action_representation = ActionRepresentation.CARTESIAN_31D
uses_cartesian_actions = action_representation is ActionRepresentation.CARTESIAN_31D

spec = BimanualActionSpec(
    physical_horizon=8,
    timebase=ActionTimebase.RAW_CONTROL_60_HZ,
    control_frequency_hz=59.4,
    robot_id="<validated-station-id>",
    embodiment_version="<validated-embodiment-version>",
    coordinate_frame=("<validated-base-frame>" if uses_cartesian_actions else None),
    action_mode=ActionMode.ABSOLUTE,
    action_representation=action_representation,
    hand_normalization=HandNormalization.PER_HAND,
    rotation_6d_convention=(
        Rotation6DConvention.MATRIX_FIRST_TWO_COLUMNS_COLUMN_MAJOR
        if uses_cartesian_actions
        else None
    ),
    kinematics_calibration_version=(
        "<validated-calibration-version>" if uses_cartesian_actions else None
    ),
    command_semantics_version="<validated-absolute-command-version>",
    left_arm_joint_order=tuple(f"left_arm_j{i}" for i in range(7)),
    right_arm_joint_order=tuple(f"right_arm_j{i}" for i in range(7)),
    left_hand_joint_order=tuple(f"left_hand_j{i}" for i in range(22)),
    right_hand_joint_order=tuple(f"right_hand_j{i}" for i in range(22)),
    hand_mapping_version="<validated-hand-column-and-mirror-version>",
    left_wrist_link=("<validated-left-wrist-link>" if uses_cartesian_actions else None),
    right_wrist_link=("<validated-right-wrist-link>" if uses_cartesian_actions else None),
    clock_domain="<shared-runtime-clock-domain>",
    max_group_timestamp_skew_ms=2.0,
    max_alignment_timestamp_error_ms=2.0,
    max_control_period_error_ms=8.0,  # unit-test / Cartesian default; joint_29d site contract uses 20.0
    max_observation_age_ms=50.0,
    max_command_lead_ms=25.0,
)
```

该规格表示 `K=8` 个物理双手控制步。`action_representation` 必须二选一：

- `CARTESIAN_31D`：`wrist position(3) + rotation-6D(6) + hand joints(22)`，
  需要可信 FK，补 1 个零到 32D；
- `JOINT_29D`：`arm joints(7) + hand joints(22)`，直接保留 commanded joints，
  禁止使用 FK，补 3 个零到 32D。

当前 `sharpa_data` 派生路径只接受 `ActionMode.ABSOLUTE`。它不会从 absolute commanded
joint positions 自动构造 delta/residual target；选择其他 action mode 会立即失败。若后续需要
delta/residual，必须先定义 reference state、时间对齐、归一化和推理积分/合成语义，再单独实现和
验收，不能只修改 metadata 枚举值。

构造 Joint spec 时，`coordinate_frame`、`rotation_6d_convention`、
`kinematics_calibration_version` 和左右 `wrist_link` 必须显式为 `None`；metadata 也固定记录
JSON `null`。这些字段不会用占位字符串冒充 Joint 训练的 FK/Cartesian provenance。

两种模式的 OpenPI shape 都是 `[B, 2K, 32] = [B, 16, 32]`，序列严格按
`L0,R0,L1,R1,...` 排列。有效宽度 `D=spec.logical_action_dim`，padding mask 由 spec
派生，不允许训练、归一化或部署代码自行硬编码 31。

metadata 中的 `layout_version=3` 与 `metadata_schema_version=3` 绑定
`action_representation`、动态 logical width、关节列顺序和 `hand_mapping_version`。
checkpoint 的 `pi_dex` mapping 不接受未知扩展；部署握手只额外允许并验证 v3
`wire_format`、`execution_horizon` 与每次服务启动唯一的 `session_id`。旧 v2
metadata/checkpoint/wire 不含完整的表示选择，必须双向 fail closed；迁移时必须离线确认
原 checkpoint 是 Cartesian v2、重新生成 v3 sidecar/normalization provenance 并完成验证，
不能仅手改版本号，更不能把 v2 checkpoint 猜成 joint 模式。

## 2. Sharpa 数据与 FK 边界

`data/schema.json` 提供每侧 7D 手臂和 22D 手部 commanded joint angles。数据适配器
必须构造四个 `CommandedJointGroup`、canonical `AlignedTimeline` 和 episode provenance，
然后严格按 spec 选择路径：

- Cartesian 31D：仓库没有可信机器人描述或标定，调用方必须注入实现
  `ForwardKinematicsProvider` 的标定 FK；
- Joint 29D：`kinematics` 必须为 `None`，数据边界直接拼接 7D arm 与 22D hand，
  不得为了接口统一而创建或调用 FK。

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
    kinematics=(validated_fk_provider if spec.requires_forward_kinematics else None),
)
```

两条路径都会验证 canonical HDF5 path、各 group 的 `N`、时间戳、采样周期、
robot/embodiment、手臂与手部关节顺序和左右手列映射/镜像版本；只有 Cartesian 路径还会
验证 calibration、wrist link、FK metadata/output 和 rotation-6D 正交性。构造
`CommandedJointGroup` 时必须传入数据源的实际 `joint_order`；
`EpisodeActionProvenance.hand_mapping_version` 必须来自可追溯的转换/录制配置，不能根据
22 列数值猜测。每个 group 都使用自己的 `aligned_index`；实现不会用
`2 * frame_index`，也不会回退到 measured `state/<side>_arm/tcp_pose`。

供训练的随机访问 dataset 在 repack/data transforms 后必须为每个样本产生：

```text
state         floating [state_dim]
left_actions  floating [K, D]  # D=31 Cartesian 或 D=29 Joint
right_actions floating [K, D]
```

图像、prompt、state 和 HDF5 I/O 的具体映射仍由后续 Sharpa observation dataset 负责。这里的
`state` 是一个未 batch 的一维向量；`[T,D]` 等额外时间轴不能静默展平，因为 π0.5 会把 state
编码进离散 prompt。实现 dataset 前必须把 state 的逐列来源/顺序/单位、prompt/annotation
规则、episode 级 split 和三个模型相机槽写入版本化配置。候选相机映射也必须由数据负责人确认，
不得因名称相似自动决定如何处理两个 head stereo view。

OpenPI 模型边界应收到三个固定 key：`base_0_rgb`、`left_wrist_0_rgb`、
`right_wrist_0_rgb`，以及同 key 的 bool masks。Dataset/transform 必须在进入 OpenPI 前把图像
明确转成 RGB、固定 layout、受支持 dtype/range 和声明的 resize/crop 结果；不能依赖上游
PyTorch preprocessing 通过 `shape[1] == 3` 猜测 BCHW/BHWC。缺失视角必须用显式 padding
图像且 mask 为 false，不能用黑图并标成有效视角。

HDF5 首个训练闭环必须设置 `num_workers=0`。OpenPI 在多 worker 模式使用 spawn 和
persistent workers；只有 first-party dataset 实现每个 worker 独立打开/关闭 HDF5、确定性
worker seed、异常清理并通过测试后，才能提高该值。不得让父进程 h5py handle 被 worker 继承。

## 3. 归一化与 OpenPI loader

归一化统计必须在 padding 和交错之前计算。π0.5 路径严格要求 `state`、
`left_actions`、`right_actions` 三个统计项，每项都有一维 `mean/std/q01/q99`；两侧动作
统计 shape 必须是 `[spec.logical_action_dim]`，即 Cartesian `[31]` 或 Joint `[29]`。

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

当前 loader 支持单进程训练，以及在 ``torch.distributed`` 已初始化时使用
``DistributedSampler``（``shuffle`` 必须为 ``False``）。``--batch-size`` 始终是
**per-rank local batch**。火山引擎多机请用 ``pi-dex-volc-train`` /
``scripts/volc_ddp_train.sh``。``data/repack`` output transforms 必须为空；model output
在首次配置前也必须为空，重复配置时则只允许已存在唯一的 ``UnpackBimanualActions``。这样
PI-DEX 会在 inverse normalization 前把 ``actions`` 解成两侧字段，且不会和其他 model
output transform 重排语义。

变换顺序固定为：

```text
repack/data inputs
→ Validate state[state_dim] and optional left/right targets[K,D]
→ Normalize(state, left_actions, right_actions)
→ prompt/image/state model inputs
→ each hand D→32D (Cartesian 补1维；Joint 补3维)
→ interleave L/R as [2K,32]
```

推理按逆序先解交错、丢弃所选表示的整个 padding 后缀，再分别反归一化两手。部署入口
还会把 `action_in_proj` 中所有 padding 输入列，以及 `action_out_proj` 中所有 padding
输出行/偏置确定性置零，使 stock OpenPI denoising 的随机 padding 不再反馈污染有效维；
该策略写入 checkpoint 的 `padding_inference_policy` 契约。应始终通过
`create_bimanual_trained_policy` 加载，不要绕过该投影直接发布原始 OpenPI policy。

## 4. `pi05_base` 初始化与 PyTorch 训练

### 4.1 初始化 checkpoint 决策

Cartesian 31D 和 Joint 29D 都默认从官方
`gs://openpi-assets/checkpoints/pi05_base` 开始。该 checkpoint 是新 embodiment 的通用
fine-tuning 起点；`pi05_droid` 只允许作为受控对照，`pi05_libero` 不用于 Sharpa 初始化。
不得因为 action projection 都是 32D 就假定 expert checkpoint 的动作列、state、归一化或时序
语义与 Sharpa 一致。

官方发布的是 JAX/Orbax checkpoint。PyTorch runner 不能直接使用 JAX 的
`CheckpointWeightLoader(.../params)`；必须先转换出含 `model.safetensors` 的只读初始化目录，
再由 runner 从该目录实际加载。路径语义不同：

- converter 的 `--checkpoint_dir` 接收 checkpoint 根目录
  `.../pi05_base`，内部自行读取 `.../pi05_base/params/`；
- JAX `CheckpointWeightLoader` 才接收 `.../pi05_base/params`；
- PyTorch `pytorch_weight_path` 接收 converted 输出目录，trainer 从其中读取
  `model.safetensors`。

两种表示只有在完整 OpenPI model config 指纹一致时才能共享一份 converted base。真实训练
配置必须通过 `create_pi05_model_config(spec)` 固定 `pi05=True`、`action_dim=32`、
`action_horizon=2*K`、`discrete_state_input=True`、Gemma variants、`max_token_len` 和 dtype。
上游 registry 当前没有名为 `pi05_base` 的可直接转换 config，仓库也还没有注册 first-party
PI-DEX `TrainConfig`。服务器 agent 必须提供经审阅、纳入版本控制的 experiment config 和
converter wrapper：wrapper 从配置解析 spec，调用 `create_pi05_model_config(spec)` 后直接调用或
修补转换函数；`joint_29d`/`cartesian_31d` 只是 representation 值，不得冒充
`--config_name`。不能无说明借用 LIBERO/DROID 配置。

服务器端操作轮廓如下，所有路径和制品必须先登记；完整验收见
[server-validation.md](server-validation.md) 的阶段 A.2：

```bash
cd "$PI_DEX_REPO/openpi"
export OPENPI_DATA_HOME=/srv/pi-dex-cache/openpi
export PI_DEX_PI05_BASE_ACQUIRED="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base"
export PI_DEX_PI05_BASE_JAX=/srv/pi-dex-artifacts/pi05_base-source/REPLACE_WITH_MANIFEST_HASH
export PI_DEX_PI05_BASE_PT=/srv/pi-dex-artifacts/pi05_base-pytorch-bfloat16

# 下载到显式 cache；保存输出路径和源逐文件 manifest/hash。
uv run --locked python -c \
  'from openpi.shared.download import maybe_download; print(maybe_download("gs://openpi-assets/checkpoints/pi05_base"))'
test -d "$PI_DEX_PI05_BASE_ACQUIRED/params"
# 由受控 acquisition/publish 入口验真并原子发布 source snapshot 后再继续。
test -d "$PI_DEX_PI05_BASE_JAX/params"
test ! -e "$PI_DEX_PI05_BASE_PT"

# 调用版本控制中的 PI-DEX wrapper；具体入口由服务器实现并写入实验记录。
uv run --locked python -m pi_dex_server.convert_pi05 \
  --checkpoint-root "$PI_DEX_PI05_BASE_JAX" \
  --experiment-config /srv/pi-dex-site/REPLACE_WITH_EXPERIMENT_CONFIG \
  --output "$PI_DEX_PI05_BASE_PT"
```

上述 wrapper module 名只是待实现接口轮廓，不是当前仓库已有入口或独立 `PASS` 证据。当前
vendored converter：

- 根据 checkpoint 路径字符串是否包含 `pi05` 选择 AdaRMS 映射；
- 使用 `load_state_dict(..., strict=False)` 却不验证 missing/unexpected keys，可能把随机初始化
  参数保存进看似完整的 safetensors；
- 从 `checkpoint_dir.parent/assets` 尝试复制 assets，而非建立 Sharpa stats；
- 只写简化 `config.json`，没有完整记录 pi05、discrete-state/tokenizer 等契约；
- 不清空已存在的 output 目录。
- PaliGemma 结构与 action expert mapper 分别硬编码为 `gemma_2b`/`gemma_300m`；未实现通用 mapper
  前必须拒绝其他 variants。

因此发布级转换必须使用经审阅的 wrapper/补丁：按 model config 选择 π0.5 分支；捕获并严格
审查所有 missing/unexpected keys；目标目录必须不存在；记录源/工具/config/输出哈希；最后用
固定 post-transform inputs、actions、noise 和 flow timestep 做 JAX↔PyTorch vector-field
parity。容差在执行前登记，不能在看到结果后放宽。转换后即使 strict load 成功，也不能反向
证明 converter 没把随机初始化参数保存进去，所以源参数覆盖率检查和 parity 都不可省略。
Wrapper 必须输出到全新 staging 目录，在完整逐文件 manifest、provenance、strict load 和 parity
通过后原子发布到不存在的目标；失败清理 staging，绝不覆盖已发布制品。

Converted base 只是训练初始化制品，不包含 Sharpa 的 `[31]`/`[29]` stats 或
`pi_dex.json`，不能 resume、serve 或直接部署。两个模式必须分别在自身训练数据上计算 stats，
完成训练后再发布完整 PI-DEX checkpoint。

### 4.2 PyTorch 模型与训练 step

`action_dim` 必须保持 32，才能加载 π0.5 的 action projection 权重。下面是外部 runner
内部应采用的训练骨架；launcher 本身不会构造这些对象。`load_verified_pi05_base` 是待服务器
实现的 first-party 严格加载入口，不是当前仓库已有函数：

```python
import jax
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from pi_dex.pytorch_training import PiDexPytorchTrainer

device = torch.device(device_name_from_runtime_config)
model = PI0Pytorch(train_config.model).to(device)
load_verified_pi05_base(
    model,
    converted_base_dir,
    expected_weights_sha256=expected_base_sha256,
)
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

训练核心会把 target 与 flow-matching noise 的全部 padding 后缀置零，并只在动态的 31 或
29 个有效维上求平均。trainer 要求每个可训练参数恰好属于一个 optimizer parameter group，
并拒绝遗漏、重复或不属于 model 的参数。loss 在 backward 前必须为有限标量；backward 后
无论是否启用裁剪都会检查总 gradient norm，非有限值不会进入 `optimizer.step()`。

不得直接把 `openpi/scripts/train_pytorch.py` 当成 PI-DEX runner 或在外面只包一层 launcher。
stock 脚本对 `[B,2K,32]` 的 elementwise loss 直接调用 `.mean()`，也不会执行 PI-DEX 的
padding noise 策略，因此与 31D/29D 训练契约不兼容。First-party runner 必须调用
`PiDexPytorchTrainer`（或经过同等审阅和测试的语义等价实现）。

当前 vendored PyTorch 路径只支持 full bfloat16 或 full float32 训练，不支持 LoRA、FSDP、
EMA 或 mixed precision/AMP。PI-DEX 已支持 torchrun / 火山 MLP 多机 DDP（local batch ×
world_size = global batch；rank-0 写 checkpoint）。上游对 full fine-tuning 的粗略估算是
单卡显存超过 70 GB（A100 80GB/H100 级别），但正式资源门槛必须用 PI-DEX 的实际 `2*K`、
图像/state/prompt、batch、optimizer 和 compile 设置做完整 step probe 后决定。普通 DDP
复制模型，不能替代 FSDP 降低单卡显存。

`TrainConfig.pytorch_training_precision` 与 `train_config.model.dtype` 是两个独立字段；
first-party config builder/runner 必须在模型构造前要求二者逐字相等，并验证实际参数 dtype
policy。部署训练固定二者为 `bfloat16`，float32 诊断固定二者为 `float32`。不得沿用 stock
trainer 在启动后改写 `model.dtype` 的行为，也不得让 sidecar 记录的 model dtype 与实际训练
precision 分离。

### 4.3 表示可切换的 launcher 接缝

仓库安装后提供 `pi-dex-train-pytorch`，但它不是 first-party 训练实现。它只解析表示、
校验 Cartesian 必须有 FK factory/Joint 禁止 FK，然后把
`PytorchTrainingLaunchContext` 交给显式的 `module:callable` runner。runner 自己负责
HDF5 dataset、OpenPI config、模型/optimizer、训练循环、checkpoint/resume 和日志。

```bash
# Joint 29D：不得传 --fk-provider-factory
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner your_project.training:run \
  -- --config /server/path/joint.yaml --steps 1000

# Cartesian 31D：必须传零参数 FK provider factory
pi-dex-train-pytorch \
  --action-representation cartesian_31d \
  --runner your_project.training:run \
  --fk-provider-factory your_project.kinematics:create_fk \
  -- --config /server/path/cartesian.yaml --steps 1000
```

`--` 是 launcher 参数和 runner 参数的边界；其后的字符串不解析、不改写地放入
`context.runner_args`。runner 必须把实际供 dataset、OpenPI transforms、PyTorch trainer、
checkpoint 和部署共同使用的 spec 传给 `spec = context.bind_action_spec(spec)`，并只继续使用
返回的验证副本。Cartesian runner 随后必须调用 `context.create_kinematics()` 并让数据边界
验证 provider metadata；Joint runner 不得调用该方法。返回 `None` 或 `0` 前未绑定 spec，
或者 Cartesian 未通过 context 获取 FK，launcher 都会拒绝成功；非零 runner 退出码仍原样
返回，以保留配置建立前的失败。除普通返回 `None` 视为 0 外，runner 的普通返回 code 与
`SystemExit` code 都只允许 shell 可稳定表达的 `0..255`；`SystemExit(None/0)` 同样必须先
完成成功握手，禁止用进程退出绕过校验。
上述名称和路径仅为接口
示意，仓库目前没有 `your_project` 或任何 first-party runner，因此这些命令不能原样执行，
也不能据此宣称训练已端到端打通。

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

上述 helper 只写入和验证 PI-DEX sidecar，不是完整 checkpoint manager。First-party runner 还
必须原子保存并恢复：optimizer、scheduler 或可精确重建的 schedule、global step、
Python/NumPy/Torch CPU/每张 CUDA 设备的 RNG、data shuffle/sampler 状态或精确 next-sample
cursor、完整训练配置、dataset/split manifest、父 `pi05_base` source/converted hashes、
tokenizer hash 以及失败清理/保留策略。仅恢复 `model.safetensors` 不得描述为可复现 resume。

Resume 至少要比较一次连续 `N+M` step 与 `N` step 保存、全新进程恢复后再训练 `M` step 的
受控结果，预先规定学习率、下一批样本、RNG、loss 和最终参数允许的差异。若 worker、compile
或非确定性 kernel 使逐位比较不成立，必须在运行前声明可解释的容差和证据，不能事后放宽。

当前契约绑定了 `max_token_len`、Gemma variants 与离散 state 模式，但 vendored OpenPI
仍从独立 GCS/cache 读取 PaliGemma tokenizer model，实际 tokenizer 文件字节尚未随
checkpoint 固化。服务器必须先在隔离、可写的 acquisition cache 下载并验真，再原子发布为
hash-pinned immutable snapshot；`maybe_download()` 会创建 lock/目录、调整权限并可能刷新条目，
不能把它直接指向典型只读挂载。First-party data/policy config 必须提供 local-path tokenizer
resolver，使运行时直接读取 snapshot 中已验 SHA-256 的文件且不写 cache；在该入口及 tokenizer
资产纳入 sidecar 前，不能把 checkpoint 描述为完全自包含。

## 6. 策略服务

### 6.1 PI-DEX model server（推荐）

机器人侧保留 Zenoh/SDK；GPU 机只跑推理服务。首个入口：

```bash
# conda activate pi-dex 后
pi-dex-serve \
  --checkpoint-dir /path/to/run/10000 \
  --observation-contract configs/site/joint_29d_observation.reviewed.json \
  --assets-dir /path/to/assets-Insert_Battery \
  --asset-id sharpa_joint_29d_insert_battery \
  --robot-id POC22005 \
  --host 127.0.0.1 \
  --port 8000

# 或
bash scripts/serve_joint29d.sh --checkpoint-dir /path/to/run/10000
```

协议：与 OpenPI 相同的 WebSocket + msgpack-numpy。连接后先收 metadata，再循环
`observation → action`。Observation 必须含 OpenPI 字段，并附加：

```text
observation_timestamp_ns
clock_domain
```

相对 stock OpenPI server 的默认加固：默认只绑 `127.0.0.1`、有
`--max-message-bytes`、失败时不回传 traceback、可选 `--api-key` /
`--infer-timeout-s`。跨网或上真机前仍须按 `server-validation.md` 阶段 E 补 TLS/
认证评审与独立 controller watchdog（机侧）。

环回探针（server 已启动时）：

```bash
pi-dex-serve-probe --host 127.0.0.1 --port 8000 --api-key "$API_KEY"
```

### 6.1.1 从端推理桥（Zenoh ↔ serve）

机器人侧已有 Sharpa 启动脚本（NUC `start.sh` / `start-nuc.sh` + Orin
`start-remote-orin.sh`），pendant **F6** 切遥操作↔推理、**F2** 走
init→standby↔moving。PI-DEX **不替代**这些进程；缺的是推理模式下向
`inference/action` 发布 `UhrActionBundle` 的策略桥。

推荐拓扑：

| 机器 | 进程 |
|------|------|
| 从端 NUC (+ Orin) | `bash start.sh`（或拆开的 nuc/orin 脚本） |
| GPU 机 | `pi-dex-serve` / `scripts/serve_joint29d.sh` |
| 从端 NUC（同 Zenoh 域） | `pi-dex-robot-client` / `scripts/robot_client_joint29d.sh` |

```bash
# 离线确认 protobuf→SDK→OpenPI 观测（无需 Zenoh / GPU）
pi-dex-robot-client --mode codec-smoke

# 真机：先拉起机器人栈与 model server，再起桥
bash scripts/robot_client_joint29d.sh \
  --serve-host <GPU_IP> \
  --serve-port 8000 \
  --prompt "insert the battery"

# pendant: F6 → 推理模式，F2 → moving
```

默认 topic 与参考 SDK（`examples/sharpa_north_sdk.py`）一致：
`north_observation` → `inference/action`。编解码见 `pi_dex.north_codec`
（schema：`examples/north.proto`，生成码：`pi_dex.north_pb2`）。NUC 需安装
`eclipse-zenoh`；lease / e-stop / watchdog 仍属后续 `BimanualController`。

推理机 / 从端环境安装与联调步骤见专文 [inference-env.md](inference-env.md)。

### 6.2 Stock OpenPI server（仅环回诊断）

服务端 adapter 只接受 OpenPI 已经完成 inverse normalization 的
`left_actions/right_actions[K,D]`，其中 `D` 必须与 checkpoint/spec 的 31D Cartesian 或
29D Joint 表示一致。它不会把 raw `[2K,32]` 模型输出冒充物理命令。

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
actions.left       float32[E,D]  # D=spec.logical_action_dim
actions.right      float32[E,D]
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

每个缓存步都保留 source timestamp、clock domain、独立的 `chunk_sequence_id` 和
`chunk_step_index`。dispatcher 绑定服务端 metadata 验证出的 execution horizon 和动作表示，
并验证 chunk 身份、严格连续的 chunk ID、连续且不越界的 step、同 chunk source、一致的跨
chunk 控制周期及递增 target。两种模式都执行动态 shape/dtype/finite/逐维限幅；Cartesian
还验证 rotation-6D 非退化，Joint 则不会把 arm joint 槽误当旋转。任一失败都会请求经
controller 状态确认的 `hold` 并永久锁存该 dispatcher 实例。

具体 Sharpa controller 必须公开与 dispatcher 完全相等且运行期间不可变的 `action_spec`、`safety_faulted` 与单调 `recovery_epoch`，并在后端原子维护每个 recovery epoch 唯一的 dispatch lease。`acquire_dispatch_lease()` 必须绑定 spec、clock domain 和 epoch；若抛错或返回无效 token，不得遗留 active lease。`read_clock_ns()` 从该 lease 的可信 controller 时钟读数，调用方不再提供可用于准入判断的 current time。dispatcher 在获得自身锁后及 apply 前各重读一次时钟，并把第二次读数作为 `not_before_timestamp_ns`；controller 必须在其内部同一临界区验证 lease、epoch 和 `not_before_timestamp_ns <= now <= not_after_timestamp_ns < target_timestamp_ns`，然后才原子下发双手（或针对同一 target 先 stage 两侧再统一 commit）。上下界同时关闭 read→apply 期间的时钟前跳与回拨窗口。

`hold()` 必须原子锁存安全状态、使当前 lease 失效，并返回 `BimanualHoldReceipt(safety_faulted=True, recovery_epoch=<same epoch>)`。外部硬件安全恢复必须先原子进入已验证安全态并撤销旧 lease，成功后才可清除 fault、递增 epoch 并允许新 lease；随后调用方必须 reset broker 并重建 dispatcher。若旧 dispatcher 发现 epoch 已变化，它的 stale token 不得再 hold 新 epoch，因而只能本地锁存并报告 hold acknowledgement 失败；此时硬件安全性必须由已经完成的 recovery 事务保证。仓库尚无控制 SDK，不能声称已经实现硬件同步、急停或 safe-hold。当前 WebSocket reset 不会传播到服务端；未来有状态 policy 需新增 reset RPC 或在客户端显式管理。

wire v3 假定一个 policy adapter 实例服务一个机器人控制 session，并把动作表示及动态宽度
绑定进完整 metadata。adapter 或 server process 重启后，chunk ID 会重新起始；任何连接
丢失都必须视为 session 失效。恢复流程是重新获取并校验 v3 metadata、完成 controller 的
硬件安全恢复，再新建 broker 与 dispatcher，不能在旧对象上自动续跑。v2 wire 不含完整
双表示契约，必须 fail closed；不能根据 response 宽度猜测表示或只改版本字符串。

## 8. 依赖与当前边界

根包以 Python 3.11 为基线。`torch==2.7.1` 位于可选 `pytorch` extra，与 vendored OpenPI 精确版本一致，避免同时安装不兼容的 PyTorch；PyTorch 使用 BSD-3-Clause 许可证，其 CPU/CUDA/MPS wheel 和 GPU backend 可用性由部署平台决定。核心动作、schema 与 normalization 模块仍只依赖 NumPy。根包的 `pytorch` extra 不是完整 OpenPI 集成环境；训练和 policy 示例必须在 `openpi/` 的锁定环境中以 editable 方式安装根包。Sharpa data factory 还必须显式配置不含 `/`、`..` 或路径分隔符的单层 `assets.asset_id`，不能依赖可能含组织前缀的 `repo_id` 回退值。

当前实现的是训练 launcher seam，不是 first-party training runner；完整 checkpoint manager、
Sharpa HDF5 observation dataset、可信 FK 实现、DDP lifecycle、控制 SDK、急停和真实硬件测试
仍未完成。同步 inference API 本身也没有 deadline 参数；stock WebSocket 还缺少服务端
认证/TLS、消息大小上限和安全错误封装。需要在服务器执行的验证步骤和双模式结果矩阵见
[server-validation.md](server-validation.md)，推荐开发顺序见
[server-handoff.md](server-handoff.md)。
