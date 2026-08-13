# PI-DEX

PI-DEX 是面向 Sharpa North 双手灵巧操作的研究代码，基于 OpenPI 的 π0.5（`pi05`）。当前首个开发里程碑聚焦 PyTorch 训练和部署的动作边界，尚未宣称具备真实机器人端到端能力。

当前版本为 `0.1.0` 研究预览；`pi_dex` 公共 Python API 均为实验性接口。动作数值布局、metadata schema、checkpoint 和 wire 语义通过独立版本及严格字段校验拒绝不兼容输入，但数据集 factory、FK provider 和 controller lease 协议预期在 Sharpa SDK 与真实数据接入时替换或收窄；升级时必须按文档迁移，不应依赖未记录的实现细节。

当前已经实现：

- 单手 `31D ↔ 32D` 唯一编解码，以及 `L, R` 双手交错/解交错；
- 显式的 horizon、时间基准、坐标系、动作模式、rotation-6D、手臂/手部关节顺序与镜像映射、标定版本和延迟契约；
- 基于各 HDF5 group 自身 `aligned_index` 的 30/60 Hz 动作窗口选择；
- 必须注入标定 FK provider 的 `7D arm + 22D hand → 31D` 派生边界；
- 31 个有效维度上的逐手/共享归一化统计，以及 normalization asset 指纹；
- 保持 OpenPI 原始 `PI0Pytorch` 与 checkpoint shape 不变的 padding-neutral 训练核心；
- checkpoint 对 action、OpenPI 模型/tokenizer 配置、真实权重文件与 normalization asset 的指纹绑定，以及 PyTorch policy 加载；其中 tokenizer 仅绑定配置，实际 model 文件字节仍属下述外部边界；
- 只发布已反归一化物理动作的服务 adapter；
- 带 `peek/commit` 确认、客户端 observation 快照、chunk 序列/控制周期、可信 controller 时钟、唯一 lease、抗前跳/回拨的原子时间窗、不可变限幅，以及 recovery epoch 故障锁定的双手 dispatch 协议。

当前明确未实现：

- Sharpa North 的 URDF/MJCF、FK 和机器人标定；
- 完整 HDF5 图像/触觉 observation dataset；
- 可直接启动的训练 CLI、DDP loader/lifecycle 和完整 checkpoint manager；
- Sharpa 控制 SDK 适配、真实硬件原子下发、急停与 safe-hold 实现；
- 快系统和外部触觉编码器接入。

部署还有几项必须由外部基础设施闭合的边界：PaliGemma tokenizer 的实际文件字节尚未随
checkpoint 固化；同步推理调用必须由 transport deadline 和 controller watchdog 提供
有界失败；vendored OpenPI WebSocket server 本身没有认证/TLS、消息大小限制，还会把
traceback 返回客户端，因此只能用于受控环回开发，不能直接作为生产入口。服务器/GPU/
OpenPI/WebSocket/硬件验证清单见
[docs/server-validation.md](docs/server-validation.md)。

这些缺口不会使用 `state/*/tcp_pose`、假定 `2*k` 对齐、伪造 FK、跳过反归一化或顺序写左右手来掩盖。PyTorch 集成说明见 [docs/pytorch.md](docs/pytorch.md)。项目约束以 [AGENT.md](AGENT.md) 为准。

当前 action `layout_version=2`；它在保持 31D/32D 数值宽度的同时加入了左右手关节列语义，旧 v1 metadata 必须显式迁移，不能直接兼容加载。
