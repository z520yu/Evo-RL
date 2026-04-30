# Embodiment Mismatch / Negotiation Tokens

## 结论先行

这个想法最有“几句话讲清楚”的宏观吸引力，但在当前单台 Piper 真机上不适合作为唯一主线。

强版本：

> 一个 token 自动理解机器人 embodiment mismatch，并让 VLA 泛化到新机器人。

这个版本太大，当前设备和数据很难证明。

更严肃的版本：

> 从 command-execution telemetry、视觉变化和少量 robot metadata 中学习一个结构化 latent，表示当前机器人相对 VLA action prior 的可观测执行 mismatch，并在 action generation 或 residual control 前条件化策略。

它真正有价值的地方是把 VLA 适配从：

```text
动作生成后再翻译 / 修正
```

推进到：

```text
动作生成前让模型知道当前身体能合理执行什么
```

## 核心问题

通用 VLA 迁移到不同机器人时，失败不只是：

- 图像 domain gap；
- action normalization；
- 坐标系转换；
- action adapter 没写好。

更深的问题是：

> VLA 内部对“身体能做什么、怎么做、做到什么精度”的假设，与真实机器人不一致。

例如同一句指令“把杯子拿起来”，不同机器人代表不同约束：

- 可达空间不同；
- 夹爪宽度和力不同；
- 相机视角不同；
- 控制频率不同；
- action scale 不同；
- 延迟不同；
- 位置控制/速度控制/关节控制接口不同；
- 接触阶段的 compliance 不同。

Negotiation Tokens 试图显式表达这些差异，让 VLA 在生成 chunk 前获得身体上下文。

## 一句话定义

**Negotiation Token** 是一个 task-conditioned embodiment/execution latent。它由 robot metadata、few-shot demonstrations、recent command-execution history、visual/proprio changes 编码而来，用来条件化 VLA action chunk generation 或 residual actor，使动作语义和当前身体能力对齐。

英文式表述：

> Negotiation Tokens turn embodiment adaptation from post-hoc action translation into pre-generation policy conditioning.

## Token 应该表示什么

不要把它定义成模糊的“一个 latent”。最好拆成三层。

### 1. Static embodiment token

episode 内基本不变：

- 机器人类型；
- DoF；
- joint limit；
- workspace；
- gripper 类型和宽度；
- 相机布局；
- control frequency；
- action dimension；
- absolute / delta action；
- action scale；
- 标称延迟。

这层像“身体配置摘要”。

### 2. Dynamic execution token

最近一段时间内的执行偏差：

- commanded action vs realized proprio delta；
- VLA chunk vs sent action；
- sent action vs q_obs response；
- 延迟；
- overshoot / undershoot；
- saturation；
- chunk boundary mismatch；
- gripper lag；
- 视觉末端/物体变化和命令不一致。

这层最适合当前 Piper，因为它可由底层日志和相机估计。

### 3. Task-conditioned negotiation token

当前任务如何调用身体能力：

- 是粗抓还是精细放置；
- 是否需要夹住后拉动；
- 是否需要低速接触；
- 哪些自由度最关键；
- 是否应该缩短 chunk horizon；
- 是否应该更依赖视觉闭环；
- 是否允许 residual 介入。

这层才是 negotiation，而不是 robot ID。

## 和 action adapter 的区别

Action adapter 解决动作格式：

> VLA 输出的 action 如何变成当前机器人能执行的 command。

例如：

- 7D EEF delta 转 joint command；
- normalized action 反归一化；
- gripper scalar 转夹爪位置；
- canonical action 转 native action。

Negotiation Token 解决动作生成前的条件化：

> VLA 在知道当前身体约束后，应该生成什么动作。

例如：

- 低刚度/低速机器人生成更短、更保守的 chunk；
- 夹爪响应慢则提前闭合；
- 控制延迟大则减少高速精细动作；
- workspace 边缘任务选择不同 approach；
- wrist camera 偏移大时更依赖外部视角。

一句话：

> Adapter 翻译动作语法；Negotiation Token 改变动作语义。

## 和 domain adaptation 的区别

Domain adaptation 通常试图让新 domain 看起来像训练 domain：

- 图像风格对齐；
- feature alignment；
- action normalization；
- camera view adaptation。

Embodiment mismatch 是 capability/dynamics shift：

- 同一视觉状态下，不同机器人可行动作集合不同；
- 同一 action vector 的物理效果不同；
- 同一夹爪闭合命令对不同 gripper 含义不同；
- 同一接触事件需要不同控制方式。

Negotiation Token 不应该消除差异，而应该保留并利用差异。

## 和 RLT / RTC / OpenPI 的关系

### OpenPI

OpenPI 是自然平台，因为它本身就是多机器人 VLA 迁移场景。Negotiation Token 可以作为 fine-tuning / deployment interface：

```text
base VLA
-> robot/task data
-> learn negotiation encoder
-> condition action chunk generation
```

它要证明的不是比 full fine-tuning 永远更强，而是：

- 低数据更高效；
- 比 robot ID 更能泛化；
- 比 action adapter 更能处理执行误差；
- 在 RTC 之上仍有增益。

### RLT

RLT 是：

```text
VLA representation -> RL actor-critic
```

Negotiation Token 是：

```text
embodiment/execution -> VLA/action generator
```

它们方向相反但互补：

- Negotiation Token 让 base action 更符合当前身体；
- RLT/residual 再优化高精度或低 trust 阶段。

### RTC

RTC 处理实时 chunk 执行和推理延迟。Negotiation Token 可以把 latency、control frequency、tracking lag 编进 token，让 VLA 生成更适合当前系统时序的 chunk。

实验矩阵：

```text
VLA
VLA + RTC
VLA + negotiation token
VLA + negotiation token + RTC
VLA + negotiation token + RTC + residual
```

## 当前 Piper 环境下底层可行性

当前 Piper 是位置控制为主，能稳定支撑的是“可观测执行 mismatch”，不是完整 embodiment 理解。

最可学习：

- action scale；
- zero offset；
- joint range clipping；
- velocity / acceleration saturation；
- command-execution delay；
- chunk execution lag；
- gripper position tracking error；
- chunk boundary discontinuity；
- speed_ratio 改变带来的响应变化。

部分可学：

- 末端相机坐标偏差；
- tool center point 偏差；
- 夹爪闭合太早/太晚；
- 抽屉/物体运动响应慢；
- 粗略接触失败模式。

基本不可学，除非加传感器或强视觉标签：

- 真实接触力；
- 摩擦系数；
- 夹爪真实夹持力；
- 物体是否被夹紧但视觉不可见；
- 力矩饱和真实原因；
- 策略语义错误 vs 物理执行错误。

因此当前设备上不要主张：

> token learns embodiment in general.

应该主张：

> token learns observable command-execution mismatch modes that condition chunk generation or residual authority.

## 结构化 token 设计

建议 token 不只是一个向量，而是带辅助头：

```text
z_emb = f(
    robot_metadata,
    A_vla,
    sent_action_history,
    q_obs_history,
    gripper_history,
    visual_delta_history,
    task_instruction
)
```

输出：

```text
z_emb
delay_hat
scale_hat
saturation_hat
gripper_lag_hat
tracking_error_hat
uncertainty_hat
recoverability_hat
```

这样可以避免被质疑为普通 adapter。

条件化位置：

1. input-side soft prompt；
2. action head conditioning；
3. residual actor input；
4. RTC / trust gate input；
5. LoRA/adapter bottleneck。

当前最现实的是：

```text
dynamic execution token -> trust/residual/RTC
```

而不是一开始改大模型 action generator。

## 最小实验路线

### 实验 1：无接触系统辨识

空载 Piper，执行：

- step；
- sine sweep；
- triangle wave；
- 不同速度比例；
- 不同 chunk length。

人为注入：

- command delay；
- action scale；
- offset；
- clipping；
- low-pass lag。

训练 token 预测 mismatch 类型和幅度。

指标：

- delay/scale/offset 回归误差；
- mismatch 分类准确率；
- 未见轨迹泛化；
- 未见 chunk length 泛化。

### 实验 2：夹爪台架

只控制 gripper：

- 空载开合；
- 不同宽度物体；
- 目标闭合速度变化；
- 夹住/未夹住视觉标签。

验证：

- gripper lag 可学；
- gripper cmd-pos mismatch 可学；
- 位置一致但未夹稳是否不可辨识。

后者即使失败也是有价值的负结果：说明低成本 telemetry 边界。

### 实验 3：任务闭环

任务：

- 夹住抽屉把手并拉开；
- 放入物体；
- 关闭抽屉。

对比：

- base VLA；
- base VLA + action adapter；
- base VLA + robot ID/static token；
- raw telemetry MLP residual；
- structured negotiation token；
- structured token + trust gate；
- structured token + RTC。

指标：

- 成功率；
- recovery 次数；
- intervention count；
- mismatch prediction accuracy；
- token swap sensitivity；
- injected delay/scale perturbation 下的鲁棒性。

## 如何证明它不是 robot ID

必须做诊断实验：

- token swap：同一 observation 换不同 token，action 是否按预期改变；
- counterfactual perturbation：注入 delay/scale，token 是否对应变化；
- interpolation：token 连续变化时 action 是否平滑变化；
- held-out composition：训练见过 delay 和 scale，但没见过 delay+scale 组合；
- auxiliary heads：token 是否能预测 physical/execution parameters；
- random token baseline；
- robot ID baseline。

如果没有这些诊断，它很容易被审稿人看成 learned prompt 或 domain embedding。

## 最大风险

1. **当前只有一台 Piper**
   跨 embodiment claim 很难成立。可以用人工注入 delay/scale/saturation 作为 pseudo-embodiment，但这只能支撑弱 claim。

2. **token 被模型忽略**
   如果只拼 token，不做 counterfactual loss 或 token dropout，大模型可能不用它。

3. **退化成 action adapter**
   如果输出只是 corrected action，没有结构化诊断，就和 adapter 没区别。

4. **任务失败原因混淆**
   视觉错误、策略错误、物理执行错误会混在一起。需要先在可控注入环境证明可辨识。

5. **对当前真机主线太大**
   如果目标是快速做出真机 paper，Negotiation Token 不如 Chunk Trust 直接。

## 可写成的论文 claim

强但可控：

> We identify observable command-execution mismatch as a distinct bottleneck in adapting chunked VLA policies to low-cost position-controlled robots, and introduce structured negotiation tokens that condition chunk generation and residual authority on estimated execution constraints.

更宏观：

> Negotiation Tokens turn embodiment adaptation from post-hoc action translation into pre-generation policy conditioning.

不建议写：

> We solve cross-embodiment VLA generalization.

## 推荐标题

```text
Negotiation Tokens for Cross-Embodiment Adaptation in Vision-Language-Action Models
```

当前设备弱化版：

```text
Execution Negotiation Tokens for Low-Cost Robot Adaptation of Chunked VLAs
```

## 最终判断

这个 idea 适合作为论文中的机制拓展或第二阶段方向，不适合作为当前单台 Piper 真机实验的唯一主线。

如果想把它放进当前项目，最合理方式是把它作为 Chunk Trust 的一个输入分支：

```text
execution mismatch token -> trust token / residual actor / RTC gate
```

这样它不会承担“跨机器人泛化”的过大 claim，而是承担更稳的角色：

> 当前机器人执行 VLA chunk 时出现了怎样的可观测 mismatch，这个 mismatch 是否应该降低 chunk trust 或改变 residual 权限。

## References

- OpenPI: https://github.com/Physical-Intelligence/openpi
- Open Sourcing pi0: https://www.pi.website/blog/openpi
- RL Tokens / RLT: https://www.pi.website/research/rlt
- LeRobot RTC docs: https://huggingface.co/docs/lerobot/rtc
