# 2026 论文补充后的方向重构：从人工 boundary labels 转向自动 contact-mode VLA+RL

更新时间：2026-04-28

## 0. 先修正上一版的问题

上一版提出的 `intervention / boundary / recoverability / correction` labels 不够好，原因很明确：

1. 人工划这么多标签没有现实意义，真机实验里会变成标注工程。
2. `failure prediction / correction / supervisor / gate` 这条线已经很挤，继续做“失败边界标签”容易撞题。
3. 代码实现不是主要瓶颈，真正瓶颈是真机任务、实验闭环、创新点能不能站住。

所以新方向必须满足：

- 不依赖人工多标签。
- 不把“接管发生了”当核心创新。
- 不把“失败预测/纠正”当泛泛主张。
- 任务必须针对 VLA 真实短板：**contact-rich last-centimeter precision**。
- 监督信号优先来自机器人自动日志，而不是人手工标注。

## 1. 2026 相关工作带来的压力

### 1.1 失败预测 / 纠正线已经很挤

- [FPC-VLA](https://arxiv.org/abs/2509.04018) 提出 VLA + failure prediction/correction supervisor，用 vision-language query 预测失败并生成纠正策略，还强调无需人工标注的大规模 failure-correction 数据生成。
- [CycleVLA](https://arxiv.org/abs/2601.02295) 做 proactive self-correction，在 critical subtask transition points 预测失败并 backtracking。
- [RACER](https://arxiv.org/abs/2409.14674) 用 rich language failure recovery 数据和 VLM supervisor 指导 actor。
- [Phoenix](https://arxiv.org/abs/2504.14588) 把高层 semantic reflection 转成 motion instruction，再用 diffusion policy 做细粒度动作纠正。
- [RaC](https://arxiv.org/abs/2509.07953) 用 human intervention 轨迹学习 recovery/correction behavior。
- [RoboFAC](https://arxiv.org/abs/2505.12224) 做机器人 failure analysis and correction 数据集与模型。

结论：

**不能再把“失败预测 + 纠正”作为主创新。**
如果继续做，必须非常具体，比如“接触阶段的 VLA 指导何时应该被信任/不信任”。

### 1.2 Human intervention / residual correction 也已经有人做

- [HIL-SERL](https://hil-serl.github.io/) 通过 demonstrations、reward classifier、人类 interventions 做真实机器人 RL。
- [RLIF](https://arxiv.org/abs/2311.12996) 把 human intervention signals 当作 RL feedback。
- [SiLRI](https://arxiv.org/abs/2512.24288) 专门处理 suboptimal interventions，指出盲目模仿人类接管会限制 RL 最终性能。
- [CR-DAgger](https://compliant-residual-dagger.github.io/) 做 compliant residual policy，从人类 delta correction 中学习 contact-rich residual，并展示 take-over correction 数据分布和稳定性问题。

结论：

**不能只说“从人类接管学 residual”。**
这个点已经被 HIL-SERL、RLIF、SiLRI、CR-DAgger 覆盖。接管只能作为数据来源或 baseline，不应是主贡献。

### 1.3 2026 contact-rich VLA 的主战场是 force/tactile

- [CRAFT](https://arxiv.org/abs/2602.12532) 指出 VLA 在 contact-rich 任务中受限于视觉/语言高熵输入和低熵但关键的 force signal 失衡，提出 force-aware curriculum fine-tuning。
- [ForceVLA2](https://arxiv.org/abs/2603.15169) 用 hybrid force-position control 和 force-aware prompts/MoE，处理 pressing、assembling、wiping 等任务。
- [HapticVLA](https://arxiv.org/abs/2603.15257) 做 tactile distillation，训练时用 tactile reward / teacher，推理时不需要 tactile sensing。
- [TaF-VLA](https://arxiv.org/abs/2601.20321) 做 tactile-force alignment，用大规模 tactile-force 数据学物理交互表征。
- [OmniVTA](https://arxiv.org/abs/2603.19201) 做 visuo-tactile world model 和 60Hz tactile reflexive controller。
- [Contact-VLA](https://openreview.net/forum?id=qPbDM5L8tE) 走 modular LLM + planning + reactive control 的 contact-rich 路线。

结论：

**contact-rich 是对的，但要避开“大力传感器/触觉硬件/大数据集”路线。**
普通实验室更有价值的问题是：没有 6-axis F/T、没有触觉阵列、没有大规模 tactile dataset 时，能不能用普通机器人日志构造 contact-aware VLA+RL？

### 1.4 VLA + RL guidance 也有 2026 新文献

- [VLAJS](https://arxiv.org/abs/2604.13733) 把 VLA 当作 early-stage transient guidance，用 directional action-consistency regularization jump-start PPO，随后 anneal 掉 VLA guidance，让高频 RL policy 超过 VLA。
- [WMPO](https://openreview.net/forum?id=qE2FyvRvuF) 用 world model 在 imagination 中做 VLA RL，目标也是降低真实交互样本。

结论：

**“VLA 指导 RL”本身也不新。**
真正缝隙是：VLA guidance 不应该只按训练时间 anneal，而应该按 **contact mode / physical mismatch** 动态信任或解除信任。

## 2. 新核心问题

推荐把题目从：

> When Should a Frozen VLA Be Corrected?

改成更具体、任务更硬的：

> When Should RL Trust a VLA in Contact-Rich Manipulation?

中文：

> 接触丰富操作中，RL 何时应信任 VLA？

核心观察：

- VLA 在 free-space approach 阶段通常有用：能找目标、接近物体、给出大方向。
- VLA 在 last-centimeter contact 阶段常常失效：看不见微小接触、摩擦、孔隙、卡滞、受力。
- 2026 force/tactile VLA 论文证明 contact signal 关键，但普通实验室未必有力/触觉硬件。
- 所以我们不做人类 boundary 标签，而是从机器人自动日志中估计 contact mode / physical mismatch，并据此动态决定：
  - 什么时候 VLA guidance 应该强；
  - 什么时候应该弱化 VLA，让 RL 探索接触策略；
  - 什么时候应该触发局部 residual / search primitive。

## 3. 最推荐方向

### 方向名

**Contact-Implicit VLA Jump-Start RL**

备用英文题目：

**Contact-Implicit VLA-Guided Reinforcement Learning for Last-Centimeter Manipulation**

中文：

**面向最后厘米操作的隐式接触感知 VLA 引导强化学习**

### 一句话贡献

不用人工划 `boundary/recoverability` 标签，也不依赖昂贵触觉/力传感器；只用普通机器人日志中的 commanded-vs-realized mismatch、proprioception、motor/current/load、视觉进度停滞等自动信号，学习 contact mode，并把 VLA guidance 做成 **contact-conditioned regularization**：free-space 信 VLA，contact/jam 阶段逐步放权给 RL。

## 4. 方法设计

### 4.1 自动 contact proxy，而不是人工 labels

不再让人标注 boundary/recoverability。只从日志自动算：

```text
commanded motion:  a_t or target EE delta
realized motion:   q_{t+1} - q_t or estimated EE delta
tracking error:    |commanded - realized|
visual progress:   target distance / keypoint distance / object state change
motor effort:      servo load / current / torque estimate if available
gripper effort:    gripper current / position mismatch if available
```

构造几个连续指标：

- `progress_ratio = realized_progress / commanded_motion`
- `tracking_error = ||a_cmd - delta_state||`
- `stagnation = no visual/proprio progress for N steps`
- `effort_proxy = motor_current or servo_load or joint_error`
- `contact_score = f(tracking_error, stagnation, effort_proxy)`

如果硬件没有电流/力：

- 用 commanded-vs-realized joint/EE mismatch。
- 用视觉关键点进度停滞。
- 用 gripper position mismatch。

这比人工 `boundary/recoverability` 标签干净很多。

### 4.2 Contact-conditioned VLA regularization

参考 VLAJS，但做关键改变：

VLAJS 是按训练时间逐渐 anneal VLA guidance。
我们改成按 contact state 动态调节：

```text
L = L_RL + lambda(c_t) * D_direction(a_rl, a_vla)
```

其中：

- free-space / approach：`lambda(c_t)` 高，RL 跟随 VLA 大方向，提升探索效率。
- contact established：`lambda(c_t)` 中等，只保留方向先验。
- jam / no progress：`lambda(c_t)` 低，甚至临时解除 VLA guidance，让 RL 学会 wiggle/search/retry。
- success-proximal：重新提高平滑/安全约束，避免过冲。

核心创新不是 residual head，而是：

**VLA guidance 的信任度由自动接触状态决定，而不是由训练步数或人工 phase 决定。**

### 4.3 Last-centimeter residual / primitive

动作可以不做复杂 actor-critic 接 VLA hidden states。更稳的是：

```text
a_exec = a_rl_lowlevel
L_RL includes contact-conditioned VLA regularization
```

或者：

```text
a_exec = a_vla + delta_a_rl
```

但只在 contact_score 高时启用 residual。`delta_a_rl` 要限制在末端小位移：

- translation：1-5 mm
- rotation：小角度
- gripper：小步开合

可以加入任务先验 micro-actions：

- peg insertion：spiral / grid search / rotate-wiggle / push-pull retry
- button pressing：normal-direction press + lateral correction
- drawer handle：hook-contact maintain + pull direction adaptation

这个任务先验不是坏事。它能让小论文更像“真实机器人 contact-rich skill adaptation”，而不是泛泛 VLA 方法。

## 5. 和 2026 文献的差异

### vs VLAJS

VLAJS 用 VLA early guidance jump-start RL，并随时间 anneal。
本方向用 **contact-conditioned trust**：VLA 在 free-space 有用，在 contact/jam 可能有害，guidance 权重由物理接触代理信号决定。

### vs ForceVLA2 / CRAFT / TaF-VLA / HapticVLA

这些工作证明 force/tactile 对 contact-rich VLA 很关键，但通常需要 force/tactile 数据、硬件或大规模训练。
本方向强调 **force/tactile-free or force-light**：只用普通机器人日志构造 contact proxy，适合低成本单臂平台。

### vs HIL-SERL / RLIF / SiLRI

这些工作以 human intervention 为核心信号。
本方向不要求人手工标 boundary，也不把接管当 reward 主体；接管可作为额外数据，但核心信号是自动 contact/progress mismatch。

### vs CR-DAgger

CR-DAgger 从 human delta correction 学 residual policy，强调 compliant intervention interface 和 force feedback。
本方向更强调 VLA+RL 中 **什么时候信任 VLA guidance**，且尽量不依赖人工 delta corrections。

### vs FPC-VLA / CycleVLA / RACER / Phoenix

它们主要做 failure prediction、language correction、backtracking、self-reflection。
本方向不做语言级失败纠正，而是做接触阶段的低层 RL 信任调度和 last-centimeter control。

## 6. 任务设计：必须针对 contact-rich last centimeter

### 任务 1：peg / USB 插入

为什么适合：

- VLA 能靠近孔位，但最后对准和插入失败。
- contact_score 有意义：卡住时 commanded push 不产生进展。
- 成功可用插入深度、视觉标记、接触开关或手工二值判定。

可做变体：

- 孔位随机偏移。
- clearance 改变。
- peg 形状 / USB 方向变化。

关键指标：

- 插入成功率。
- 平均尝试次数。
- jam 次数。
- 最大 effort proxy。
- 达到 80% 成功率所需 episodes。

### 任务 2：按钮 / 拨杆 / 开关

为什么适合：

- reset 简单，样本效率评估容易。
- VLA 可能接近按钮，但按压力方向/深度/时机不稳。
- 成功信号可以用电信号、视觉状态或简单开关反馈。

可做变体：

- 按钮刚度不同。
- 位置和角度变化。
- 小按钮 / 侧向开关。

### 任务 3：抽屉把手 / 滑块

作为扩展，不建议第一阶段主做。

价值：

- 能展示接触保持和方向适配。

风险：

- 摩擦、把手形状、抓取失败会混入太多因素。

## 7. 实验对比

最少保留：

1. `Frozen VLA`
2. `Frozen VLA + RTC`
3. `PPO/SAC from scratch`
4. `VLAJS-style global VLA regularization`
5. `Always-on VLA residual`
6. `Manual phase VLA guidance`
7. `Ours: contact-conditioned VLA guidance`

如果收了接管数据，再加：

8. `HIL-SERL / intervention reward`
9. `CR-DAgger-style residual BC`

关键消融：

- 无 contact proxy，仅时间 anneal。
- 只用 visual progress，不用 proprio mismatch。
- 只用 proprio mismatch，不用 visual progress。
- contact_score 阈值固定 vs learned。
- jam 阶段继续信 VLA vs jam 阶段降低 VLA guidance。

## 8. 这条线的真正创新点

### 创新点 1：VLA guidance 的 contact-conditioned trust

不是问“VLA 能不能指导 RL”，而是问：

**在接触丰富任务中，什么时候 VLA guidance 是有益先验，什么时候会阻碍 RL 发现接触策略？**

### 创新点 2：不靠人工 boundary labels 的自动接触代理

用 robot log 自动估计 contact / jam / progress stagnation，避免人工标签工程，也比泛泛 intervention 更接近物理失败机制。

### 创新点 3：面向 last-centimeter precision 的真实机器人实验

任务不追求大而泛，而是集中展示 VLA 的典型短板：视觉能看到目标，但不能处理最后接触。

## 9. 最小执行计划

第 1 阶段：

- 选 peg/USB 插入或按钮按压。
- 跑 frozen SmolVLA/PI05，让它能接近目标但最后失败。
- 记录 `policy_action`、executed action、joint state、视觉帧、成功信号。

第 2 阶段：

- 实现 contact proxy：
  - commanded-vs-realized mismatch
  - visual/proprio progress stagnation
  - motor current/load if available
- 离线画图验证 contact_score 是否能区分 free-space / contact / jam。

第 3 阶段：

- 实现 VLAJS-style regularization baseline。
- 实现 contact-conditioned regularization。
- 用 PPO/SAC/TD3 训练低层 policy 或 residual policy。

第 4 阶段：

- 两个任务完整对比。
- 做 contact_score 消融和 jam case 分析。

## 10. 最终建议

当前最值得押的方向不是：

- 人工 boundary/recoverability labels；
- 泛泛 failure prediction/correction；
- ACP++ prompt labels；
- 完整 RLT 复现。

而是：

**Contact-Implicit VLA Jump-Start RL：用自动物理接触代理动态调节 VLA 对 RL 的指导强度，解决 contact-rich last-centimeter manipulation。**

这条线既接得上 2026 VLA+RL 热点，也避开了 force/tactile 大硬件路线和 failure-correction 拥挤赛道，更适合普通实验室真实机器人小论文。
