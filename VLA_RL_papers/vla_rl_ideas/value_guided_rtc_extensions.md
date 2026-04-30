# Value-Guided RTC 的扩展创新点分析

## 结论先行

这个方向最值得写成论文的版本不是：

> 给 RTC 加一个价值函数，价值低了就重新切 chunk。

这个说法太像普通 adaptive replanning，容易被审稿人归到“动态 horizon / event trigger / critic gate”的老问题里。

更稳的论文表述是：

> 对冻结的 chunked VLA，学习 action chunk 内部的 value profile，并把 RTC 的固定时间切换改写成一个带不确定性、切换代价和执行延迟代价的 optimal stopping 问题；系统在旧 chunk、新 chunk、等待下一次推理之间选择最有期望进展且最少破坏连续性的切换时刻。

核心贡献应该落在：

```text
value profile over action chunks
+ cost-aware chunk switching
+ uncertainty/contact/progress calibrated switch timing
```

而不是泛泛地说“用 RL 改进 VLA”。这样和 RTC、A2C2、RLT、VLA-RL、VLAC、human-in-loop RL 都能拉开距离。

## 基础定义

冻结 VLA 在时刻 `t` 输出长度为 `H` 的 action chunk：

```text
A_t = pi_theta(o_t, l)
    = (a_{t|t}, a_{t+1|t}, ..., a_{t+H-1|t})
```

标准 RTC 关心的是如何异步生成新 chunk，并用 prefix freezing / inpainting / overlap consistency 让新旧 chunk 平滑衔接。它通常隐含一个固定执行节奏：

```text
execute m steps -> request next chunk -> overlap/inpaint -> switch
```

Value-Guided RTC 改成：

```text
when to switch = argmax over candidate switch times
```

在执行旧 chunk `A_old` 的同时，新 chunk `A_new` 可能已经生成，也可能仍在推理。对每个候选切换时刻 `h`，构造混合执行计划：

```text
P_h = old prefix for h steps + new suffix after h
```

学习一个 chunk 内价值剖面：

```text
Q_t(h) = expected future progress / success if switching at offset h
U_t(h) = uncertainty of that estimate
C_sw(h) = physical / semantic switching cost
C_lat(h) = latency cost from waiting or requesting another chunk
```

选择：

```text
h* = argmax_h [
    Q_t(h)
  - lambda_sw  * C_sw(h)
  - lambda_lat * C_lat(h)
  - lambda_u   * U_t(h)
]
```

这给论文一个清楚的技术命题：

> RTC 解决 chunk 边界的连续性；Value-Guided RTC 进一步决定哪些边界值得保留、哪些时间承诺应该被价值证据打破。

## 论文味的核心假设

固定 RTC 的弱点不是“它不够聪明”，而是它默认 chunk 内时间坐标足够可靠：

```text
chunk index k ~= task progress
```

真实机器人操作里这个假设经常不成立：

- 同一个 chunk 第 `k` 步，一次 rollout 可能刚接触物体，另一次已经进入卡住状态；
- 新 observation 表明任务状态已跨过事件边界，但旧 chunk 的后半段仍在执行旧意图；
- 价值下降通常不是均匀发生，而是集中在接触、夹取、插入、拉动、释放等窗口；
- 只追求平滑会把错误动作执行得更平滑；
- 只追求快速重采样会造成抖动、反复切换和推理延迟放大。

因此主问题可以写成：

> Given two temporally plausible chunks, when is the old temporal commitment no longer worth preserving?

中文表述：

> 当新旧 chunk 都是 VLA 认为合理的轨迹时，系统如何判断旧 chunk 的时间承诺已经失去价值，应该付出切换代价？

## 可引入创新点总览

| 创新点 | 适合作为主贡献吗 | 推荐角色 |
|---|---:|---|
| Value profile / Q profile | 是 | 主创新 |
| Uncertainty-aware switching | 否 | 必要安全阀 |
| Switching cost / latency cost | 是 | 主创新的一部分 |
| Progress reward / process value | 否 | 训练 value profile 的数据来源 |
| Human correction | 否 | 校准 value 和边界的强监督 |
| Phase detection | 否 | 辅助减少状态混淆 |
| Contact-aware switching | 否 | 机器人论文味很强的副创新 |
| Value disagreement | 否 | 不确定性和主动纠错信号 |
| World model rollout | 不建议第一篇做主线 | 可作为增强版或 future work |
| Teacher anchor | 否 | 稳定训练和安全约束 |
| Distillation | 否 | 工程闭环和实时性贡献 |

下面逐项分析。

## 1. Uncertainty-aware switching

### 为什么有用

Value-guided switching 最大的风险是 critic 误判。固定 RTC 虽然笨，但稳定；一个不校准的价值函数可能在 chunk 中段频繁触发切换，制造更差的控制轨迹。

因此不确定性不是锦上添花，而是让方法可信的必要条件。决策不应基于点估计：

```text
switch if Q_new > Q_old
```

而应基于保守改进：

```text
switch if LCB(Q_new - Q_old) > C_sw + C_lat
```

其中 `LCB` 是 lower confidence bound。

### 怎么实现

可选实现从轻到重：

- critic ensemble：训练 `K` 个 chunk value heads，均值给 value，方差给 uncertainty；
- bootstrapped replay：不同 correction / rollout 子集训练不同 head；
- distributional critic：输出 value 分布而不是单个标量；
- temporal consistency：如果相邻 offset 的 value 剖面剧烈跳变，增加不确定性；
- OOD score：用 VLA action likelihood、flow denoising residual、视觉 embedding 距离或 action chunk 偏离训练集程度估计分布外风险；
- disagreement with progress model：critic 说会成功，但视觉 progress model 说没有进展，则视为高不确定。

切换规则可以写成：

```text
Delta_Q(h) = Q_new(h) - Q_old(h)
score(h) = mean(Delta_Q(h))
         - beta * std(Delta_Q(h))
         - lambda_sw * C_sw(h)
         - lambda_lat * C_lat(h)
```

只有 `max_h score(h) > eta` 时才切换。

### 需要什么额外数据/模型

- 多头 critic 或 ensemble；
- rollout / demonstration / correction 数据；
- 用于校准的 held-out episodes；
- 可选 OOD detector 或 VLA action likelihood 接口。

### 风险

- ensemble 会增加延迟，必须蒸馏或降低频率；
- uncertainty 可能过于保守，导致系统在明显坏的 chunk 上坚持太久；
- 高不确定不等于高风险，有些任务阶段天然多模态；
- 如果没有可靠校准，LCB 只是形式上安全。

### 与已有论文是否撞车

和 RL 中 ensemble uncertainty、safe RL、offline RL OOD penalty 都有重叠，不能把“不确定性估计”当创新。它的定位应是：

> uncertainty is used to decide chunk commitment under RTC, not to build a generic safe RL algorithm.

和 A2C2 不撞车：A2C2 是每步 residual correction；这里是不确定性约束下的 chunk 切换时机。

## 2. Switching cost / latency cost

### 为什么有用

如果只看 value，系统会倾向于“新 chunk 看起来稍好就切”。真实机器人里切换不是免费的：

- 新旧 action 在速度、夹爪、末端方向上不连续；
- 频繁切换会造成 jerk 和视觉闭环抖动；
- 推理延迟会让新 chunk 到达时已经过时；
- 接触阶段切换可能破坏已经形成的物理约束；
- RTC inpainting 本身也有计算和调度开销。

切换代价让方法从“critic gate”变成“控制问题”。

### 怎么实现

切换代价可以分成三类。

第一类是运动连续性代价：

```text
C_motion(h) =
    || a_old[h] - a_new[0] ||^2
  + rho_v || v_old[h] - v_new[0] ||^2
  + rho_g 1[gripper mode changes abruptly]
```

第二类是语义承诺代价：

```text
C_semantic(h) =
    1[old and new chunks imply different object / affordance / phase]
```

可以用视觉 attention、object keypoint、language-conditioned progress head 或 phase token 近似。

第三类是延迟代价：

```text
C_latency(h) =
    expected progress loss while waiting for next VLA inference
  + probability that observation becomes stale
```

简单实现可以直接使用实测推理耗时 `delta_infer`：

```text
C_latency = alpha * delta_infer + beta * queue_age + gamma * missed_control_steps
```

控制上加滞回：

```text
switch only if score_new > score_keep + margin
minimum dwell time after switch
maximum stale age before forced replan
```

### 需要什么额外数据/模型

- 机器人控制日志：action、executed action、FK pose、gripper state；
- RTC 队列状态：chunk age、inference latency、overlap residual；
- 可选 phase/object detector；
- 不一定需要额外大模型，很多代价可手工定义。

### 风险

- 代价权重太大时，系统会保守到错过纠错窗口；
- 手工代价可能被审稿人认为工程 heuristic；
- 如果没有 ablation，难证明是 value 起作用还是 hysteresis 起作用；
- 切换代价和 RTC 原本的 smoothness 目标容易表述重复。

### 与已有论文是否撞车

和 MPC、hybrid systems、event-triggered control 的 switching cost 概念重叠明显。论文不能声称“首次引入切换代价”。更好的定位是：

> We instantiate switching costs for VLA action chunks under asynchronous RTC, where the cost includes physical discontinuity, stale inference, and semantic commitment.

这和 RTC 撞得不严重，因为 RTC 的核心是实时执行和边界平滑，不是基于任务价值和代价选择切换时刻。

## 3. Progress reward / process value

### 为什么有用

chunk 内 value profile 不能只靠 episode success 训练。成功/失败太晚，无法判断：

- 是 chunk 前半段方向错了；
- 是接触后卡住；
- 是旧 chunk 应该多执行两步；
- 还是新 chunk 应该提前接管。

需要一个短期 process reward：

```text
r_prog(t, h) = Phi(s_{t+h}, l) - Phi(s_t, l)
```

`Phi` 是任务进展函数。它不是泛用 reward model，而是为 chunk switching 服务。

### 怎么实现

按任务复杂度分层：

- 几何 progress：末端到目标、物体到目标、抽屉开合距离、按钮状态；
- 视觉 progress：keypoint / AprilTag / segmentation 估计任务轴；
- contact progress：夹住后物体是否跟随，插入深度是否增加；
- VLM progress：对难以写几何 reward 的任务，用视频语言模型判断阶段进展；
- human correction progress：干预前后任务状态改善量作为 weak label。

训练目标：

```text
Q_phi(o_t, A_t, k) ~= E[sum_{i=0}^{h} gamma^i r_prog(t+i)]
```

也可以只学二分类：

```text
y_t(k) = 1[executing prefix up to k produces measurable progress and no intervention]
```

### 需要什么额外数据/模型

- 每个任务的 progress detector；
- 轨迹级成功标签；
- 短窗口状态变化；
- 可选 VLM reward 或 VLAC/SOLE-R1 类 process reward；
- correction/intervention 日志。

### 风险

- progress 不一定单调：绕开障碍、预抓取、对准阶段可能短期远离目标；
- reward hacking：机器人可能优化可测进展而不是任务完成；
- 几何 progress 任务特化强，泛化 claim 要收窄；
- VLM reward 延迟大且不稳定，不适合作为实时切换唯一信号。

### 与已有论文是否撞车

和 VLAC、SOLE-R1、VLA-RL、process reward model 高度相关，所以它不适合单独做主创新。最稳定位是：

> Progress reward is not the contribution; it is the supervision that makes chunk-level value profiles observable.

也就是说，我们不是提出新的 reward model，而是把 process reward 用在 RTC 的 chunk switching boundary 上。

## 4. Human correction / intervention labels

### 为什么有用

人类纠正信号天然回答了 Value-Guided RTC 的问题：

> 从哪个时间点开始，当前 chunk 不值得继续相信？

干预开始时刻通常是旧 chunk 价值已经明显下降的证据；干预后的动作则提供一个局部 teacher plan。

这比 episode failure 更细：

- failure 只告诉最后没成功；
- human correction onset 告诉哪里开始不可信；
- corrected trajectory 告诉更好的局部方向；
- correction magnitude 告诉旧 chunk 偏离程度。

### 怎么实现

记录：

```text
(o_t, A_old, A_new, executed action, intervention flag, correction action)
```

构造标签：

```text
y_bad(t, k) = 1[human intervenes within next h steps]
y_good(t, k) = 1[no intervention and progress improves]
```

训练：

- value head：预测未来 progress 和 intervention risk；
- switch head：预测 keep old / switch new / ask correction；
- teacher anchor：用 correction action 对新 chunk 或 residual head 做局部监督；
- active query：当 value disagreement 高时请求 human check，而不是等失败。

切换规则中加入 intervention risk：

```text
score(h) = Q_prog(h)
         - lambda_risk * P(intervention soon | h)
         - lambda_cost * C_sw(h)
```

### 需要什么额外数据/模型

- teleop 或 shared autonomy 接口；
- intervention flag 的精确时间戳；
- 人类 correction action；
- correction 前后 progress；
- 可选 operator confidence / reason 标签。

### 风险

- 人类干预有延迟，onset 不一定等于真实错误开始；
- 不同操作者风格差异大；
- 纠正数据集中在失败边界，分布偏；
- 如果过度依赖人类，论文会被归到 HIL-RL，而不是 RTC 创新。

### 与已有论文是否撞车

和 HIL-SERL、RECAP、human-in-the-loop RL、DAgger 类方法重叠明显。避免撞车的表述：

> Human corrections are used to calibrate chunk value decay and switching hazards, not to train a full policy from scratch.

这能和 Value-Guided RTC 绑定起来。

## 5. Phase detection

### 为什么有用

同一个 action chunk offset 在不同任务阶段含义完全不同。比如 `k=10`：

- 接近阶段：可以频繁重采样；
- 抓取闭合阶段：需要稳定执行；
- 接触插入阶段：小幅修正比整体切换更安全；
- 释放后：可以快速切下一个 chunk。

如果 value head 不知道 phase，会把不同阶段的数据混在一起，导致 Q profile 模糊。

### 怎么实现

学习一个轻量 phase token：

```text
z_phase = f_phase(o_t, proprio_t, A_t, progress_t, contact_proxy_t)
```

它不直接决定动作，只调节：

- value head；
- switch threshold；
- switching cost；
- uncertainty calibration；
- residual authority。

可监督来源：

- demonstration 中的弱 phase label；
- progress landmark：接触、夹住、开始移动、到达目标、释放；
- human correction onset；
- visual change point；
- RTC overlap residual peak。

如果不想显式命名 phase，可以用 latent phase：

```text
Q(o, A, k, z_phase)
C_sw(o, A_old, A_new, k, z_phase)
```

### 需要什么额外数据/模型

- 少量 phase label 或自动事件 detector；
- 视觉/proprio 历史；
- 可选 HMM、temporal transformer、change-point detector；
- 与 progress/contact 信号共享特征即可。

### 风险

- 容易退化成人工 FSM；
- phase 边界本身可能不清楚；
- 多任务泛化时 phase taxonomy 很难统一；
- 如果 phase 只是另一个分类器，贡献会显得薄。

### 与已有论文是否撞车

和 skill segmentation、option discovery、event-conditioned control 有重叠；和本 repo 里 Event-Conditioned RTC 的想法也接近。要避免撞车，需要强调：

> phase is a conditioning variable for value and switching cost, not a replacement for RTC or a hand-coded task controller.

如果本文主线已经是 Value-Guided RTC，phase detection 适合作为 ablation 中的辅助模块，不适合作为标题关键词。

## 6. Contact-aware switching

### 为什么有用

机器人操作里，切换时机最敏感的地方通常是 contact：

- 接触前：需要快速根据视觉误差重采样；
- 初次接触：错误 chunk 会卡住或打滑；
- 稳定接触中：大幅切换可能破坏接触约束；
- 卡住时：继续平滑执行旧 chunk 只会更糟。

因此 contact-aware switching 的关键不是“检测接触”，而是：

> 在不同接触证据下，切换代价和切换收益的权重不同。

### 怎么实现

低成本 contact proxy：

```text
kappa_t = [
    commanded_EE_delta - realized_EE_delta,
    gripper_cmd - gripper_obs,
    object_motion - EE_motion,
    progress_stagnation,
    joint_tracking_error,
    motor_current_or_load_if_available,
    RTC_overlap_residual
]
```

规则或学习型 gate：

```text
C_sw_contact =
    high  if stable contact and progress positive
    low   if contact proxy high and progress stagnant
    medium if pre-contact visual alignment
```

更论文式的写法：

```text
C_sw(h) = C_motion(h) + C_contact(h, kappa_t, Delta_Phi_t)
```

其中 `kappa_t` 不是 ground-truth force，而是 contact-implicit evidence。

### 需要什么额外数据/模型

- proprio 和 gripper 反馈；
- FK 计算的 EE motion；
- 视觉 object motion 或 keypoint；
- 可选 motor current / load / force sensor；
- stuck / slip / contact success 的弱标签。

### 风险

- 没有 F/T 或 tactile 时，contact proxy 可能误报；
- progress stagnant 不一定是接触失败，可能是对准阶段；
- contact gate 太强会掩盖 value head；
- 不同硬件的 contact proxy 差异大。

### 与已有论文是否撞车

接触丰富操作、impedance control、force-aware RL 都做过很多。不要声称“提出 contact-aware manipulation”。更稳的是：

> We use contact-implicit evidence to modulate RTC switch commitment for frozen chunked VLAs.

这比泛泛 contact policy 更新更具体。

## 7. Value disagreement

### 为什么有用

在 Value-Guided RTC 里，真正重要的不是绝对 value，而是几个候选计划之间的相对排序：

```text
keep old
switch new now
switch new later
wait for next chunk
ask human / fallback
```

如果不同 value heads、progress model、teacher anchor 对排序意见不一致，说明系统正处在关键边界。这个信号可以用于：

- 增加不确定性惩罚；
- 延迟切换；
- 降低执行速度；
- 请求 human correction；
- 触发新 observation / new chunk。

### 怎么实现

定义几个 disagreement：

```text
D_ensemble = Var_i[Q_i(h)]
D_old_new  = |Q_old(h) - Q_new(h)|
D_prog_succ = |Q_progress(h) - Q_success(h)|
D_teacher = distance(A_candidate, A_teacher_or_VLA_prior)
```

切换策略：

```text
if D_ensemble high and Delta_Q small:
    keep current RTC / slow down
if D_ensemble high and intervention risk high:
    query human or fallback
if Delta_Q high and D low:
    switch
```

### 需要什么额外数据/模型

- ensemble 或至少两个不同 supervision 的 value heads；
- progress and success heads；
- teacher/VLA prior distance；
- correction labels 用于校准 disagreement 是否真的预测失败。

### 风险

- disagreement 高不一定危险，可能只是多模态；
- disagreement 指标太多会让方法显得堆模块；
- 如果没有展示它预测 intervention/stuck，审稿人会认为是普通 ensemble trick。

### 与已有论文是否撞车

和 active learning、ensemble RL、uncertainty-aware RL 有大量重叠。它适合写成：

> value disagreement is an observable signal for chunk boundary ambiguity.

不要把它写成独立创新。

## 8. World model rollout

### 为什么有用

value profile 是隐式预测；world model rollout 是显式模拟：

```text
roll out old chunk prefix
roll out new chunk prefix
compare predicted progress / contact / failure
```

它能回答一些 critic 难以直接判断的问题：

- 继续旧 chunk 会不会撞到、卡住、越过目标；
- 新 chunk 的前几步是否会破坏接触；
- 等待下一次推理期间状态是否会变得不可恢复；
- 多个候选 switch time 哪个短期视觉结果更合理。

### 怎么实现

轻量版本：

- 学一个 latent dynamics model，只预测 progress state 和 contact proxy；
- 输入 `(o_t, proprio_t, action prefix)`；
- 输出 `Phi_hat(t+h), kappa_hat(t+h), failure_hat(t+h)`；
- rollout horizon 很短，通常 `5-15` 个控制步。

重版本：

- video world model；
- VLM reflector / reward model；
- model-predictive chunk switching。

决策：

```text
Q_rollout(h) = R_hat(world_model(o_t, P_h))
score(h) = alpha * Q_critic(h) + (1-alpha) * Q_rollout(h) - costs
```

### 需要什么额外数据/模型

- 大量 transition data；
- dynamics model；
- progress/reward model；
- compute budget；
- rollout calibration data。

### 风险

- rollout error 在 contact-rich 任务中很严重；
- 视频模型太重，不适合实时 RTC；
- 容易把论文带到 world-model VLA post-training 的赛道，主线变散；
- 如果只在 sim 里验证，real-robot 贡献变弱。

### 与已有论文是否撞车

和 WORLD-ENV、VLA-RFT、RynnVLA-002、reward world model 等方向撞得比较近。建议第一篇不要把它放进主贡献。可以写成：

> A short-horizon latent rollout can optionally provide an auxiliary value estimate, but the core method does not require a full world simulator.

## 9. Teacher anchor

### 为什么有用

Value-guided switching 会引入一个危险：critic 可能为了短期 progress 选择偏离 VLA prior 的动作组合。teacher anchor 用来保证系统仍在 VLA 认为合理的行为流形附近。

它尤其适合冻结 VLA 场景：

- VLA 提供大范围泛化和语言理解；
- critic/switcher 只决定何时保留或打破 VLA chunk；
- anchor 防止小模型越权。

### 怎么实现

anchor 可以约束 action、chunk、或 latent：

```text
C_anchor(h) =
    || P_h - A_vla_reference ||^2
```

如果能拿到 flow/diffusion likelihood：

```text
C_anchor(h) = - log p_theta(P_h | o_t, l)
```

如果只能拿到 sampled chunks：

```text
C_anchor = distance to nearest VLA sampled chunk
```

切换目标：

```text
score(h) = Q(h)
         - lambda_sw * C_sw(h)
         - lambda_anchor * C_anchor(h)
```

teacher anchor 也可以用于训练：

- positive：无干预且成功的 VLA chunk；
- negative：干预前的失败 chunk prefix；
- corrected teacher：human correction 后的局部 action。

### 需要什么额外数据/模型

- VLA 输出的 reference chunks；
- 可选 VLA likelihood / denoising score；
- correction trajectory；
- 不需要额外 teacher model，原 VLA 就是 teacher。

### 风险

- anchor 太强会无法修正 VLA 的系统性错误；
- 如果 teacher 本身在 contact stage 犹豫，anchor 会保留慢动作；
- likelihood 接口对部分开源 VLA 不一定可用；
- 容易和 RLT/VLAJS 的 VLA regularization 概念重叠。

### 与已有论文是否撞车

和 RLT 的 policy anchoring、VLAJS 的 VLA regularization、RPD 的 teacher-guided RL 都有明显关系。不要把 teacher anchor 写成贡献。它在这里的独特点是：

> anchor regularizes the switch plan, not a newly trained policy.

这个角度可以作为方法组件。

## 10. Distillation

### 为什么有用

如果完整 Value-Guided RTC 包含 ensemble、progress model、contact proxy、human-risk head、world rollout，实时系统会太重。Distillation 能把复杂决策压成一个低延迟 switch head：

```text
h_psi(o_t, A_old, A_new, debug_t) -> keep / switch / shorten horizon
```

这对 RTC 很重要，因为 RTC 的初衷就是解决推理延迟。不能为了 value-guided switching 又引入更大延迟。

### 怎么实现

离线或低频运行 teacher planner：

```text
teacher decision = argmax_h full score(h)
```

收集标签：

```text
(o_t, A_old, A_new, Q_profile, U_profile, costs, teacher h*) 
```

训练轻量学生：

```text
student predicts:
    switch probability
    target switch offset
    horizon shortening factor
    uncertainty / fallback flag
```

部署时：

- 每个控制步运行 student；
- 低频运行 full critic 校准；
- 高 disagreement 时回退到 conservative RTC。

### 需要什么额外数据/模型

- full planner 产生的离线标签；
- 真实 rollout 验证 student 决策；
- 轻量 temporal model；
- 可选 quantization / small MLP。

### 风险

- student 可能学到 teacher 的错误；
- distillation 后 uncertainty 丢失；
- 如果 student 输入仍依赖昂贵特征，实时性收益不明显；
- 论文容易被看成工程压缩，需和 RTC latency 指标绑定。

### 与已有论文是否撞车

和 policy distillation、RPD、consistency distillation 都有重叠。本文可写成：

> We distill a value-based scheduler, not the VLA policy itself.

这能保持主线清晰。

## 最容易撞车的说法

以下表述应避免：

1. “我们提出一个 value function 改进 VLA。”
   这会撞 VLA-RL、RLT、VLAC、CO-RFT、offline RL、process reward。

2. “我们提出动态 RTC horizon。”
   这会显得像普通 adaptive control 或 event-triggered replanning。

3. “我们检测 phase/contact 来决定切换。”
   这会撞 skill segmentation、FSM、event-conditioned RTC、contact-aware control。

4. “我们用 human correction 训练机器人。”
   这会撞 HIL-SERL、DAgger、RECAP。

5. “我们用 world model 选择动作 chunk。”
   这会撞 WORLD-ENV、VLA-RFT、world-model RL。

更安全的中心句是：

> We formulate real-time action chunk switching as value-calibrated commitment management: a frozen VLA proposes temporally smooth chunks, while a lightweight scheduler decides when preserving the old chunk is no longer worth its value loss, switching cost, and latency cost.

## 推荐组合

### 主创新：Cost- and Uncertainty-Calibrated Value-Guided RTC

主方法：

- 为旧 chunk、新 chunk、候选 switch offset 学习 value profile；
- 把切换时机写成 optimal stopping / commitment management；
- 显式加入 switching cost、latency cost、uncertainty penalty；
- 输出不是动作，而是 `keep / switch / shorten horizon / wait`。

这个主创新足够像论文，因为它重新定义了 RTC 的决策变量：

```text
from fixed temporal schedule
to value-calibrated chunk commitment
```

### 副创新 1：Progress- and Correction-Calibrated Chunk Value

训练 value profile 的监督来自：

- 短期 progress reward；
- intervention onset；
- correction magnitude；
- no-intervention successful prefixes；
- optional success label。

它解决主创新最大的质疑：

> chunk 内 value 从哪里来？

不要把它写成新的 reward model，而要写成：

> progress and correction signals make chunk commitment empirically observable.

### 副创新 2：Contact/Phase-Aware Distilled Switch Head

部署时不能每步跑很重的 critic ensemble。因此把 full value-based scheduler 蒸馏成实时 switch head，并用 contact/phase proxy 调节阈值：

- pre-contact：允许更频繁重采样；
- stable positive contact：提高切换代价；
- contact + progress stagnation：降低切换代价，允许打断；
- phase boundary：缩短 horizon；
- high uncertainty：回退 conservative RTC 或请求 human correction。

这个副创新让论文更像机器人系统论文，而不是纯算法草图。

## 一篇完整论文的故事线

### 研究问题

Action chunking 和 RTC 让大 VLA 能实时控制机器人，但 RTC 默认 chunk 切换由固定时间和连续性约束决定。真实操作中，chunk 的价值在内部并不均匀：接触、夹取、插入、拉动等关键阶段会让旧 chunk 的承诺突然失效。问题是如何在不微调整个 VLA 的情况下，判断何时应该继续相信旧 chunk，何时应该付出切换代价。

### 方法路线

1. 冻结 VLA，保留原始 RTC 的异步推理和 overlap 机制。
2. 在 RTC 执行过程中记录旧 chunk、新 chunk、执行状态、progress、correction、latency 和 contact proxy。
3. 学习 chunk value profile，预测不同 switch offset 的未来 progress 和 intervention risk。
4. 用带 switching cost、latency cost 和 uncertainty 的 objective 选择切换时刻。
5. 用 contact/phase proxy 调节切换阈值。
6. 将完整 planner 蒸馏成低延迟 switch head，用于在线部署。

### 实验主张

需要证明的不只是成功率变高，而是下面几件事：

- 固定 RTC 在某些任务阶段会坚持低价值 chunk；
- value profile 能预测 future progress / intervention；
- cost-aware switching 比 naive value switching 更少抖动；
- uncertainty calibration 能减少错误切换；
- contact/phase gate 能改善接触阶段；
- distillation 后实时延迟可接受；
- 与 A2C2 叠加时仍有增益，说明它不是 per-step correction 的替代品。

### 关键 baseline

- no RTC / naive chunk execution；
- standard RTC；
- RTC + fixed shorter horizon；
- RTC + progress threshold switch；
- RTC + A2C2；
- RTC + value switch without cost；
- RTC + value switch without uncertainty；
- RTC + value switch without contact/phase；
- full method；
- optional：RLT-like residual / VLAJS-like VLA-regularized RL 作为强相关比较。

### 关键指标

- success rate；
- time-to-success；
- number of switches；
- jerk / action discontinuity；
- intervention rate；
- progress stagnation duration；
- contact failure rate；
- latency robustness；
- value calibration error；
- switch decision precision/recall against human intervention onset。

## 标题候选

1. **Knowing When to Rechunk: Value-Guided Real-Time Chunking for Vision-Language-Action Robots**
2. **Value-Guided RTC: Cost-Aware Action Chunk Switching for Frozen VLA Policies**
3. **Rechunking by Value, Not by Time: Commitment Management for Real-Time VLA Control**
4. **When Should a Robot Keep Its Chunk? Value-Calibrated Switching for Action-Chunked VLAs**
5. **Cost-Aware Value Profiles for Real-Time Action Chunk Execution**

我最推荐第 1 个或第 3 个。第 1 个学术味更稳，第 3 个更有记忆点。

## 贡献列表草案

可以写成三条：

1. We formulate real-time action chunk execution as a value-calibrated switching problem, replacing fixed RTC schedules with an optimal stopping objective over chunk value profiles, switching costs, latency costs, and uncertainty.

2. We introduce a progress- and correction-calibrated chunk value model that predicts when an executing VLA chunk will stop producing task progress, using short-horizon progress, intervention onset, and correction magnitude as supervision.

3. We develop a contact/phase-aware distilled switch head that preserves RTC-level real-time execution while adapting chunk commitment in contact-rich manipulation, and evaluate it against standard RTC, A2C2-style correction, and value-only switching baselines.

中文解释：

- 第一条是主理论框架；
- 第二条回答 value profile 怎么学；
- 第三条回答怎么部署在真机、为什么不是离线 oracle。

## Related Work 定位

### RTC / action chunk execution

RTC 的核心贡献是异步 chunk 推理和边界连续性。Value-Guided RTC 不替代 RTC，而是在 RTC 给出可执行的新旧 chunk 后，决定旧 chunk 的承诺是否仍有价值。

定位句：

> RTC asks how to execute action chunks smoothly under inference delay; we ask when a smooth chunk should be interrupted because its expected task value has decayed.

### A2C2 / real-time action correction

A2C2 每个控制步加 residual correction，解决 observation 更新后 chunk 内动作过时的问题。Value-Guided RTC 解决的是 chunk-level scheduling，不直接输出每步 residual。两者可以叠加：

```text
A2C2: correct the action being executed
Value-Guided RTC: decide which chunk should own the next actions
```

定位句：

> Per-step correction improves local reactivity; value-guided switching changes the temporal commitment structure of RTC.

### RLT / VLA-RL / online RL with VLA

RLT 使用 VLA 内部 RL token 和 actor-critic 做在线强化学习。本文不要求 VLA 暴露 native token，也不训练新的主 policy，而是学习一个 post-hoc scheduler。

定位句：

> Rather than fine-tuning or augmenting the VLA policy, we learn a lightweight scheduler that decides when to preserve or revise the VLA's existing chunk commitments.

### Process reward / VLAC / SOLE-R1

这些工作强调过程奖励、视觉语言 reward 或 critic。本文不把 reward model 当主贡献，而是把 process reward 用在 action chunk 内的 switching supervision。

定位句：

> Process rewards provide supervision for value profiles, but the decision problem is chunk switching under real-time latency and continuity constraints.

### Human-in-the-loop RL

Human correction 已经是成熟路线。本文只用 correction onset 和 correction magnitude 作为 chunk value decay 的校准信号。

定位句：

> Human interventions are treated as temporal labels for lost chunk trust, not as the primary mechanism for policy learning.

### World model VLA post-training

World model 可以辅助预测 switch 后果，但第一篇不应押宝。否则论文会变成虚拟环境训练，而不是 RTC 扩展。

定位句：

> Short-horizon rollout can provide an auxiliary estimate, while our core method remains executable without a full world simulator.

## 最终推荐写法

论文不要叫“Value-Guided RTC Extensions”。正式方向应收敛成：

```text
Value-Guided Real-Time Chunking
```

副标题强调：

```text
Cost-Aware Chunk Commitment for Frozen Vision-Language-Action Policies
```

核心卖点：

- 不改 VLA；
- 不要求 native RL token；
- 不替代 RTC；
- 不依赖完整 world model；
- 用 value profile 判断 chunk commitment；
- 用 cost/uncertainty 防止乱切；
- 用 progress/correction/contact 让 value profile 可学、可解释、可部署。

最强故事线：

> 当前 RTC 让 VLA “能实时执行 chunk”，但没有回答“应该执行这个 chunk 多久”。我们提出 Value-Guided RTC，把 chunk 执行长度从固定时间超参变成一个由任务价值、切换代价、推理延迟和不确定性共同决定的 stopping problem。通过 progress 和 human correction 学到 chunk 内 value profile，再用 contact/phase-aware distillation 实时部署，系统能在保留 RTC 平滑性的同时更早打断低价值 chunk，减少接触阶段卡住和人为干预。

