# Progress-Producing Chunk Trust / Contact-Implicit Chunk Trust Tokens

## 结论先行

这个想法可以作为主线，但必须把 claim 收窄。

不应说：

> 低成本 Piper 在没有 F/T、tactile 的情况下能理解接触、提前预测失败。

更严肃的说法是：

> 对冻结 chunked VLA 输出的 action chunk，学习一个后验 trust token，预测当前 chunk 或 chunk prefix 在未来短窗口内是否仍然产生可测任务进展；当 trust 下降时，才打开 chunk-local residual / RL / human correction。

它的创新点不是“读到了接触”，而是：

```text
chunk-level progress trust
+ contact-implicit failure boundary
+ residual control authority
```

这比“给 VLA 外接一个 RL token”更清楚，因为 token 的语义可测、可消融、可和机器人底层日志绑定。

## 核心问题

PI0、PI0.5、SmolVLA、OpenPI 类模型通常不是输出单步动作，而是输出一段 action chunk：

```text
A_t = (a_t, a_{t+1}, ..., a_{t+H-1})
```

真实失败经常不是第一步就错，而是：

- chunk 前半段能接近目标；
- 后半段进入接触、夹取、拉动、插入、放置等阶段；
- 继续执行原 chunk 不再产生 progress；
- RTC 可以让动作更平滑，但平滑错误 chunk 仍然会失败；
- episode-level success 太晚，无法告诉模型“从哪段 chunk 开始不可信”。

因此这里要回答的问题是：

> 当前 VLA chunk 还值得继续执行吗？

更具体地：

> 如果继续沿当前 chunk 的末端方向、夹爪动作或局部轨迹执行 `m` 个控制步，它是否会让任务状态朝成功方向移动？

## 一句话定义

**Contact-Implicit Chunk Trust Token** 是一个 post-hoc token/adaptor。它输入冻结 VLA 的 observation、language、proprioception、action chunk、executed action、RTC debug/correction、visual progress 和可选 intervention 信号，输出当前 chunk 在短 horizon 内产生任务进展的 calibrated trust score，并用这个 score 门控 chunk 执行、chunk 局部编辑或 residual RL。

英文式表述：

> We learn a post-hoc trust token for frozen chunked VLAs that predicts whether a proposed action chunk remains progress-producing under implicit contact evidence, and use it to gate chunk-local residual reinforcement.

## 理论形式

冻结 VLA：

```text
A_t = pi_theta(o_t, l)
```

其中：

- `o_t`：RGB、proprio、历史动作等观测；
- `l`：语言任务；
- `A_t`：长度为 `H` 的 action chunk；
- `pi_theta`：冻结，不做全参训练。

定义任务进展函数：

```text
Phi(s_t, l) in R
```

在 Piper 真机上，`Phi` 不应一开始依赖复杂 VLM reward。更可靠的来源是：

- 末端到目标点距离；
- 目标物体位移；
- 抽屉开合距离；
- 按钮/开关状态；
- 夹爪是否夹住物体后物体跟随；
- AprilTag / ArUco / keypoint 估计出的任务轴 progress；
- human intervention 前后的修正结果。

短期 progress：

```text
Delta Phi_t^h = Phi(s_{t+h}, l) - Phi(s_t, l)
```

隐式接触/执行异常代理：

```text
kappa_t = g(o_t, A_t, a_exec_t, q_t, q_{t+1}, image_t, image_{t+1})
```

它不等于真实力觉，只是低成本可观测证据：

- commanded EE delta 和 realized EE delta 不一致；
- 命令仍在推进，但任务轴 progress 下降；
- 视觉上物体不动或偏离；
- 夹爪目标位置和实际位置不一致；
- RTC overlap correction norm 变大；
- residual / human correction 开始出现；
- 位置控制下出现持续 stagnation。

trust label 可以写成：

```text
y_t = 1[
    Delta Phi_t^h > eps_progress
    and no_intervention_t
    and no_safety_stop_t
    and not_stuck_t
]
```

也可以用软标签：

```text
y_t = sigmoid(
    alpha * Delta Phi_t^h
  - beta  * stagnation_t
  - gamma * intervention_t
  - lambda * risk_t
)
```

trust token：

```text
z_trust_t = f_psi(o_t, l, A_t, h_vla_t, r_rtc_t, kappa_t)
tau_t = sigmoid(w^T z_trust_t)
```

控制：

```text
if tau_t >= eta:
    execute VLA chunk with RTC
else:
    execute VLA chunk + clipped chunk-local residual
```

连续版本：

```text
A_exec = A_vla + (1 - tau_t) * clip(Delta A_residual, -epsilon, epsilon)
```

## 为什么它比普通 plug-in RLT 更像研究问题

普通 plug-in RLT 容易被质疑为：

> 把 RLT 的 native RL token 换成外接 adaptor。

这个想法更清楚，因为 token 的语义不是泛化的“RL state”，而是一个可测命题：

```text
current chunk / prefix will produce short-horizon progress
```

它的对象是 action chunk，不是单步 action；它的标签是 progress-producing，不是 success classifier；它的用途是分配控制权，不是只给 policy 多一个 latent。

和 RLT 的关系：

| 维度 | RLT | Chunk Trust Tokens |
|---|---|---|
| token 来源 | VLA native token | post-hoc adaptor |
| 是否需要预训练期接口 | 需要 | 不需要 |
| token 语义 | RL-friendly state | chunk 是否产生 progress |
| 控制方式 | actor-critic refinement | trust-gated residual/editor |
| 适合设置 | 内部模型接口可用 | 冻结开源 VLA / Evo-RL |

最稳的定位：

> RLT shows that VLA representations can bootstrap online RL when native RL tokens are available. We study the complementary open-source setting where native tokens are unavailable, and ask whether chunk-level progress trust can be learned post-hoc from corrections and execution traces.

## 和 RTC 的关系

RTC 解决 chunk continuity：

> 新旧 chunk 怎么平滑衔接。

Chunk Trust 解决 chunk correctness：

> 当前 chunk 虽然平滑，但是否还在推动任务成功。

二者最自然的结合是把 RTC debug 信号作为 trust 输入：

```text
r_rtc = || A_new_overlap - A_prev_leftover ||
```

如果 RTC 为了接上旧 chunk 需要很大 correction，这不必然表示失败，但很可能表示 VLA 的 chunk prediction 和真实执行状态不一致。这个不一致在接触、夹取、拉动、插入窗口尤其有用。

关键实验：

- `r_rtc` 是否预测 future intervention；
- `r_rtc` 是否预测 progress stagnation；
- `r_rtc + visual progress + execution mismatch` 是否优于单独 progress critic；
- trust-gated residual 是否优于 vanilla RTC。

## 当前 Piper 环境能读到什么、不能读到什么

最可用：

- VLA / policy 输出 action chunk；
- 实际发送给 Piper 的 joint 或 EEF command；
- Piper 读回的 joint position；
- 由 FK 计算的 EE pose / EE delta；
- gripper target position 和 gripper observed position；
- 相机 RGB；
- RTC action queue、latency、overlap correction；
- human intervention / teleop correction；
- 任务 progress 的外部视觉估计。

谨慎使用：

- motor speed：可用作运动响应特征，但不等于接触；
- motor current / effort：如果 SDK 暴露，可能噪声较大，受姿态、温度、摩擦影响；
- collision/stall flag：如果有，可以做辅助标签或 safety signal，但不要作为唯一 evidence；
- raw RGB：应先变成 keypoint、tag、object displacement、drawer distance 等 progress 特征。

不应依赖：

- 真实 contact force；
- 真实 torque；
- 夹爪真实夹持力；
- 接触法向；
- 物体是否被稳定夹紧但视觉不可见；
- 无视觉标签下的任务语义失败原因。

位置控制下尤其要注意：不要设计“硬顶住再判断”的实验。这个方法适合检测 sustained stagnation 或可恢复失败，不适合让机械臂反复顶硬接触。实验应限制速度、动作幅度、末端 force proxy 和安全停止。

## 最适合的任务

当前设备上更适合：

1. **抽屉任务**
   - 打开抽屉、放入物体、关闭抽屉；
   - progress 可由抽屉位移和物体位置估计；
   - 接触主要是夹住把手、拉动、释放，不需要硬顶。

2. **按钮 / 开关**
   - 事件边界清楚；
   - progress 可由按钮位移、电信号或视觉状态估计；
   - 低速按压风险较低。

3. **粗公差插入**
   - 只适合作为后续任务；
   - 需要大 clearance、低速、软材料或机械限位；
   - 不应从高精度小孔插入开始。

4. **夹取后放入容器**
   - trust 主要判断 gripper/物体是否跟随；
   - 比硬接触装配更安全。

## 训练信号

### Supervised trust learning

正样本：

- chunk 后 task progress 增加；
- 无 intervention；
- 无 safety stop；
- 夹爪闭合后物体跟随；
- 抽屉沿正确方向移动；
- value/progress critic 上升。

负样本：

- progress 停滞或下降；
- human intervention；
- gripper 闭合但物体不跟随；
- 抽屉把手没夹住仍继续拉；
- RTC correction norm 持续变大；
- residual 大幅修正；
- 进入不可恢复失败。

### Progress delta regression

```text
L_progress = || Delta Phi_pred - Delta Phi_observed ||^2
```

这让 token 不只是二分类，而能估计 progress 幅度。

### Chunk-action contrast

同一状态下替换不同 chunk：

- push/pull 正确方向；
- 相反方向；
- lateral search；
- retract；
- 来自其他 episode 的 chunk。

如果 trust 不随 chunk 改变，说明它只是 contact/stuck classifier，不是 chunk trust。

### Residual BC + online RL

先用 human correction 做 residual BC：

```text
Delta A_target = A_human - A_vla
```

再只在低 trust 窗口做 online RL：

```text
r_t = alpha * Delta Phi_t
    + beta  * success_t
    - gamma * intervention_t
    - lambda * ||Delta A_t||
    - mu    * unsafe_t
```

## 最小实验路线

### 实验 1：trust token 是否预测未来 progress

任务：抽屉拉动或按钮按压。

采集：

- 成功轨迹；
- 夹不住把手继续拉；
- 方向偏了导致抽屉不动；
- 物体放入失败；
- human correction。

指标：

- AUC / PR-AUC；
- trust drop 相对 intervention 的提前量；
- high-trust chunk 的 progress precision；
- false trust rate；
- false alarm rate。

### 实验 2：chunk 绑定验证

同一状态下输入不同候选 chunk：

```text
trust(correct pull) > trust(no-op / wrong direction)
trust(retract after jam) > trust(continue pushing into jam)
```

这是证明“不是普通 stuck detector”的关键。

### 实验 3：gating 因果验证

对比：

- Frozen VLA；
- Frozen VLA + RTC；
- VLA + naive residual；
- VLA + manual gate；
- VLA + progress-only gate；
- VLA + shuffled trust；
- VLA + delayed trust；
- Ours。

如果 learned trust 不能明显优于 shuffled/delayed trust，说明贡献不成立。

### 实验 4：安全边界

必须报告：

- residual activation ratio；
- residual magnitude；
- safety stop；
- 继续硬推比例；
- intervention count；
- task completion time。

否则审稿人会质疑只是牺牲安全换成功率。

## 最大风险

1. **progress signal 噪声大**  
   视觉遮挡、物体不明显、tag 丢失会让 label 失真。

2. **contact 和 failure 混淆**  
   成功任务也需要接触。模型不能把所有接触都判低 trust。

3. **trust gate 退化成 heuristic**  
   如果只用时间阈值、距离阈值就能做到，创新弱。

4. **无反事实**  
   只看执行过的 chunk，无法直接知道另一个 chunk 是否更好。需要 chunk replacement / paired test。

5. **位置控制安全问题**  
   不能把“顶住”作为主要数据来源，只能用低速、软接触、夹取、抽屉位移、按钮位移等可恢复任务。

## 可写成的论文 claim

Claim 1：

> Chunked VLAs expose action sequences but not whether those chunks remain reliable after execution enters contact-rich or constraint-rich regimes. We formulate chunk trust as the probability that a proposed action chunk will produce measurable task progress over a short horizon.

Claim 2：

> Instead of requiring force/torque sensors or manual contact labels, we learn contact-implicit trust from execution traces, progress deltas, human corrections, proprioceptive inconsistencies, and RTC chunk inconsistency.

Claim 3：

> Trust-gated residual control preserves the frozen VLA in high-trust regions while focusing online adaptation on recoverable low-trust chunk windows.

## 推荐标题

```text
Progress-Producing Chunk Trust for Contact-Rich Adaptation of Frozen Vision-Language-Action Policies
```

或：

```text
Contact-Implicit Chunk Trust Tokens for Residual Reinforcement of Frozen Chunked VLAs
```

## 最终判断

这是三条思路里最适合做当前真机主线的一个。

原因不是它最宏大，而是它的变量最可观测、实验最容易闭环、和 Evo-RL / RTC / RLT 的接口最自然。它可以先在抽屉、按钮、夹取放入这类低风险任务上证明“什么时候该信 VLA chunk”，再逐步扩展到更强接触任务。

最重要的边界是：它不能宣称低成本 robot log 等于 contact sensing。它只能宣称：

```text
robot log + visual progress + chunk evidence
can learn whether current VLA chunk remains progress-producing.
```

## References

- RL Tokens / RLT: https://www.pi.website/research/rlt
- OpenPI: https://github.com/Physical-Intelligence/openpi
- Open Sourcing pi0: https://www.pi.website/blog/openpi
- LeRobot RTC docs: https://huggingface.co/docs/lerobot/rtc
- VLA-RL: https://huggingface.co/papers/2505.18719
