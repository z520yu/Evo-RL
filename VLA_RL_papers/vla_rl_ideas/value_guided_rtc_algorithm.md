# Value-Guided RTC: 基于 action chunk 内 Q/value profile 的自适应实时切换

## 0. 一句话目标

把 RTC/real-time chunking 中“按固定时刻或固定锁定前缀切换 chunk”的执行策略，改成“根据当前 chunk 内部的价值曲线、候选新 chunk 的价值、切换代价和推理延迟，自适应决定继续执行还是重规划”的实时控制算法。

建议命名：

- **VG-RTC**: Value-Guided Real-Time Chunking
- **PCC**: Profiled Chunk Critic
- **OVS**: Option-Value Switcher

核心创新不是再训练一个更强 VLA，而是在已有 chunking VLA 和 RTC 执行器之间加一个轻量的价值评估层，使 chunk 边界从固定调度变量变成可学习的决策变量。

## 1. 问题定义

### 1.1 基础设定

机器人以控制频率 `f_ctrl` 执行动作。VLA policy 每次根据观测和语言指令输出一个长度为 `H` 的 action chunk：

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

其中 `a_t` 可以是末端位姿增量、关节位置、夹爪动作或低层控制目标。真实部署中，VLA 推理耗时 `tau_inf` 通常不可忽略，且 `tau_inf` 可能随 GPU 负载、视觉分辨率、语言长度、网络状态变化。

传统 RTC 的基本做法是：

1. 执行当前 chunk 时异步生成下一个 chunk。
2. 对已经必然会执行的 prefix 进行锁定或 inpainting 约束。
3. 在固定规则下把执行权从旧 chunk 交给新 chunk。

这解决了停顿和动作不连续问题，但切换本身通常仍然是固定调度：例如固定执行 `K` 步、固定保留 prefix、候选 chunk ready 后尽快接管，或按固定 overlap 权重混合。

### 1.2 要解决的问题

固定切换会忽略一个关键事实：同一个 action chunk 内部不同时间步的“可靠性”和“任务价值”并不均匀。

典型例子：

- 前半段是稳定接近目标，后半段进入接触，继续执行旧 chunk 的风险快速上升。
- 当前 chunk 在视觉变化后已经 stale，但动作仍然平滑，固定 RTC 可能继续执行过久。
- 候选新 chunk 虽然来自最新观测，但第一步与当前速度/接触状态冲突，立即切换会造成抖动或失败。
- 旧 chunk 的后几步虽然看起来低 value，但在接触建立阶段是必要动作；过早重规划会反复打断进度。

因此我们定义运行时决策：

```text
给定当前观测 o_t、语言目标 g、正在执行的旧 chunk A_old、旧 chunk 内指针 k、
候选新 chunk A_new、推理延迟估计 tau、机器人状态 x_t，
决定：
  continue: 继续执行 A_old[k]
  replan:   切换到 A_new 或 hybrid chunk
  wait:     继续执行若干步，等更好的切换点
  stop/safe: 触发安全策略
```

目标是在保证动作连续和实时性的前提下最大化任务回报：

```text
max E[sum_t gamma^t r_t]
```

其中 `r_t` 不只包含最终成功，也应包含进度、时间成本、碰撞/过力惩罚、动作平滑、接触稳定等部署相关指标。

### 1.3 新问题的形式化

把一个 action chunk 看成一个 temporally extended option。旧 chunk 的剩余 suffix 和候选新 chunk 是两个候选 option：

```text
Option continue:
  执行 A_old[k : k + d - 1]，之后继续由 base VLA/RTC 接管

Option replan:
  从当前或未来某个切换点切到 A_new 或 inpainted hybrid chunk
```

我们需要估计两个 option value：

```text
Q_continue(t, k) = 继续执行当前 chunk suffix 的期望回报
Q_replan(t, k)   = 切到候选 chunk 或 hybrid chunk 的期望回报
```

如果只比较 chunk 级标量 value，容易错过“价值突然下跌”的局部风险。因此要求 critic 输出 action chunk 内部的 **value profile**：

```text
P_old = [q_old_0, q_old_1, ..., q_old_{H-1}]
P_new = [q_new_0, q_new_1, ..., q_new_{H-1}]
```

其中 `q_i` 表示在当前状态、语言目标和候选 chunk 条件下，从 chunk 的第 `i` 个动作附近继续执行的风险调整价值。

## 2. 核心假设

### 2.1 Chunk 内价值 profile 是可学习的

VLA 输出的 action chunk 往往包含隐式阶段结构：接近、对齐、接触、插入、释放、撤退。即使没有显式阶段标签，chunk 内动作、视觉特征、机器人 proprioception、VLA hidden state 和未来成功之间也存在统计关系。

假设存在一个轻量 critic：

```text
C_theta(o_t, g, A, k, meta) -> {
  value_profile: R^H,
  uncertainty_profile: R^H,
  support_profile: R^H
}
```

它能给出每个 action index 附近的任务价值、估计不确定性和离线数据支持度。

### 2.2 切换是否有利不能只看新 chunk 的 value

立即切到新 chunk 的收益需要扣除：

- 动作位置不连续：`||a_new_0 - a_old_k||`
- 速度/加速度不连续：`||Delta a_new_0 - Delta a_old_k||`
- 接触模式不连续：例如已经插入孔内时突然横向重规划
- 推理延迟：候选 chunk ready 时世界已经变化
- OOD 风险：critic 对候选 chunk 的价值可能过估计

所以实际决策应比较：

```text
Q_replan_adjusted - Q_continue_adjusted
```

而不是简单比较 `V(A_new)` 和 `V(A_old)`。

### 2.3 RTC 的平滑机制仍然需要保留

VG-RTC 不应该抛弃 RTC 的 prefix locking、inpainting、velocity guidance 或 blending。价值决策只负责“何时切、切到哪个候选、是否等待”，低层动作连续性仍由 RTC/hybrid chunk 生成器保证。

### 2.4 Critic 必须保守

部署时的候选 chunk 可能来自最新观测，与离线数据有分布差异。critic 一旦系统性高估 replan，执行器会变成高频重规划器，导致抖动和失败。因此 critic 训练要包含 conservative penalty、ensemble uncertainty、OOD support 估计或行为策略约束。

## 3. 系统模块

建议把 VG-RTC 拆成六个模块，方便实现和写论文。

### 3.1 Base VLA policy

输入：

```text
o_t: 多相机 RGB/RGB-D、proprioception、可选 tactile
g:   language instruction 或 task embedding
```

输出：

```text
A_t: length-H action chunk
z_t: 可选 VLA hidden tokens / pooled embedding
```

实现建议：

- 如果能访问 VLA hidden state，把最后几层 action token、视觉 token pooled feature、语言 token pooled feature缓存给 critic。
- 如果不能访问 hidden state，critic 使用观测 encoder、机器人状态和 action chunk 自己训练。
- base VLA 可以冻结；VG-RTC 初版只训练 critic 和 switcher。

### 3.2 RTC/hybrid chunk generator

负责生成可平滑执行的候选 chunk。

输入：

```text
A_old, k, o_now, g, locked_prefix_len, tau_est
```

输出：

```text
A_new_raw:  最新观测下 VLA 输出的新 chunk
A_hybrid:   prefix-locked / inpainted / blended 后的候选执行 chunk
```

`A_hybrid` 的构造可以从简单到复杂：

- 简单 blending：前 `B` 步从旧动作平滑过渡到新动作。
- RTC inpainting：锁定已经承诺执行的 prefix，让生成模型补全后续动作。
- Contact-aware bridge：接触状态下加大切换平滑约束，避免突然改变力方向。

### 3.3 Profiled Chunk Critic

核心模块。输入当前状态、语言、chunk、当前 pointer、chunk age、延迟估计等，输出 value profile：

```text
P, U, S = C_theta(o_now, g, A, k, meta)

P[i]: 从 chunk index i 继续执行的风险调整价值
U[i]: epistemic/aleatoric uncertainty
S[i]: support score，表示该状态-动作片段是否接近训练分布
```

推荐接口：

```python
critic.forward(
    obs=o_now,
    instruction=g,
    chunk=A,
    pointer=k,
    meta={
        "chunk_age": age,
        "latency_est": tau_est,
        "obs_delta": obs_delta,
        "contact_mode": contact_mode,
        "ee_pose": ee_pose,
        "gripper": gripper_state,
    },
) -> ChunkValueProfile
```

其中 `ChunkValueProfile` 至少包含：

```text
value:       shape [H]
uncertainty: shape [H]
support:     shape [H]
progress:    optional shape [H]
safety_risk: optional shape [H]
```

### 3.4 Switch evaluator

把 critic profile 转成可比较的 option value：

```text
Q_continue = F_continue(P_old, U_old, S_old, k, meta)
Q_replan   = F_replan(P_hybrid, U_new, S_new, switch_cost, tau_est, meta)
```

这里的 `F` 不应该只是取均值。推荐 risk-sensitive aggregation：

```text
Agg(P[k:k+M]) =
  alpha * mean(P[k:k+M])
  + beta * min(P[k:k+M])
  + (1 - alpha - beta) * P[k]
```

其中 `min` 或 CVaR 项用于发现 chunk 内部的价值悬崖。

### 3.5 Safety gate

不让 critic 独自承担安全责任。任何时候以下条件优先于 value switch：

- 距离碰撞阈值过近。
- 力/力矩超过安全阈值。
- 夹爪或末端状态与动作意图冲突。
- 视觉追踪目标丢失。
- action 超出机器人动态约束。

输出：

```text
safe_stop / hold_position / retreat / human_intervention
```

### 3.6 Async executor

维护两个线程或进程：

- Control loop：高频执行动作、计算 critic、做切换。
- Inference loop：低频异步生成候选 chunk，更新候选队列。

关键状态：

```text
active_chunk
active_pointer
active_generation_time
candidate_chunk_queue
last_switch_time
cooldown_counter
latency_ema
```

## 4. Chunk critic 如何训练

### 4.1 数据来源

至少需要三类数据。

**第一类：原始 demonstration 数据**

格式：

```text
(o_t, g, a_t, r_t, done)
```

从连续轨迹中滑窗构造 chunk：

```text
A_t = [a_t, ..., a_{t+H-1}]
```

优点是稳定、成功率高。缺点是缺少失败片段和 replan counterfactual。

**第二类：base VLA/RTC rollout 数据**

在仿真或真实机器人上运行 base VLA + 固定 RTC，记录：

```text
o_t, g, A_old, k, A_new, executed_action, switch_decision, latency, reward, done
```

训练时要故意注入：

- 随机推理延迟。
- 观测延迟。
- 目标轻微移动。
- 初始位姿扰动。
- 固定切换点扰动。
- 早切/晚切的对照策略。

这样 critic 才会看到“继续旧 chunk 失败”和“过早 replan 失败”的例子。

**第三类：counterfactual chunk 数据**

从同一状态构造多个候选 chunk：

```text
A_demo:     真实成功动作 chunk
A_base:     base VLA 输出
A_stale:    旧观测生成的 chunk
A_perturb:  加噪声或时间错位的 chunk
A_hybrid:   旧 chunk prefix + 新 chunk suffix
A_bad_mix:  动作平滑但语义错误的 chunk
```

这些样本可用于 ranking loss 和 conservative loss，即使没有真实执行每个候选，也能让 critic 学到相对偏好。

### 4.2 奖励设计

最小可行版可以只使用 episode success：

```text
r_t = 1[success at episode end] - lambda_time
```

论文版建议用组合奖励：

```text
r_t =
  w_success  * success_delta
  + w_prog   * progress_delta
  - w_time   * Delta t
  - w_force  * force_violation
  - w_col    * collision
  - w_jerk   * ||a_t - 2a_{t-1} + a_{t-2}||^2
  - w_switch * switch_event_cost
```

`progress_delta` 可来自：

- 任务专用距离：目标物体到目标区域距离、插入深度、姿态误差。
- 学习到的 success/progress classifier。
- 人工阶段标签。
- VLM/VLA embedding 的目标相似度，但要谨慎校准。

### 4.3 Critic 输出定义

推荐训练两个相关但不同的量：

1. **Chunk-start value**

```text
Q_theta(o_t, g, A_t, k)
```

表示从当前 pointer `k` 开始执行该 chunk suffix，并在 chunk 结束后回到默认 VLA policy 的期望回报。

2. **Profile value**

```text
P_theta(o_t, g, A_t)[i], i = 0 ... H-1
```

表示“如果当前执行位置位于 chunk index `i`，继续执行该 chunk suffix”的价值。

实际实现可以共用 encoder，输出长度为 `H` 的向量：

```text
h = Encoder(o_t, g, A_t, meta)
P = MLP_or_TransformerHead(h)  # [H]
```

训练时只对可监督的 index 计算 loss，未覆盖 index mask 掉。

### 4.4 TD 目标

对真实执行轨迹中的 chunk suffix，使用 n-step TD：

```text
y_{t,k}^{(n)} =
  sum_{j=0}^{n-1} gamma^j r_{t+j}
  + gamma^n V_bar(o_{t+n}, g)
```

critic loss：

```text
L_td =
  E[(Q_theta(o_t, g, A_t, k) - y_{t,k}^{(n)})^2]
```

对 profile 中每个 index `i`，如果轨迹中有对应后续状态：

```text
y_{t,i} =
  sum_{j=i}^{i+n-1} gamma^{j-i} r_{t+j}
  + gamma^n V_bar(o_{t+i+n}, g)
```

```text
L_profile =
  E_i[mask_i * (P_theta(o_t, g, A_t)[i] - y_{t,i})^2]
```

`V_bar` 可以是 target critic 对 base VLA chunk 的估计：

```text
A_next ~ pi_vla(o_{t+n}, g)
V_bar(o_{t+n}, g) = Agg(C_bar(o_{t+n}, g, A_next))
```

如果不想在训练中频繁调用 VLA，可以预先缓存 `A_next` 和 VLA hidden state。

### 4.5 Ranking loss

只靠 TD，critic 对未执行候选 chunk 的排序可能不稳。加入 pairwise ranking：

```text
A_pos: 成功 demo chunk 或高回报 rollout chunk
A_neg: stale/perturbed/bad_mix/失败 rollout chunk
```

```text
L_rank =
  max(0, margin - Score(o,g,A_pos) + Score(o,g,A_neg))
```

其中：

```text
Score(o,g,A) = Agg(P_theta(o,g,A))
```

ranking loss 的作用：

- 教 critic 区分“动作平滑但过时”和“语义正确且及时”。
- 教 critic 惩罚接触阶段的错误 replan。
- 缓解 sparse reward 下 profile 学不出来的问题。

### 4.6 Conservative / support loss

为避免高估 OOD candidate，使用以下至少一种方法。

**CQL 风格保守项**

对同一状态采样多个候选 chunk：

```text
A_sample ~ {VLA samples, stale chunks, perturbed chunks, random chunks}
```

惩罚未被数据支持的高 Q：

```text
L_cql =
  logsumexp_A Score(o,g,A_sample) - Score(o,g,A_data)
```

**IQL/expectile 风格**

不显式最大化 OOD 动作，只学习数据分布内的 value，并让 switcher 只比较来自 VLA/RTC 的候选 chunk。

**Support head**

训练一个二分类或密度 head：

```text
S_theta(o,g,A,i) = P((o,g,A_i) in data support)
```

runtime 中如果 support 低，则降低 `Q_replan`：

```text
Q_adjusted = Q_raw - lambda_ood * (1 - S)
```

### 4.7 Uncertainty 估计

推荐使用 critic ensemble：

```text
C_1, ..., C_N
```

均值作为 value，不同 critic 之间方差作为 epistemic uncertainty：

```text
P_mean = mean_n P_n
U = std_n P_n
```

运行时使用风险调整 value：

```text
P_safe = P_mean - lambda_unc * U
```

MVP 可以用 dropout MC 或 bootstrap ensemble，论文版建议用 3-5 个轻量 head。

### 4.8 总损失

```text
L =
  L_td
  + lambda_profile * L_profile
  + lambda_rank    * L_rank
  + lambda_cql     * L_cql
  + lambda_smooth  * L_profile_smooth
  + lambda_support * L_support
```

其中 profile smoothness 约束为：

```text
L_profile_smooth = sum_i ||P[i+1] - P[i]||_Huber
```

注意不要过强，否则会抹掉真正的价值 cliff。推荐只用小权重，或在接触/阶段边界处关闭平滑约束。

## 5. Q_continue 如何计算

### 5.1 基础输入

当前时刻：

```text
o_now
g
A_old
k              # 当前执行到旧 chunk 的第 k 个动作
age            # 当前 chunk 从生成到现在经过多久
tau_est        # 推理延迟估计
contact_mode
obs_delta      # 当前观测与生成旧 chunk 时观测的差异
```

critic 输出：

```text
P_old, U_old, S_old = C_theta(o_now, g, A_old, k, meta)
```

### 5.2 Profile 聚合

只看 `P_old[k]` 不够，因为旧 chunk 后面几步可能即将掉入低价值区。定义一个短窗口 `M`：

```text
I = [k, k+1, ..., min(k+M-1, H-1)]
```

风险调整 profile：

```text
P_safe_old[i] =
  P_old[i]
  - lambda_unc * U_old[i]
  - lambda_ood * (1 - S_old[i])
```

聚合：

```text
V_suffix_old =
  alpha * mean_{i in I}(P_safe_old[i])
  + beta  * min_{i in I}(P_safe_old[i])
  + eta   * P_safe_old[k]
```

其中 `alpha + beta + eta = 1`。建议初始：

```text
alpha = 0.4
beta  = 0.4
eta   = 0.2
```

`beta` 控制对即将出现的 failure cliff 的敏感性。

### 5.3 Staleness penalty

旧 chunk 越 stale，继续执行越危险：

```text
C_stale =
  w_age * age
  + w_obs * ||phi(o_now) - phi(o_gen_old)||
  + w_ee  * ||x_now - rollout_kinematics(A_old[0:k])||
```

实际中 `rollout_kinematics` 不必很精确，可用执行器记录的实际末端位姿和旧 chunk 预测末端位姿差。

### 5.4 Tail penalty

接近 chunk 尾部时，如果没有 ready candidate，继续执行的可靠性下降：

```text
C_tail = w_tail * max(0, k + tau_est_steps - H + H_guard)
```

其中 `H_guard` 表示至少要预留多少步给下一次推理和 bridge。

### 5.5 最终 Q_continue

```text
Q_continue =
  V_suffix_old
  - C_stale
  - C_tail
  - C_safety_old
```

`C_safety_old` 来自 safety risk head 或外部安全检查器。

## 6. Q_replan 如何计算

### 6.1 候选 chunk 类型

运行时可能有三种候选：

```text
A_new_ready: 已经由异步 VLA 生成完毕的新 chunk
A_new_pending: 正在生成，预计 tau_remaining 后 ready
A_fast: 小模型/缓存策略快速给出的临时候选
```

MVP 只处理 `A_new_ready`。论文增强版和最终版应考虑 pending candidate，因为真正的切换决策是“现在切、等 d 步再切、还是不切”。

### 6.2 Hybrid chunk

直接切到 `A_new` 可能不连续。构造：

```text
A_hybrid(d) =
  [
    A_old[k : k+d-1],
    bridge(A_old[k+d-1], A_new[0:B]),
    A_new[B:]
  ]
```

`d` 是等待步数。`d=0` 表示立即切。`B` 是 bridge 长度。

如果使用 flow/diffusion VLA，`A_hybrid` 应优先由 RTC inpainting 产生：锁定已经承诺执行的 prefix，让模型补全后续；简单线性 blending 作为 fallback。

### 6.3 新 chunk value

```text
P_new, U_new, S_new = C_theta(o_now, g, A_hybrid(d), pointer=d, meta)
```

风险调整：

```text
P_safe_new[i] =
  P_new[i]
  - lambda_unc * U_new[i]
  - lambda_ood * (1 - S_new[i])
```

聚合：

```text
V_new =
  alpha * mean(P_safe_new[d:d+M])
  + beta  * min(P_safe_new[d:d+M])
  + eta   * P_safe_new[d]
```

### 6.4 Switch cost

切换代价要显式进入 `Q_replan`。

动作连续性：

```text
C_pos = ||a_entry_new - a_entry_old||_W^2
```

速度连续性：

```text
C_vel = ||(a_entry_new - a_prev) - (a_entry_old - a_prev)||_Wv^2
```

加速度/jerk：

```text
C_jerk = ||a_new_1 - 2a_new_0 + a_prev||^2
```

接触状态惩罚：

```text
C_contact =
  1[contact_mode in critical_contact] * contact_switch_penalty
```

语义阶段惩罚：

如果有 phase classifier，可在插入、拧紧、夹取闭合等阶段提高切换 margin：

```text
C_phase = phase_risk(g, o_now, k) * phase_switch_penalty
```

总切换代价：

```text
C_switch =
  w_pos * C_pos
  + w_vel * C_vel
  + w_jerk * C_jerk
  + w_contact * C_contact
  + w_phase * C_phase
```

### 6.5 Delay cost

如果候选还没 ready，等待 `d` 步后才能切：

```text
C_delay = w_delay * d
```

更准确的形式是把等待期间旧 chunk 的价值也算进去：

```text
Q_replan(d) =
  R_hat_old(k, d)
  + gamma^d * V_new_after_wait(d)
  - C_switch(d)
  - C_delay(d)
```

其中：

```text
R_hat_old(k, d) = sum_{j=0}^{d-1} gamma^j r_hat_old[k+j]
```

MVP 可以没有 `r_hat_old`，用 `P_old[k] - P_old[k+d]` 的差值近似等待代价。

### 6.6 最终 Q_replan

对单个候选 chunk：

```text
Q_replan(d) =
  V_new
  - C_switch(d)
  - C_delay(d)
  - C_safety_new
```

对多个等待步数选择最优切换点：

```text
d_star = argmax_{d in D} Q_replan(d)
Q_replan = Q_replan(d_star)
```

其中：

```text
D = {0, 1, ..., min(W_switch, H-k-1)}
```

`W_switch` 不宜过大，否则每个控制步计算过重。可以只评估 `{0, 1, 2, 4, 8}` 或根据控制频率设置 100-300 ms 的窗口。

## 7. 切换规则

### 7.1 基础规则

```text
if safety_gate triggers:
    execute safe policy
elif candidate not available and tail risk high:
    slow_down / hold / request fast candidate
elif Q_replan > Q_continue + margin:
    switch to candidate at d_star
else:
    continue current chunk
```

### 7.2 动态 margin

固定 margin 容易在不确定时过度切换。推荐：

```text
margin =
  m0
  + m_unc * U_replan
  + m_contact * 1[critical_contact]
  + m_cooldown * 1[recently_switched]
  - m_drop * 1[value_cliff_detected]
```

其中 value cliff：

```text
value_cliff_detected =
  min(P_old[k:k+M]) < cliff_threshold
  or P_old[k] - min(P_old[k:k+M]) > cliff_gap
```

当旧 chunk 即将掉入低价值区时，降低 margin，允许更早重规划。

### 7.3 Hysteresis 和 cooldown

避免在两个 chunk 之间来回切：

```text
min_dwell_steps: 每次切换后至少执行 N 步
cooldown_steps:  切换后若干步提高 replan margin
```

除非 safety gate 触发，否则 cooldown 期间不允许 replan。

### 7.4 Contact-aware rule

接触任务中，动作平滑不等于物理安全。建议规则：

```text
if contact_mode == "free_space":
    allow low switch cost
elif contact_mode == "pre_contact":
    allow replan if Q gain large
elif contact_mode == "stable_contact":
    require higher margin and bridge length
elif contact_mode == "insertion_or_constraint":
    only allow replan to contact-consistent candidate
```

contact-consistent 可由以下条件判断：

- 末端速度方向与接触法向不冲突。
- 夹爪状态不突然开合。
- 候选动作不会显著增大横向力。
- phase classifier 认为候选还在同一阶段。

### 7.5 Emergency drop rule

当旧 chunk profile 出现明显悬崖，即使候选 value 不高，也应脱离旧 chunk：

```text
if min(P_old[k:k+M]) < hard_fail_threshold:
    if candidate_safe:
        switch
    else:
        safe_hold_or_retreat
```

这是为了处理“继续执行明显会撞/掉/越界”的场景。

## 8. 伪代码

### 8.1 Runtime VG-RTC

```python
def value_guided_rtc_loop(vla, critic, rtc_generator, safety_gate, env):
    obs = env.get_observation()
    active_chunk = vla.sample_chunk(obs, instruction)
    active_pointer = 0
    active_gen_obs = obs
    active_gen_time = now()

    candidate = None
    inference_worker.request(obs, instruction, prefix=None)

    while not env.done():
        obs_now = env.get_observation()
        robot_state = env.get_robot_state()
        latency_est = inference_worker.latency_ema()

        if inference_worker.has_result():
            raw_new = inference_worker.pop_result()
            candidate = rtc_generator.make_candidate(
                old_chunk=active_chunk,
                pointer=active_pointer,
                new_chunk=raw_new,
                obs_now=obs_now,
                robot_state=robot_state,
            )

            inference_worker.request(
                obs=obs_now,
                instruction=instruction,
                prefix=active_chunk[active_pointer:],
            )

        safe_action = safety_gate.check(obs_now, robot_state)
        if safe_action is not None:
            env.step(safe_action)
            continue

        old_profile = critic.forward(
            obs=obs_now,
            instruction=instruction,
            chunk=active_chunk,
            pointer=active_pointer,
            meta={
                "chunk_age": now() - active_gen_time,
                "latency_est": latency_est,
                "obs_delta": obs_distance(obs_now, active_gen_obs),
                "contact_mode": estimate_contact(robot_state),
            },
        )

        q_continue = compute_q_continue(old_profile, active_pointer)

        switch = False
        d_star = None
        next_chunk = None

        if candidate is not None and cooldown_expired():
            best = None
            for d in candidate_switch_delays(active_pointer):
                hybrid = rtc_generator.hybridize(
                    old_chunk=active_chunk,
                    pointer=active_pointer,
                    candidate=candidate,
                    delay_steps=d,
                )

                new_profile = critic.forward(
                    obs=obs_now,
                    instruction=instruction,
                    chunk=hybrid,
                    pointer=d,
                    meta={
                        "chunk_age": 0.0,
                        "latency_est": latency_est,
                        "obs_delta": 0.0,
                        "contact_mode": estimate_contact(robot_state),
                    },
                )

                q_replan_d = compute_q_replan(
                    old_chunk=active_chunk,
                    hybrid_chunk=hybrid,
                    old_pointer=active_pointer,
                    switch_delay=d,
                    new_profile=new_profile,
                    robot_state=robot_state,
                )

                if best is None or q_replan_d > best.q:
                    best = SwitchCandidate(q=q_replan_d, d=d, chunk=hybrid)

            margin = dynamic_margin(
                old_profile=old_profile,
                new_profile=best.profile,
                contact_mode=estimate_contact(robot_state),
                recently_switched=not cooldown_expired(),
            )

            if best.q > q_continue + margin:
                switch = True
                d_star = best.d
                next_chunk = best.chunk

        if switch and d_star == 0:
            active_chunk = next_chunk
            active_pointer = 0
            active_gen_obs = obs_now
            active_gen_time = now()
            reset_cooldown()

        action = active_chunk[active_pointer]
        env.step(action)
        active_pointer += 1

        if switch and d_star is not None:
            d_star -= 1
            if d_star == 0:
                active_chunk = next_chunk
                active_pointer = 0
                active_gen_obs = obs_now
                active_gen_time = now()
                reset_cooldown()

        if active_pointer >= len(active_chunk):
            if candidate is not None:
                active_chunk = candidate.chunk
                active_pointer = 0
                active_gen_obs = obs_now
                active_gen_time = now()
            else:
                env.step(safe_hold_action(robot_state))
```

实现时可以简化：

- MVP 不评估多个 `d`，只比较立即切换。
- 论文版保留多个 `d`，但 `D` 取很小集合。
- 控制循环中 critic forward 要足够快；可只在每 `N` 个 control tick 评估一次，其余 tick 复用上次结果。

### 8.2 Critic 训练伪代码

```python
def train_profiled_chunk_critic(dataset, vla, critic, target_critic):
    replay = build_chunk_replay(dataset, horizon=H)

    for step in range(num_updates):
        batch = replay.sample(batch_size)

        # data chunk
        obs = batch.obs
        instr = batch.instruction
        chunk = batch.chunk
        pointer = batch.pointer
        rewards = batch.rewards
        next_obs = batch.next_obs

        with no_grad():
            next_chunk = batch.cached_next_chunk
            if next_chunk is None:
                next_chunk = vla.sample_chunk(next_obs, instr)

            next_profile = target_critic.forward(
                obs=next_obs,
                instruction=instr,
                chunk=next_chunk,
                pointer=0,
                meta=batch.next_meta,
            )
            next_v = aggregate_profile(next_profile)
            td_target = n_step_return(rewards, next_v, gamma)

        profile = critic.forward(
            obs=obs,
            instruction=instr,
            chunk=chunk,
            pointer=pointer,
            meta=batch.meta,
        )

        loss_td = mse(profile.value_at(pointer), td_target)
        loss_profile = masked_profile_td_loss(profile, batch.profile_targets)

        pos_chunk, neg_chunk = make_counterfactual_pair(batch)
        pos_score = aggregate_profile(critic.forward(obs, instr, pos_chunk, pointer, batch.meta))
        neg_score = aggregate_profile(critic.forward(obs, instr, neg_chunk, pointer, batch.meta))
        loss_rank = relu(rank_margin - pos_score + neg_score).mean()

        sampled_chunks = sample_chunks_for_cql(batch, vla)
        loss_cql = conservative_chunk_loss(critic, obs, instr, chunk, sampled_chunks)

        loss = (
            loss_td
            + lambda_profile * loss_profile
            + lambda_rank * loss_rank
            + lambda_cql * loss_cql
        )

        critic.optimize(loss)
        soft_update(target_critic, critic)
```

## 9. 训练流程

### 9.1 Stage A: 数据准备

1. 收集或加载 demonstration。
2. 用滑窗生成长度 `H` 的 chunks。
3. 对每个 chunk 记录：

```text
obs at generation time
obs at execution time
instruction
action chunk
pointer k
robot state
contact estimate
reward / success
episode id
timestamp
```

4. 如果能运行 base VLA，则离线缓存：

```text
VLA hidden tokens
base VLA chunk
stale chunk
sampled alternative chunks
```

### 9.2 Stage B: 训练 progress/success reward model

如果任务只有 sparse success，先训练一个轻量 progress model：

```text
G_phi(o_t, g) -> progress scalar
```

监督来源：

- 成功轨迹的时间归一化进度伪标签。
- 人工阶段标签。
- 物体位姿到目标位姿距离。
- sparse success 的 temporal contrastive learning。

reward shaping：

```text
r_prog_t = G_phi(o_{t+1}, g) - G_phi(o_t, g)
```

这一步不是必须，但能显著提高 profile 的可学习性。

### 9.3 Stage C: 训练 profiled chunk critic

训练 critic 的推荐顺序：

1. 先用 demonstration 做 Monte Carlo / n-step TD 预训练。
2. 加入失败 rollout 和延迟扰动数据。
3. 加入 counterfactual ranking。
4. 加入 conservative loss 和 ensemble。
5. 在 held-out rollout 上校准 value 与真实成功率的相关性。

关键指标：

```text
profile AUC: 低 value index 是否对应失败/危险片段
pairwise accuracy: A_pos 是否高于 A_neg
calibration: predicted Q bin 与 empirical success rate 是否一致
switch oracle agreement: 是否接近离线 oracle 切换标签
```

### 9.4 Stage D: 训练或调参 switcher

MVP 可以手动设置 `alpha, beta, eta, margin, lambda_switch`。

论文版可以学习一个小 switcher：

```text
S_psi(profile_old, profile_new, meta) -> switch probability / margin
```

但建议不要让 switcher 完全黑盒化。更可发表且可解释的方式是：

- critic 学 value profile；
- switcher 使用显式 option-value 公式；
- 只有少数权重通过 validation search 或离线 RL 学习。

离线 oracle switch 标签可这样构造：

```text
对同一时刻比较 continue rollout 和 replan rollout 的最终回报。
若 replan return - continue return > threshold，则 label= switch。
```

真实机器人上无法大量采集 paired rollout 时，可在仿真中生成 oracle，在真实上只做校准。

### 9.5 Stage E: 闭环微调

部署初期记录所有决策：

```text
old_profile
new_profile
Q_continue
Q_replan
decision
outcome
human_intervention
force/collision
```

周期性加入 replay 更新 critic。真实机器人上建议只更新 critic 和 switch parameters，不直接在线更新大 VLA。

## 10. 部署流程

### 10.1 启动

1. 加载 base VLA。
2. 加载 critic ensemble 和 normalization stats。
3. 初始化 RTC generator。
4. 初始化 safety gate。
5. 进行一次 warm-up inference，得到首个 active chunk。

### 10.2 控制循环

每个 control tick：

1. 读取最新观测和机器人状态。
2. 更新 active chunk pointer。
3. 检查 safety gate。
4. 如果有候选 chunk ready，构造 hybrid candidates。
5. 用 critic 评估旧 suffix 和候选 chunk。
6. 计算 `Q_continue` 和 `Q_replan`。
7. 应用 hysteresis/cooldown/contact-aware rules。
8. 执行动作。
9. 异步请求下一次 VLA inference。

### 10.3 工程优化

- critic 输入的图像 encoder 尽量复用 VLA vision features。
- critic 只在 5-20 Hz 运行，低层控制可在 50-200 Hz 插值。
- candidate 只保留最新一个或 top-K；过时候选直接丢弃。
- 记录推理耗时 EMA，用于 `tau_est`。
- 若 GPU 忙，critic 可用小 MLP/Transformer head 放 CPU 或低优先级 GPU stream。
- 对 action chunk 做标准化，critic 和 switch cost 使用同一尺度。

### 10.4 安全 fallback

当以下任一情况发生：

```text
candidate missing and active_pointer near H
critic uncertainty too high
support too low
safety risk high
VLA inference timeout
```

执行：

```text
safe_hold -> slow_retreat -> human_intervention
```

不要让系统在 chunk 尾部盲目重复最后一个动作。

## 11. 候选算法版本

### 11.1 版本一：最小可行版 Value-Threshold RTC

目标：两周内可实现，能验证“value profile 比固定切换更好”。

特点：

- base VLA 冻结。
- 训练一个单 critic，输出 `P[i]`。
- 不做多步切换点搜索，只比较“继续当前 chunk”与“立即切到 ready candidate”。
- `Q_continue` 使用旧 chunk suffix 的 mean/min 聚合。
- `Q_replan` 使用新 chunk 前 `M` 步 value 减去简单动作不连续 cost。
- 手写 hysteresis 和 cooldown。

公式：

```text
Q_continue =
  Agg(P_old[k:k+M]) - w_age * age - w_tail * tail_risk

Q_replan =
  Agg(P_new[0:M])
  - w_pos * ||A_new[0] - A_old[k]||^2
  - w_unc * U_new

switch iff Q_replan > Q_continue + m0
```

训练：

- 用 demonstration + base rollout。
- TD + ranking loss。
- 可先不做 CQL，只用候选来自 base VLA 限制 OOD。

优点：

- 实现简单。
- 便于和固定 RTC 做直接对比。
- 能快速得到 profile 可视化图。

缺点：

- 不处理 candidate pending。
- 不优化未来切换点。
- 容易受 critic overestimate 影响。

适合实验：

- LIBERO / Kinetix / 简单真实抓取任务。
- 延迟扰动下比较 success、cycle time、jerk、switch count。

### 11.2 版本二：论文增强版 Profile-Ranking VG-RTC

目标：形成较完整论文方法。

新增：

- critic ensemble 输出 uncertainty。
- counterfactual chunk augmentation。
- ranking loss + conservative loss。
- contact-aware switch cost。
- value cliff detection。
- 动态 margin。
- 小集合 `D={0,1,2,4}` 的切换等待搜索。

公式：

```text
d_star = argmax_{d in D} [
  gamma^d * Agg(P_hybrid_d[d:d+M])
  - C_switch(d)
  - C_delay(d)
  - C_unc(d)
  - C_ood(d)
]

switch iff
  Q_replan(d_star) > Q_continue + margin(meta)
```

训练：

- 加入延迟扰动 rollout。
- 构造 `A_stale, A_perturb, A_hybrid, A_bad_mix`。
- 使用 pairwise ranking 监督候选排序。
- 使用 CQL/support head 降低 OOD overestimation。

优点：

- 创新点清楚：chunk value profile + option-value switch。
- 可解释性强：能画出价值曲线和切换点。
- 与 RTC、StreamingVLA、RLT 都是正交关系。

缺点：

- 工程复杂度中等。
- 需要较多负样本和扰动 rollout。
- reward/progress 设计影响大。

适合论文主版本。

### 11.3 版本三：最终推荐版 Option-Value Guided RTC

目标：作为最强系统和论文最终算法。

核心思想：

把“继续旧 chunk”和“切到新 chunk”都看成 temporally extended options，由 critic 估计 latency-aware option value，switcher 选择最优 option 和最优切换时间。

新增：

- `Q_continue(d)` 与 `Q_replan(d)` 同时建模。
- 对 pending candidate 做等待价值估计。
- 使用 RTC inpainting 生成多个 hybrid candidate。
- 使用 RL token 或 VLA hidden token 作为 critic state。
- 对接触阶段使用 phase/contact-conditioned switch cost。
- 运行时根据 uncertainty 自动降级到保守 RTC。

决策：

```text
For d in D:
  option_continue_d = execute old suffix for d steps
  option_replan_d   = execute old suffix for d steps, then switch to hybrid new

Q_replan(d) =
  R_old_prefix(k,d)
  + gamma^d * V(hybrid_new_after_d)
  - C_switch(d)
  - C_latency(d)
  - C_risk(d)

d_star = argmax_d Q_replan(d)

if max_d Q_replan(d) > Q_continue_now + margin:
    schedule switch at d_star
else:
    continue
```

推荐架构：

```text
VLA frozen backbone
  -> hidden/token cache
  -> Profiled Chunk Critic ensemble
  -> Option-Value Switcher
  -> RTC inpainting/hybrid generator
  -> safety gate
```

为什么推荐：

- 保留 RTC 的动作连续性优势。
- 解决固定切换无法感知任务价值的问题。
- 相比直接 RL fine-tune VLA，样本效率更高、部署风险更低。
- 可以自然扩展到 StreamingVLA/RLT：StreamingVLA 负责异步流水线，RLT 提供更好的 state token 和 residual actor，VG-RTC 负责 runtime option selection。

## 12. 与 CO-RFT / StreamingVLA / RLT 的区别

### 12.1 与 CO-RFT 的区别

CO-RFT 的核心是把 action chunk 纳入 offline RL fine-tuning，用 chunked TD learning 优化 VLA policy，使 policy 本身在目标任务上更强。

VG-RTC 的核心不是直接优化 VLA policy，而是学习一个 chunk critic 和 runtime switcher，决定当前 chunk suffix 与候选 replan chunk 之间如何切换。

关键差异：

```text
CO-RFT:
  优化对象：VLA policy 参数
  决策层级：训练期策略改进
  重点问题：用少量 demos 做 chunked offline RL fine-tuning
  部署行为：policy 输出更优 chunk，但切换仍可固定

VG-RTC:
  优化对象：critic + switcher，可冻结 VLA
  决策层级：部署期 chunk option selection
  重点问题：实时执行中何时继续、何时重规划、何时等待
  部署行为：chunk 边界随 value profile 自适应变化
```

二者可组合：用 CO-RFT 训练更好的 chunk policy，再用 VG-RTC 做 value-guided runtime switching。CO-RFT 的 chunked TD 也可作为 VG-RTC critic 训练的基础。

### 12.2 与 StreamingVLA 的区别

StreamingVLA 关注 VLA pipeline 的异步化和延迟隐藏，例如让观察、动作生成、执行流水化，并通过 action flow matching 或 adaptive observation 改善流畅性。

VG-RTC 关注的是切换决策本身：即使系统已经 streaming，仍然需要决定旧动作流什么时候被新动作流接管。

关键差异：

```text
StreamingVLA:
  目标：降低端到端 latency，减少等待和 halting
  方法：流水化 observation/action/execution，改变生成方式或 observation 触发
  切换依据：主要围绕时序效率和执行流畅性

VG-RTC:
  目标：根据任务价值选择 chunk 边界
  方法：学习 chunk 内 value profile，比较 continue/replan option value
  切换依据：任务成功概率、进度、风险、切换代价、延迟共同决定
```

二者可组合：StreamingVLA 提供更快候选流，VG-RTC 从候选流中选择何时接管。

### 12.3 与 RLT 的区别

RLT/RL Token 的核心是让冻结 VLA 暴露紧凑 token，再在其上训练轻量 actor-critic，用在线 RL 改善高精度阶段动作。

VG-RTC 与 RLT 的区别：

```text
RLT:
  重点：学习 residual actor / small RL policy，提高关键阶段动作质量
  critic 用途：训练 actor，评估 RL policy
  改变动作：通常会输出动作修正或替代动作

VG-RTC:
  重点：学习 chunk profile critic，用于 runtime continue/replan 切换
  critic 用途：评估旧 chunk suffix 与候选 chunk 的 option value
  改变动作：主要改变切换时机和 hybrid chunk 选择，不一定改 VLA 动作
```

二者可强组合：用 RLT token 作为 VG-RTC critic 的状态表示；RLT residual actor 负责精细动作修正，VG-RTC 负责何时让新 chunk 或 residual policy 接管。

## 13. 潜在失败点和解决方案

### 13.1 Critic 高估 replan，导致频繁切换

表现：

- switch count 很高。
- 动作 jerk 上升。
- 接触阶段反复退出/进入。
- 成功率下降但 predicted Q 很高。

解决：

- 使用 ensemble uncertainty，`Q = mean - lambda * std`。
- 加 CQL/support penalty。
- 提高 cooldown 和 contact margin。
- 在训练数据中加入 bad replan negative examples。
- 使用 switch cost 显式惩罚速度/接触不连续。

### 13.2 Critic 过于保守，退化成固定 RTC

表现：

- 几乎不切换。
- 旧 chunk stale 后仍继续执行。
- value profile 没有明显区分度。

解决：

- 增加失败 rollout 和 stale chunk 负样本。
- 增加 progress reward，避免只有 sparse success。
- 降低 `m0` 或 `lambda_unc`。
- 使用 value cliff rule：旧 chunk 低于 hard threshold 时强制脱离。

### 13.3 Sparse reward 下 profile 学不出来

表现：

- profile 只在 episode 末端有信号。
- chunk 内 `P[i]` 近似常数。

解决：

- 训练 progress model 做 dense shaping。
- 用阶段标签或自动分段。
- 用 pairwise ranking：成功 chunk 高于失败/stale chunk。
- 用 temporal contrastive loss 学“更接近成功”的排序。

### 13.4 切换点正确但动作不连续

表现：

- `Q_replan` 高，但真实机器人有 jerk、力峰值或夹爪错误。

解决：

- 不直接切 raw new chunk，必须经过 RTC inpainting/hybrid。
- switch cost 加速度和 jerk 项。
- contact-aware bridge length。
- 对低层控制加速度/速度限幅。
- 训练 critic 时加入 hybrid chunk，而不是只看 raw chunk。

### 13.5 候选 chunk 已经过时

表现：

- candidate ready 时环境又变了，切过去反而失败。

解决：

- candidate 记录 generation timestamp 和 obs fingerprint。
- `Q_replan` 中加入 candidate age / obs delta penalty。
- 超过 TTL 的 candidate 丢弃。
- 允许 fast re-evaluation：用 critic 在最新 obs 下重评估旧 candidate。

### 13.6 接触阶段价值判断错误

表现：

- 插入、拧紧、拉链、扎带等任务中，视觉变化小但力学状态关键。

解决：

- critic 输入 force/torque、tactile、gripper current。
- contact mode 作为 meta。
- contact phase 单独训练 switch margin。
- 对稳定接触阶段只允许 contact-consistent candidate。

### 13.7 计算开销破坏实时性

表现：

- critic 或多候选搜索导致控制 loop 掉帧。

解决：

- 复用 VLA features。
- critic head 小型化。
- 每 `N` 个 tick 评估一次。
- `D` 只取少量候选等待步。
- candidate top-K 限制。
- 异步计算 profile，控制 loop 使用最近一次结果。

### 13.8 离线训练与真实部署分布不同

表现：

- 仿真上有效，真实机器人误判。

解决：

- 真实部署只先 shadow mode：记录 VG-RTC 会怎么切，但不执行。
- 用 shadow labels 校准 margin。
- 加入真实失败和 human intervention 数据微调 critic。
- 对真实环境使用更高 uncertainty penalty。

## 14. 实验设计建议

为了证明可发表性，实验不要只报成功率。建议至少包含：

```text
success rate
cycle time / task completion time
control halting time
switch count
mean jerk / max jerk
force violation count
latency robustness curve
stale observation robustness
value profile calibration
```

关键 ablation：

- 固定 RTC。
- RTC + random adaptive switch。
- RTC + heuristic switch cost。
- VG-RTC without profile min/CVaR。
- VG-RTC without ranking loss。
- VG-RTC without uncertainty/support penalty。
- VG-RTC full。

可视化：

- 在轨迹上画 `P_old[k:k+M]` 和 `P_new[0:M]`。
- 标出实际切换点。
- 标出 value cliff 与任务阶段。
- 展示 early switch、late switch、no switch 的失败对比。

## 15. 实现落地清单

最小实现需要以下文件/模块：

```text
profiled_chunk_critic.py
chunk_replay_dataset.py
counterfactual_chunk_sampler.py
value_guided_switcher.py
rtc_hybrid_generator.py
async_chunk_executor.py
train_chunk_critic.py
eval_vg_rtc.py
```

核心数据结构：

```python
class ChunkCandidate:
    chunk: Tensor          # [H, action_dim]
    gen_obs_id: str
    gen_time: float
    source: str            # raw_vla / rtc_inpaint / blend / fast
    locked_prefix_len: int

class ChunkValueProfile:
    value: Tensor          # [H]
    uncertainty: Tensor    # [H]
    support: Tensor        # [H]
    safety_risk: Tensor    # [H]

class SwitchDecision:
    action: str            # continue / switch / wait / safe
    switch_delay: int
    q_continue: float
    q_replan: float
    margin: float
    reason: str
```

## 16. 可作为论文贡献的算法表述

可以在论文中这样表述 VG-RTC：

> We propose Value-Guided Real-Time Chunking (VG-RTC), a latency-aware execution algorithm for chunking-based vision-language-action policies. Instead of committing to a fixed chunk boundary, VG-RTC learns a profiled chunk critic that assigns a risk-adjusted value profile to each action index inside a predicted action chunk. At runtime, the executor compares the option value of continuing the currently active chunk suffix against the option value of switching to an RTC-inpainted candidate chunk generated from the latest observation. The comparison explicitly accounts for value cliffs within the chunk, inference latency, chunk staleness, uncertainty, support under the training distribution, and contact-aware switching costs. The resulting switcher adaptively schedules when a new chunk should take over, preserving RTC's smooth asynchronous execution while making chunk boundaries task-value aware.

中文贡献点可写成：

1. **Chunk value profile**：提出面向 action chunk 的逐 index critic，不只评估整个 chunk，而是预测 chunk 内的价值走势和潜在价值悬崖。
2. **Option-value switching**：把继续执行旧 chunk 和切换到新 chunk 形式化为两个 temporally extended options，通过 `Q_continue` 和 `Q_replan` 做实时选择。
3. **Latency/contact-aware adjustment**：在切换价值中显式建模推理延迟、chunk staleness、动作连续性、接触状态和 OOD 不确定性。
4. **Conservative profile critic training**：通过 TD、counterfactual ranking、保守正则和 uncertainty ensemble 训练可部署的 chunk critic，降低 replan 高估风险。
5. **Orthogonal plug-in design**：该方法不要求重训 base VLA，可叠加在 RTC、StreamingVLA、CO-RFT 后的 policy 或 RLT actor-critic 之上。

最简洁的算法名：

```text
Value-Guided Real-Time Chunking with Profiled Chunk Critics
```

最核心的公式：

```text
d* = argmax_d [
    gamma^d V_theta(o_t, g, A_hybrid(d), d)
    - C_switch(d)
    - C_latency(d)
    - C_uncertainty(d)
    - C_ood(d)
]

switch iff
    Q_replan(d*) > Q_continue + margin(o_t, contact, uncertainty)
```

这句话是论文摘要级别的贡献：

> VG-RTC turns real-time chunking from a fixed-latency smoothing heuristic into a value-aware online option selection problem, where action chunk boundaries are chosen by learned within-chunk value profiles rather than by a pre-defined execution schedule.
