# Event-Conditioned RTC / Event-Synchronized Chunking

## 结论先行

这个想法有研究味道，但强版本不成立。

不应说：

> action chunk 的 index 可以从时间坐标彻底变成事件坐标。

更稳的说法是：

> 底层控制仍然按时间步执行，但学习一个连续、带不确定性的 event coordinate / boundary hazard，用它调节 RTC 的 overlap weight、rechunk timing、horizon 和 residual authority，使 chunk consistency 只在 event-local 范围内强约束。

也就是说，它应该叫：

```text
Event-Conditioned RTC
```

而不是彻底的：

```text
Event-Indexed Action Chunking
```

## 核心问题

RTC 和 A2C2 解决了 chunked VLA 的两个现实问题：

- RTC：大模型推理慢，如何异步生成并平滑执行 action chunks；
- A2C2：chunk 执行中 observation 变了，如何每个控制步做轻量 correction。

它们通常默认 chunk 内第 `k` 步的意义主要由时间位置决定：

```text
tau = k / H
```

但真实操作往往不是按固定时间推进，而是按事件推进：

- 末端接触到把手；
- 夹爪闭合并夹住物体；
- 抽屉开始移动；
- 按钮触发；
- peg 碰到孔边；
- 物体放入容器；
- gripper 释放完成。

同样是 chunk 第 20 步，一次 rollout 可能还没接触，另一次已经进入接触保持。继续用固定时间 overlap 约束，就可能把“接触前动作”强行延续到“接触后状态”。

核心问题是：

> RTC 的 chunk consistency 是否应该只在同一物理事件段内强约束？

## 一句话定义

**Event-Conditioned RTC** 学习一个事件编码器：

```text
E_psi(h_t) = (xi_t, b_t, u_t, z_t)
```

其中：

- `xi_t`：连续 event progress coordinate；
- `b_t`：boundary hazard，表示是否接近事件边界；
- `u_t`：uncertainty；
- `z_t`：事件上下文 token。

它不直接决定动作，只调节：

- RTC overlap/prefix weights；
- 是否提前 rechunk；
- 执行 horizon；
- A2C2/residual correction 权限；
- trust gate 的敏感度。

## 理论形式

冻结 VLA 输出 chunk：

```text
A_t = pi_vla(h_t, l)
    = (a_{t|t}, a_{t+1|t}, ..., a_{t+H-1|t})
```

普通 RTC 使用上一段 leftover/prefix：

```text
P_t = (a_{t|t-1}, ..., a_{t+m|t-1})
```

并根据时间位置给 overlap weight：

```text
w_k = f_time(k, inference_delay, execution_horizon)
```

Event-Conditioned RTC 改成：

```text
w_k = f_event(k, xi_old, xi_new, b_t, u_t)
```

一个简化形式：

```text
w_k = w_time(k)
    * exp(- || xi_new,k - xi_old,k ||^2 / sigma^2)
    * (1 - u_k)
```

直觉：

- 新旧 chunk 处在同一事件段：强 overlap consistency；
- 已跨过事件边界：降低旧 chunk 约束；
- boundary hazard 高：缩短 horizon，提前重采样；
- uncertainty 高：退回标准 RTC；
- event 完成：允许新 chunk 快速切换，而不是为平滑继续执行旧意图。

事件边界可以用 stopping time 写：

```text
T_j = inf { t > T_{j-1} : b_t > theta and C(h_t) }
```

`C(h_t)` 是滞回、最小间隔、置信度等条件，避免噪声频繁触发。

## 它不是 FSM

这点必须写清楚。

FSM 是：

- 人工定义离散 phase；
- 规则转移；
- 每个 phase 绑定动作模板；
- `if contact then do insertion_mode`。

Event-Conditioned RTC 应该是：

- event module 输出连续 `xi, b, u, z`；
- 不输出动作；
- 不决定任务策略；
- 只调 RTC mask、guidance、horizon、rechunk、correction authority；
- 动作仍由 VLA、RTC、A2C2 或 residual actor 生成。

一句话区分：

> FSM decides what the robot should do; event-conditioned chunking decides how much the current chunk should trust old temporal commitments under semantic progress drift.

## 和 RTC / A2C2 / RLT 的关系

### RTC

RTC 解决 continuity，Event-Conditioned RTC 解决 continuity 的适用范围。

标准 RTC 问：

> 新 chunk 怎么接上旧 chunk？

Event-Conditioned RTC 问：

> 旧 chunk 是否还处在同一物理事件段，值得继续强约束？

### A2C2

A2C2 每个控制步做 lightweight correction。它通常用 chunk position feature，例如 `k/H`。

Event-conditioned 版本可以把：

```text
tau = k / H
```

扩展为：

```text
(tau, xi_t, b_t, u_t)
```

也就是告诉 correction head：

- 现在在 chunk 的第几步；
- 现在在任务事件中的哪里；
- 是否接近边界；
- 事件估计是否可靠。

### RLT

RLT 关心 native RL token 给 actor-critic。Event-Conditioned RTC 不等于 RLT，但可以给 RLT-style residual 提供更好的控制权坐标：

```text
low event uncertainty + same event segment -> trust RTC / VLA
high boundary hazard -> allow residual / RL
```

## 当前 Piper 环境下哪些事件可观测

相对可行：

- 抽屉开始移动：视觉或 tag 位移；
- 抽屉到达开/关阈值：位移阈值；
- 夹爪闭合完成：gripper observed position；
- 夹住物体后物体跟随：视觉相对运动；
- 按钮/开关触发：视觉状态或电信号；
- 任务轴 progress 停滞：视觉 progress；
- command-execution mismatch：FK EE delta 与 command delta；
- RTC overlap inconsistency：debug tracker。

较弱但可辅助：

- motor speed 下降；
- motor current 或 load，如果 SDK 稳定暴露；
- gripper cmd-pos mismatch；
- joint tracking error。

不适合作为主事件：

- 真实接触 onset；
- 接触法向变化；
- 摩擦状态；
- 微小力矩卡滞；
- 高精度装配中的力控相变。

所以最适合的任务不是高精度插孔，而是：

- 抽屉打开/关闭；
- 夹住把手后拉动；
- 放入物体；
- 按钮/开关；
- 可视滑块/限位。

## 最适合的实验任务

### 抽屉任务

事件链：

```text
approach handle
-> gripper close
-> object/handle attached
-> drawer starts moving
-> drawer reaches open threshold
-> place object
-> drawer starts closing
-> drawer reaches closed threshold
```

它适合 Event-Conditioned RTC，因为事件边界明显，且不要求硬顶。

关键可测量量：

- 抽屉位移；
- 夹爪位置；
- 物体是否进入抽屉；
- chunk boundary jerk；
- RTC correction norm；
- intervention count；
- event boundary error。

### 按钮/开关

事件链：

```text
approach
-> contact/progress
-> activated
-> retract
```

如果有电信号或视觉状态，event label 很干净。

### 粗公差插入

后续可做，不建议第一个做。需要低速、软材料、大 clearance，并把失败定义为 progress stagnation，而不是真实 force event。

## 训练与监督

事件编码器可以用弱监督：

- visual change point；
- task progress derivative；
- gripper state transition；
- intervention timing；
- RTC correction norm spike；
- command-execution mismatch；
- success/failure boundary；
- human correction start/end。

损失可以包括：

```text
L_event = BCE(b_t, boundary_label)
L_progress = || xi_pred - normalized_progress ||^2
L_uncertainty = calibration_loss(u_t, boundary_error)
L_smooth = || xi_t - xi_{t-1} || with monotonic regularization
```

不需要人工 phase name。可以用自动事件：

- 抽屉位移超过阈值；
- gripper 到达闭合阈值；
- 物体进入容器区域；
- 按钮状态改变；
- progress derivative 从正变零。

## Baselines

必须对比：

- Frozen VLA；
- Frozen VLA + standard RTC；
- Frozen VLA + A2C2-style correction；
- RTC + A2C2；
- short-horizon chunking；
- fixed time rechunk；
- manual phase/oracle event boundary；
- Event-Conditioned RTC；
- Event-Conditioned RTC + residual。

特别重要：

- manual phase baseline：证明不是手写 FSM；
- oracle event boundary：证明 detector 上限；
- random/delayed event：证明事件时机有因果价值；
- time-warp / delay sweep：证明 event coordinate 优于 time index。

## 指标

不要只报 success rate。

核心指标：

- event-boundary failure rate；
- event alignment error；
- cross-rollout synchronization；
- RTC correction norm before/after event；
- post-event recovery time；
- chunk boundary jerk；
- rechunk frequency；
- intervention count；
- under-delay robustness；
- under-time-warp robustness。

最关键实验：

> 在同一任务中随机化事件发生时间，例如抽屉初始位置、把手位置、物体重量、VLA inference delay。标准 time-indexed RTC 退化，而 event-conditioned RTC 更稳定。

## 最大风险

1. **事件检测抖动**  
   会导致频繁 rechunk 和 action jitter。需要 hysteresis、minimum dwell time、uncertainty fallback。

2. **事件不可观测**  
   位置控制、无 F/T 下很多微接触看不到。必须选视觉 progress 可见的任务。

3. **A2C2 已经足够**  
   如果任务没有 event timing drift，per-step correction 可能足够，Event RTC 不会赢。

4. **退化成工程 phase 切段**  
   如果每个任务手写事件，就不像研究。需要共享 detector、软 gate、oracle/manual baseline。

5. **牺牲平滑**  
   降低 overlap consistency 可能带来突变。需要 event-local smoothness，而不是取消 smoothness。

## 可写成的论文 claim

强 claim：

> Time-indexed action chunking is structurally misaligned with event-driven manipulation under variable event timing.

方法 claim：

> We propose event-conditioned real-time chunking, a plug-in synchronization layer that replaces fixed chunk-position schedules with learned event-coordinate schedules for frozen chunked VLAs.

保守最终 claim：

> Event coordinates provide a better alignment variable than chunk time indices for real-time execution of VLA action chunks in event-driven manipulation.

## 推荐标题

```text
Event-Conditioned Real-Time Chunking for Frozen Vision-Language-Action Policies
```

或：

```text
Event-Synchronized Action Chunking for Real-Time VLA Control
```

## 最终判断

这个 idea 适合作为第二主线或强辅助主线。它的理论叙事漂亮，但真机底层要求比 Chunk Trust 更高：你必须证明事件坐标确实比时间坐标更好，而不是只多了一个 phase detector。

在当前 Piper 上，最合理定位是：

```text
event signals modulate RTC and residual authority
```

而不是：

```text
event signals replace time-indexed control
```

如果和 Chunk Trust 组合，Event-Conditioned RTC 可以解释“什么时候重新信任新 observation / 放松旧 chunk”，而 Chunk Trust 解释“当前 chunk 是否仍值得执行”。这两个组合会比较强。

## References

- LeRobot RTC docs: https://huggingface.co/docs/lerobot/rtc
- A2C2 / Leave No Observation Behind: https://openreview.net/forum?id=y5SGBsndWv
- OpenPI: https://github.com/Physical-Intelligence/openpi
- RL Tokens / RLT: https://www.pi.website/research/rlt
