# Contact-Implicit VLA+RL 五轮子智能体 Battle 与最终方案

更新时间：2026-04-28
基础文件：`2026_contact_rethink.md`、`papers_2026_addendum.csv`
目标：基于 2026 contact-rich VLA / VLA+RL 文献，经过至少 5 轮辩论，收敛出一个具体、可做、避免人工多标签的新方向。

## 0. 参战角色

**A：创新性 / 相关工作审稿人**

关注：是否会被 VLAJS、CRAFT、ForceVLA2、HapticVLA、FPC-VLA、CycleVLA、HIL-SERL、CR-DAgger 打掉。

**B：真实机器人实验负责人**

关注：单臂夹爪、1 张 GPU、少量示教、真实在线交互条件下，任务和指标能否完成。

**C：算法 / 实现负责人**

关注：在 Evo-RL / LeRobot 风格代码里，怎么最小实现 contact-conditioned VLA guidance。

## 1. Round 1：独立评估

### A：创新性压力

A 认为当前 `Contact-Implicit VLA Jump-Start RL` 有缝隙，但很窄。

最危险的攻击：

- **VLAJS**：已经做 VLA regularization jump-start RL；我们只是把 time annealing 换成 contact-conditioned annealing。
- **CRAFT / ForceVLA2 / HapticVLA / TaF-VLA**：contact-rich VLA 的关键是 force/tactile；我们的 commanded-vs-realized mismatch 只是低保真 proxy。
- **FPC-VLA / CycleVLA**：failure prediction / correction 已经有人做，不能把失败检测当贡献。
- **HIL-SERL / CR-DAgger**：human intervention / residual correction 已经有人做，不能再把接管 residual 当主线。

A 要求收窄为：

> 在没有 force/tactile 传感器和人工 phase labels 的真实机器人 last-centimeter contact 任务中，如何用机器人自动日志估计 VLA guidance 何时应该被信任，并据此动态调节 VLA-to-RL regularization。

初始投票：**有条件支持，但当前表述接近反对。**

### B：真机任务压力

B 认为方向可做，但任务必须收窄。

任务排序：

1. **按钮 / 拨杆 / 开关**：最适合调试闭环，reset 简单，成功信号强。
2. **大公差 peg / 3D 打印插孔**：最适合主论文，能制造 last-centimeter jam。
3. **滑块 / 简化抽屉**：可扩展，但抓取和摩擦会混入太多因素。
4. **真实 USB / 小孔高精度插入**：暂不推荐，容易变成硬件装配工程。

B 初始投票：**有条件支持，7/10。**

条件：

- 先按钮/开关验证闭环。
- 主任务用大公差 peg。
- 不默认有六维力、触觉阵列、mocap。
- 核心指标必须是自动 contact/progress proxy 和样本效率提升。

### C：算法路线压力

C 比较三条路线：

| 路线 | 形式 | 评价 |
|---|---|---|
| A. RL policy + contact-conditioned VLA regularization | `L_actor = L_SAC + λ(c) * D_dir(a_rl, a_vla)` | 最贴题、最小实现、主路线 |
| B. VLA action + contact-gated residual | `a_exec = a_vla + g(c)*delta_a` | 训练语义复杂，critic/credit assignment 不稳 |
| C. micro-action primitive + RL 选择 | spiral/grid/wiggle 等 | 真机可能有效，但像任务工程 |

C 初始投票：

- A：0.60，主路线。
- C：0.25，保底 baseline / ablation。
- B：0.15，等 A 跑通再说。

## 2. Round 2：互相质疑后第一次收窄

### A 对 B/C 的回应

A 接受 B 的任务收窄，也接受 C 的 SAC regularization 主线。

但 A 强调：

- 不能写成“低成本 contact-rich VLA+RL 框架”。
- 只能写成 **VLAJS 的物理状态条件化信任调度**。
- 必须证明 jam/contact 阶段继续信 VLA 会伤害 RL。

A 修改题目：

**Contact-Conditioned VLA Regularization for Low-Cost Last-Centimeter Robot RL**

必须保留的 baseline：

- `Frozen VLA`
- `SAC from scratch`
- `VLAJS-style global/time annealing`
- `Always-on VLA regularization`
- `Ours contact-conditioned`

### B 对 A/C 的回应

B 设计了能证明 VLA guidance 负迁移的 peg 实验：

- 制造可重复 jam：孔位横向偏 2-5 mm、角度偏 3-8 度。
- VLA 通常会继续给“往孔里推”的大方向。
- 对比：
  - `VLAJS time anneal`
  - `jam 阶段 always-on VLA guidance`
  - `ours contact-conditioned downweight`

负迁移证据不只看成功率，还要看：

- jam 后继续推的比例。
- jam 持续时间。
- 最大 tracking error。
- jam 后恢复成功率。

B 要求算法方每步记录：

- `a_vla`
- `a_rl`
- `a_exec`
- `lambda(contact_score)`
- `contact_score`

并支持强制 `lambda=high/low/zero` 做 ablation。

### C 对 A/B 的回应

C 给出在线 contact_score：

```text
track_t = ||u_{t-1} - Δx_t|| / (||u_{t-1}|| + eps)
prog_t  = max(0, d_{t-1} - d_t) / (||u_{t-1}|| + eps)
stag_t  = EMA[ ||u_{t-1}|| > τ_u and ||Δx_t|| < τ_x and prog_t < τ_p ]
eff_t   = normalized current/load EMA, optional
c_t     = sigmoid(w1*track_t + w2*stag_t + w3*eff_t - w4*prog_t - b)
```

严格约束：

- 只用截至当前步的信息。
- 不用未来帧。
- 不用 episode success。
- 不用 reward classifier。
- 电流/load 只是可选项，不能作为主依赖。

VLA 和 SAC 的动作对齐：

- 全部投影到 EE delta translation direction。
- 只做 cosine directional regularization。
- 不正则 gripper、rotation、动作幅值。

## 3. Round 3：候选方案成形

### A 的意见

A 认为题目还要再压：

**Contact-Conditioned VLA Regularization for Low-Cost Last-Centimeter Manipulation**

主张只限于：

> 在无力/触觉传感器、无人工 phase labels 的低成本平台上，用在线 contact proxy 调节 frozen VLA regularization，提高 last-centimeter RL 样本效率和成功率。

A 认为 schedule 可能被说成 heuristic，所以必须做：

- 固定高 `lambda`
- 固定低 `lambda`
- time annealing
- 反向 schedule：jam 高信 VLA
- 无 stagnation
- 无 mismatch
- learned vs hand-tuned lambda

关键图：

1. `contact_score` 有效性图。
2. 训练曲线图。
3. `lambda` 行为与失败恢复案例图。

### B 的意见

B 将按钮降级：

- 按钮/拨杆只保留为调试和 appendix sanity check。
- 主论文任务只做可控大公差 peg 插入。

peg 装置建议：

- 3D 打印孔板。
- 可替换孔径。
- 倒角孔。
- peg 直径 8-10 mm。
- 初始孔径比 peg 大 1.0-1.5 mm。
- 后续可收紧到 0.5-0.8 mm。
- 孔板贴 AprilTag/ArUco。
- peg 或夹具贴色块 / tag。
- 固定俯视或斜视相机。

预算：

- `Frozen VLA`：20 episodes。
- `SAC from scratch`：30-40 episodes。
- `VLAJS/time anneal`：40 episodes。
- `Always-high lambda in contact/jam`：40 episodes。
- `Ours`：50-60 episodes。
- `Manual phase` 可选 20 episodes。

### C 的意见

C 最终把 lambda 定成显式、可解释 schedule：

```text
lambda_t = lambda_high * clamp(1 - contact_score_t, 0, 1)^gamma
```

默认：

- `gamma = 1`
- `lambda_high` 同一任务共享
- 主方法不做黑盒可学习 gate

loss：

```text
L_dir = 1 - cos(norm(P(a_rl)), norm(P(a_vla)))
L_actor = L_SAC + lambda_t * L_dir
```

baseline：

```text
Always-on:
lambda_t = lambda_high

Global/time-annealed:
lambda_t = lambda_high * max(0, 1 - train_step / T_anneal)

Zero:
lambda_t = 0

Low:
lambda_t = lambda_low

Ours:
lambda_t = lambda_high * (1 - contact_score_t)
```

VLA 慢或坐标不一致的保底：

- 每 `K` 步调用一次 frozen VLA，中间 hold direction。
- 若 VLA 输出 absolute pose，则用 `target_ee - current_ee`。
- 若 VLA 输出 joint action，则 FK/Jacobian 投影到 EE delta。
- 最坏情况用离线 VLA direction cache。

## 4. Round 4：失败模式压力测试

### A：审稿失败模式

1. **contact_score 是手调阈值，不能泛化。**
   必须跨孔位偏移、clearance、初始姿态报告同一组参数是否可用。

2. **赢 VLAJS 只是因为调参更多。**
   必须给 time-annealed baseline 做网格调参，报告 best baseline。

3. **peg 太简单，Frozen VLA 或 SAC 已经能做。**
   必须证明 Frozen VLA 能接近但插入失败，SAC scratch 学得慢，jam 是主要失败原因。

4. **contact_score 像 reward shaping。**
   必须做只用 proprio mismatch、只用 progress 的消融；并强调 contact_score 不进入 reward，只调 regularization。

5. **directional loss 不公平。**
   必须明确重采样/hold、归一化、只约束方向不约束幅值。

A 投票：**有条件进入最终方案。**

### B：真机失败模式

1. **contact proxy 不准。**
   先离线采脚本轨迹：free-space、正常插入、故意横向顶住；调到三类曲线可分。

2. **peg 装置太难。**
   先大 peg、大倒角、低摩擦材料；逐步收紧 clearance。

3. **baseline 太多。**
   主表只保留 4 个：`Frozen VLA`、`SAC`、`VLAJS/time anneal`、`Ours`；负迁移另做固定 jam ablation。

4. **reward 稀疏。**
   训练用自动 dense reward：`+axial_progress`、`-lateral_error`、`-jam_score`、`-large_action`；评估仍用成功率和 jam 恢复率。

5. **安全/磨损。**
   单步 EE translation 1-3 mm；连续 jam 1-2 秒自动退回 5 mm；tracking error 超阈值停机。

B 收敛预算：**140-170 个 peg episodes**。

### C：算法失败模式

1. **VLA 方向质量差。**
   只保留 cosine direction prior；屏蔽与 progress 负相关的维度。

2. **SAC 不稳定。**
   demo/replay warm start，限制 action scale，降低 entropy temperature。

3. **contact_score 噪声大。**
   EMA + hysteresis；lambda 用连续值，不硬切。

4. **坐标不一致。**
   全部投影到 EE delta translation。

5. **稀疏奖励学不动。**
   训练可加极小 dense progress，评估用 sparse success。

C 给出最小伪代码：

```python
for episode in train:
    obs = env.reset()
    prev_x = get_ee_or_joint(obs)
    prev_exec = zeros(action_dim)
    vla_dir = zeros(3)

    for t in range(T):
        a_rl = sac.select_action(obs)

        if t % K == 0:
            a_vla_raw = frozen_vla(obs, task)
            vla_dir = project_to_ee_translation(a_vla_raw, obs)

        x = get_ee_or_joint(obs)
        dx = x - prev_x
        effort = read_current_or_zero(obs)
        progress = estimate_progress(obs)

        c = contact_proxy(prev_exec, dx, effort, progress)
        c = ema_hysteresis(c)
        lam = lambda_high * clamp(1 - c, 0, 1)

        a_exec = clip(a_rl)
        next_obs, reward, done = env.step(a_exec)

        replay.add(
            obs, a_exec, reward, next_obs, done,
            info={
                "a_rl": a_rl,
                "a_vla": vla_dir,
                "a_exec": a_exec,
                "lambda": lam,
                "contact_score": c,
            },
        )

        prev_x = x
        prev_exec = a_exec
        obs = next_obs
        if done:
            break

for batch in replay:
    L_sac = sac_actor_loss(batch)
    L_dir = 1 - cosine(P(pi(batch.obs)), stopgrad(batch.info["a_vla"]))
    L_actor = L_sac + batch.info["lambda"] * L_dir
    update_actor(L_actor)
```

## 5. Round 5：最终投票

### A 最终投票

最终题目：

**Contact-Conditioned VLA Regularization for Low-Cost Last-Centimeter Manipulation**

中文：

**面向低成本最后厘米操作的接触条件化 VLA 正则强化学习**

主 claim：

> 在无力/触觉传感器、无人工 phase labels 的低成本机器人设定下，仅用在线 robot-log contact proxy 动态调节 frozen VLA 对 SAC actor 的 directional regularization，可比全局/时间退火 VLA regularization 更稳定地提升大公差 peg 插入的样本效率与成功率。

A 最终投票：**有条件接受。**

### B 最终投票

最终任务装置：

- 可调公差 peg 插入。
- 3D 打印孔板 + 倒角孔 + 可替换孔径。
- peg 直径 8-10 mm。
- 初始孔径比 peg 大 1.0-1.5 mm，后续可收紧到 0.5-0.8 mm。
- 孔板 AprilTag/ArUco。
- peg 或夹具色块/tag。
- 固定俯视或斜视相机。
- 按钮/拨杆只做调试和 appendix，不进主表。

episode 预算：

- `Frozen VLA`: 20 episodes
- `SAC from scratch`: 30 episodes
- `VLAJS / time anneal`: 40 episodes
- `Ours contact-conditioned`: 50 episodes
- 固定 jam ablation：20-30 episodes，比较 `lambda=high` vs `lambda=low/zero`

总计约 160-170 个 peg episodes。

B 最终投票：**支持，8/10。**

### C 最终投票

最终算法：

```text
track = ||u_{t-1} - Δx_t|| / (||u_{t-1}|| + eps)
stag  = EMA[||u_{t-1}||>τu and ||Δx_t||<τx and progress<τp]
eff   = normalized motor current/load, optional
c_t   = clip(EMA(w1*track + w2*stag + w3*eff), 0, 1)
```

```text
lambda_t = lambda_high * (1 - c_t)
```

```text
L_actor = L_SAC + lambda_t * L_dir
L_dir = 1 - cosine(P(a_rl), stopgrad(P(a_vla)))
```

`P(.)` 只取 EE translation direction。

C 最终投票：

**支持 A 路线：contact-conditioned VLA-regularized SAC。**

## 6. 最终可行方案

### 6.1 题目

**Contact-Conditioned VLA Regularization for Low-Cost Last-Centimeter Manipulation**

中文：

**面向低成本最后厘米操作的接触条件化 VLA 正则强化学习**

### 6.2 核心问题

不是“VLA 能不能指导 RL”，而是：

> 在无力/触觉传感器、无人工 phase labels 的低成本接触任务中，RL 什么时候应该信任 frozen VLA 的方向先验，什么时候应该在接触/卡滞阶段降低对 VLA 的信任？

### 6.3 方法

1. 使用 frozen VLA 只提供 EE translation direction prior，不直接执行 VLA 动作。
2. 用普通机器人日志在线计算 `contact_score`：
   - commanded-vs-realized mismatch
   - progress stagnation
   - optional motor current/load
3. 用 `contact_score` 调节 SAC actor 的 VLA directional regularization：

```text
L_actor = L_SAC + lambda_high * (1 - contact_score) * (1 - cos(P(a_rl), P(a_vla)))
```

4. free-space 阶段 `contact_score` 低，信 VLA。
5. contact / jam 阶段 `contact_score` 高，降低 VLA guidance，让 RL 学恢复 / search / insertion。

### 6.4 主任务

只做主表任务：

**可调公差 peg 插入**

按钮/拨杆：

- 仅用于调试系统和 appendix sanity check。
- 不作为主论文任务。

### 6.5 Baselines

主表必须保留：

1. `Frozen VLA`
2. `SAC from scratch`
3. `VLAJS-style time-annealed VLA regularization`
4. `Always-on VLA regularization` 或固定 `lambda=high` jam ablation
5. `Ours: contact-conditioned VLA regularization`

可选：

6. `Manual phase guidance`
7. `lambda=zero/low/high` schedule sweep

### 6.6 指标

每 episode：

- success rate
- success time
- total steps
- jam count
- longest jam duration
- jam recovery success rate
- max tracking error
- final insertion depth
- safety stop count

每 step：

- `a_vla_dir`
- `a_sac_dir`
- `a_exec`
- `lambda`
- `contact_score`
- `tracking_error`
- `progress_ratio`
- `lateral_error`
- `axial_progress`
- `joint_pos/vel`
- `ee_pose_fk`
- `gripper_pos/cmd`

### 6.7 必须展示的图

1. **Contact proxy 有效性图**
   一条 episode 时间轴上叠加 `tracking_error`、`progress_ratio`、`contact_score`、jam/success 事件。

2. **主任务训练曲线图**
   peg 插入上比较 Frozen VLA、SAC scratch、VLAJS time anneal、always-on、ours。

3. **Jam 恢复案例图**
   展示 jam 发生后，always-on VLA 继续推导致停滞，而 ours 降低 `lambda` 后恢复。

### 6.8 不能写的过度 claim

不能声称：

- 通用解决 contact-rich manipulation。
- 超过 force/tactile 方法。
- 提出了新 contact perception。
- 提出了 failure prediction/correction 框架。
- 提出了 residual learning / human intervention learning。
- 完全无启发式。

应该写：

> 我们提出一个 physics-motivated online contact proxy，用于调节 frozen VLA 对 RL actor 的方向正则强度，在低成本 last-centimeter peg 插入中减少 contact/jam 阶段 VLA guidance 的负迁移。

## 7. 最终结论

五轮 battle 后，三方收敛到同一个可行方案：

**不要做人工 boundary labels，不要做泛泛 failure correction，不要做 residual RLT。**

做：

**Contact-conditioned VLA-regularized SAC for low-cost peg insertion.**

这个方案的创新点足够窄、真机实验足够可控、代码实现也不需要重造 VLA/RLT 系统。它的成败关键不是大模型，而是能否用自动 contact proxy 明确证明：

**VLA 在 free-space approach 有帮助，但在 contact/jam 阶段继续强信 VLA 会产生负迁移；contact-conditioned trust scheduling 能缓解这个问题。**
