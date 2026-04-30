# Evo-RL 方向多轮 Battle Log

更新时间：2026-04-28
目标：不是一次性汇总观点，而是让代码落地派、创新性挑刺派、真实机器人可行性派连续互相攻击，直到收敛出一个可做且不弱创新的方案。

## 0. 参战角色

**A：代码审计 / 落地派**

立场：必须贴 Evo-RL 现有代码链路，不要为了题目好看重造系统。Evo-RL 最成熟的积木是 human-inloop recording、`policy_action`/executed `action`、ACP、`pistar06` value、RTC。

**B：创新性挑刺派**

立场：不能把已有系统拼一下就当论文。`Correction-Guided Plug-in RL Tokens` 太容易被打成 RLT 降级版、ACP 标签增强版、HIL-SERL 接到 VLA 上。

**C：真实机器人可行性派**

立场：2-3 个月、单臂、1 张 GPU、少量示教，只能做冻结开源 VLA 的小样本真实机后适配。必须压低任务数、压低系统复杂度、避免大模型全参和完整 RLT/RECAP。

## 1. Round 1：独立立场

### A 的初始判断

A 读 Evo-RL 后认为：

- Evo-RL 已经有接管记录、`pistar06` value、ACP prompt、RTC。
- 没有 native RLT token。
- 没有 VLA base action + residual actor + critic 的统一训练栈。
- 最贴代码的方向是把二值 ACP 升级成 correction-structured ACP，即从 `policy_action`、executed `action`、`is_intervention`、接管持续时间、是否成功恢复中生成更细的 correction tokens。

A 的初始推荐：

**Correction-Structured ACP++ + RTC curriculum**
先别上 full residual RL。

### B 的初始判断

B 认为原题不够硬：

- “Plug-in RL Token” 容易被说成 RLT 去 native token 后的工程平替。
- “Correction-guided” 容易被说成 RECAP/HIL-SERL 早就用了接管/纠错。
- hidden-state adaptor 不是创新，因为开源 VLA 本来就能 hook hidden states。

B 的初始推荐：

**Counterfactual / Failure-Boundary Token + Boundary-Gated Residual**
核心必须是从接管中学习 VLA 的 failure boundary / recoverability，而不是学更细的 prompt tag。

### C 的初始判断

C 认为 2-3 个月内最稳的是：

- `SmolVLA` 或 `PI05/OpenPI`，冻结主干。
- 任务选按钮/拨杆、peg/假 USB 插入。
- VLA 负责粗接近，RL 或 residual 只补最后几厘米。
- 不碰 Evo1 stage2、OpenVLA 7B 全参、world model、完整 RLT/RECAP。

C 的初始推荐：

**SmolVLA + 接管数据 + 小 residual RL / phase-gated residual RL**

## 2. Round 2：互相攻击

### A 攻击 B/C

A 对 B：

- B 的 `Boundary Token + Gated Residual` 论文味强，但 Evo-RL 没有现成训练闭环。
- RTC 只是推理期 correction，不是训练模块。
- VLA residual actor-critic 需要新建 residual head、critic、buffer schema、off-policy update、在线稳定策略，已经超出 2-3 个月小改。

A 对 C：

- C 的“小 residual RL”听起来保守，但并不贴 Evo-RL 当前代码。
- Evo-RL 的 SAC/HILSERL 是独立 RL policy，不是接在 SmolVLA/PI05 后面的 residual head。
- 真要做 residual RL，工程风险不比 B 小。

A 第二轮结论：

**ACP++ 仍应作为主线，但必须从 prompt 文案升级成结构化纠错条件接口。RTC 可以作为辅助技术点。Boundary residual 不适合作为当前主线。**

A 投票：

1. ACP++：赞成，但必须是 structured correction tokens。
2. RTC Curriculum：赞成。
3. Universal Interface：中立偏反对。
4. Boundary Token + Gated Residual：反对作为主线。
5. Intervenability RL：反对作为当前主线。

### B 攻击 A/C

B 对 A：

- ACP++ 最大问题是“标签升级，不是问题升级”。
- 如果只是 `Advantage: positive/negative` 变成 correction 强度/类型，审稿人会认为是 ACP 的 richer-label 版本。
- 这不是新 RL interface，也不是新 VLA+RL 机制。

B 对 C：

- “冻结 VLA + 接管数据 + 小 residual policy + 精细阶段 gate”每一步都合理，但每一步都不新。
- 容易被说成 RLT-lite、HIL-SERL 接到 VLA 后面、或者普通 residual RL。
- 如果 gate 只是 heuristic，就更像系统技巧。

B 第二轮结论：

**A 的保守版最容易写，也最容易死；C 的保守版最稳做结果，也最容易被说成工程 recipe。论文主角必须是 failure boundary / recoverability / intervention timing。**

B 投票：

1. Boundary Token + Gated Residual：强烈支持。
2. Universal Interface：支持但不建议第一枪。
3. RTC Curriculum：谨慎支持，做组件。
4. Intervenability RL：支持但需和 Boundary Token 结合。
5. ACP++：反对作为主线，只能做 baseline/组件。

### C 攻击 A/B

C 对 A：

- “最容易接 Evo-RL”不等于“最容易出论文结果”。
- ACP++ 如果没有 phase/boundary 结构，真实机上可能能跑，但不一定稳；还容易被打成 ACP 工程增强。
- 真实机最怕动作修改范围不受控，最后 residual 学乱改。

C 对 B：

- 反事实失败边界这个概念好，但如果做成大理论，会被真实数据量拖垮。
- 边界标签、可恢复状态、counterfactual 轨迹都很难在 2-3 个月真实机里做硬。
- 不要做“通用 boundary token”或跨模型统一接口。

C 第二轮结论：

**保留 B 的骨架，删掉 B 的野心；拒绝 A 的裸 ACP++。最合适的是 intervention-conditioned gate + chunk-local residual + RTC。**

C 投票：

1. Intervenability RL：最高票，但必须收缩成 intervention-conditioned gate。
2. Boundary Token + Gated Residual：第二，但不要重型 token 理论。
3. RTC Curriculum：中票，作为部署/稳定组件。
4. ACP++：低票，只做 baseline/MVP 组件。
5. Universal Interface：最低票，当前资源下太虚。

## 3. Round 2 后的冲突点

三方分歧非常明确：

- A 要可落地：**ACP++ + RTC**。
- B 要可发表：**Failure-Boundary Token + Gated Residual**。
- C 要真机可做完：**Intervenability gate + chunk residual + RTC**。

因此不能直接选 A，也不能直接选 B，也不能直接选 C。

需要一个折中方案满足：

1. 不把 ACP++ 当论文主贡献。
2. 不做重型反事实理论。
3. 不做完整 VLA actor-critic/RLT。
4. 仍然要回答一个尖锐问题：**冻结 VLA 什么时候应该被纠正？**
5. 能贴 Evo-RL 现有接管数据和 RTC 代码。

## 4. Round 3：收敛方案

我给三方抛出折中方案：

**When Should a Frozen VLA Be Corrected? Boundary-Gated Residual Adaptation from Human Takeovers**

最小方法：

1. 用 Evo-RL 接管数据生成 `intervention / boundary / recoverability / correction` labels。
2. 学一个 gate / recoverability score，预测当前是否值得纠正 frozen VLA。
3. gate 高时启用 clipped chunk-local residual editor。
4. ACP++ 只是标签生成、训练条件或 baseline，不当主贡献。
5. RTC 只是部署稳定组件，不当主贡献。

### A 的最终投票：有条件接受

A 接受理由：

- 这版比纯 ACP++ 更像主论文。
- 又没有膨胀到完整 residual RL 或重型反事实理论。
- 还能复用 Evo-RL 的 human-inloop data、`policy_action` vs executed `action`、ACP baseline、RTC。

A 的条件：

- 砍掉 `Intervenability RL` 叙事，不要写成 actor-critic / online RL 论文。
- `recoverability` 只做辅助信号，不要升成理论核心。
- residual 必须是 editor，不是第二个 policy。
- 边界 token 不是主创新，主创新是 boundary-gated correction trigger。

A 要求的最小模块：

1. Takeover-derived labels。
2. Boundary gate。
3. Chunk-local residual editor + RTC。
4. ACP++ 作为 baseline/ablation。

### B 的最终投票：有条件接受

B 接受理由：

- 题目从“更细 prompt”升级成了“什么时候该纠正 frozen VLA”。
- 主张有明确问题，不再像 ACP++ 小修。
- 保留了 failure boundary / recoverability / gated residual 这个创新骨架。

B 的条件：

- gate 必须证明不是普通 intervention predictor。
- Boundary 必须证明不是 ACP++ 换名。
- residual editor 必须证明不是 naive residual RL。
- 必须有 raw hidden-state baseline，证明不是 feature trick。
- RTC 必须降级为部署组件，不能抢主贡献。

B 要求保留的关键消融：

- `manual phase gate`
- `intervention classifier only`
- `recoverability-only gate`
- `binary ACP`
- `Correction-Structured ACP++`
- `always-on residual`
- `naive residual RL`
- `gate + residual BC only`
- `raw hidden-state + gate + residual`
- `without RTC` / `with RTC`

### C 的最终投票：有条件接受

C 接受理由：

- 工程起点现实：ACP++ 标签生成 + RTC。
- 论文主张清楚：不是 ACP++，而是 `When should a frozen VLA be corrected?`
- 边界落到 human takeover / recoverability / local residual，不再是重型反事实理论。

C 的条件：

- label 体系不能做胖，最小只保留：
  - `intervention`
  - `boundary`
  - `recoverable_after_takeover`
- gate 只输出：
  - `gate score`
  - `recoverability score`
- residual 必须：
  - chunk-local
  - clipped
  - gate 低时完全退回 frozen VLA
- 只做 2 个真实任务。

C 的最小数据预算：

每任务：

- 成功示教：`20-25` 条。
- policy rollout + human takeover：`15-20` 条。
- 在线交互：`60-80` episodes。
- seen 评测：`20` 次。
- unseen pose 评测：`10` 次。

## 5. 最终收敛结果

三方最终都是：

**有条件接受同一个收缩版方案。**

最终题目：

**When Should a Frozen VLA Be Corrected? Boundary-Gated Residual Adaptation from Human Takeovers**

中文题目：

**冻结 VLA 何时应被纠正：基于人类接管的边界门控残差适配**

最终主问题：

> 人类接管最有价值的信息，不只是“哪段轨迹更好”，而是暴露了 frozen VLA 在真实机器人上的可纠正失败边界。能否学习一个轻量 gate，在必要时才对 action chunk 做局部有界 residual correction，从而比 ACP 条件化、全程 residual、manual phase 更省样本、更安全？

## 6. 最终方法定义

### 6.1 输入数据

从 Evo-RL HIL 流程得到：

- observation
- task
- proprio/state
- VLA predicted `policy_action`
- actual executed `action`
- `is_intervention`
- intervention state
- episode success/failure
- optional value / advantage from `pistar06`

### 6.2 标签生成

只保留最小标签集：

- `intervention`: 当前是否人类接管。
- `boundary`: 接管开始前后的一小段窗口，表示 VLA 即将进入可纠正失败边界。
- `recoverable_after_takeover`: 接管后是否成功恢复，或 value/progress 是否改善。
- `correction_residual`: `action - policy_action`，只用于 residual editor 监督，不做复杂 taxonomy。

### 6.3 模型

```text
frozen VLA -> a_vla_chunk
        |
takeover-derived boundary/recoverability gate
        |
if gate low: execute a_vla_chunk
if gate high: a_exec = a_vla_chunk + clip(delta_a)
        |
RTC / chunk continuity stabilizer
```

关键约束：

- VLA 主干冻结。
- gate 是轻量模块。
- residual editor 不是第二个 policy，只做 chunk-local clipped correction。
- gate 低时完全退回 frozen VLA。
- RTC 只作为部署稳定组件。

## 7. 最终实验设计

### 任务

只做两个任务：

1. 按钮按压 / 拨杆开关。
2. peg-in-hole / 假 USB 插入。

先不做抽屉，避免摩擦、勾挂、reset 复杂度干扰结论。

### 必须保留的对比

最小四个：

1. `Frozen VLA + RTC`
2. `ACP++ baseline`
3. `Full residual without gate`
4. `Boundary-gated chunk residual (ours)`

建议再加：

5. `Binary ACP`
6. `Manual phase gate`
7. `Raw hidden-state gate + residual`

### 核心指标

- 成功率。
- 达到 70% / 80% 成功率所需 episodes。
- 每 20 次 rollout 的接管次数。
- 危险停止次数。
- residual magnitude。
- gate activation ratio。
- gate 是否集中在接管前/失败边界。
- seen / unseen pose 成功率。

## 8. 最终砍掉的方向

明确淘汰：

- ACP++ 独立成题。
- 完整 PI RLT/RECAP 复现。
- native RL token。
- 完整 VLA actor-critic residual RL。
- Universal post-hoc interface。
- 重型反事实失败理论。
- Evo1 stage2 / 大模型全参后训练。
- 三个以上真实任务。

## 9. 最终执行顺序

第 1-2 周：

- 跑通 `Frozen VLA + RTC`。
- 任务 1 建好，确认 VLA 能靠近但末端常失败。

第 3-4 周：

- 收 20-25 条成功示教。
- 收 15-20 条 takeover trajectories。
- 生成 `intervention / boundary / recoverable_after_takeover / correction_residual`。

第 5-6 周：

- 训练 gate / recoverability score。
- 先不做 residual，验证 gate 是否预测接管边界和 correction need。

第 7-8 周：

- 加 clipped chunk-local residual editor。
- 完成 `ACP++ / full residual no gate / ours` 三线对比。

第 9-10 周：

- 在第二个任务复现。
- 做 unseen pose 测试。

第 11-12 周：

- 补 ablation、失败案例、论文图表。

## 10. 最后结论

多轮 battle 后，最合理的停止点不是 A、B、C 任一方原始方案，而是：

**以 C 的真实机约束为边界，用 A 的 Evo-RL 代码链路起步，保留 B 要求的 failure-boundary / gated residual 作为论文核心。**

最终方向：

**冻结 VLA 何时应被纠正：基于人类接管的边界门控残差适配。**

这个方向比 ACP++ 更有论文问题，比完整 RLT/residual RL 更能落地，比单纯 SmolVLA+RL recipe 更不容易被说成复现。
