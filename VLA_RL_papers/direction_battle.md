# 方向 Battle：真正有价值的 VLA+RL 题目

## 1. 约束

- 资源：单臂夹爪，1 张 GPU，少量示教，可做真实在线交互。
- 目标：6-12 个月内做出可投稿小论文。
- 主线：VLA+RL 后训练，但不能只是复现 RLT/RECAP。

## 2. 候选方向评分

评分：5 分最好，1 分最差。

| 方向 | 热度 | 创新空间 | 资源匹配 | 风险 | 总评 |
|---|---:|---:|---:|---:|---|
| 复现 RECAP | 5 | 1 | 1 | 5 | 淘汰 |
| 复现 RLT | 5 | 2 | 3 | 4 | 不足以投稿 |
| VLA progress critic | 5 | 2 | 3 | 4 | 只能辅助 |
| VLA-guided RL student | 4 | 2 | 4 | 3 | 已被 RPD/VLAJS 占位 |
| World-model VLA RL | 5 | 3 | 2 | 5 | 太大 |
| Plug-in RL token | 5 | 4 | 4 | 3 | 推荐 |
| Plug-in RL token + correction advantage + phase gate | 5 | 5 | 4 | 3 | 最推荐 |

## 3. 为什么不是复现 RLT

RLT 的核心是 **native RL token**：模型在训练/适配时被要求产生一个紧凑 token，供小 actor/critic 做在线 RL。这需要模型内部结构和训练流程配合。

我们的题目是 **post-hoc plug-in RL token**：拿到已经训练好的开源 VLA 后，不改主干、不重训大模型，通过外接 adaptor 从 hidden states、action chunk、视觉语言特征中学习一个 RL-friendly token。

本质差异：

| 问题 | RLT | Plug-in RLT |
|---|---|---|
| 模型来源 | PI pi0.6 | OpenPI/SmolVLA/OpenVLA |
| Token 来源 | VLA 原生输出 | 后验 adaptor 学出 |
| 是否改 VLA 预训练 | 是/需要内部接口 | 否 |
| 训练对象 | 小 actor/critic + native token | token adaptor + phase gate + residual actor-critic |
| 适用性 | PI 内部模型 | 普通实验室开源模型 |
| 论文问题 | How to use native RL tokens? | How to retrofit RL tokens into frozen VLAs? |

## 4. 为什么要结合 RECAP

只做 plug-in token 还不够，因为它可能只是 RLT 的工程变体。必须引入 RECAP 的 correction/advantage 思想：

- Demonstration 教会基本任务，但不覆盖模型自己的失败状态。
- Human correction 刚好覆盖模型真实失败分布。
- Correction 前后的轨迹差异能提供比 sparse success 更密集的 advantage signal。

因此论文的独立创新应是：

**Correction-guided post-hoc tokenization**，而不是单纯 “take hidden states as RL state”。

## 5. 最终题目

**Correction-Guided Plug-in RL Tokens for Open-Source Vision-Language-Action Policies**

## 6. 方法决策

### 基座

优先级：

1. `SmolVLA`：最适合单卡、LeRobot、真实单臂。
2. `OpenPI pi0.5`：和 PI RLT/RECAP 关系最近，但算力和系统复杂度更高。
3. `OpenVLA`：强开源基线，但 7B 和机器人适配成本高。

默认先用 SmolVLA 跑通，后续加 OpenPI pi0.5 做泛化验证。

### Token adaptor

输入：

- VLA selected hidden states。
- task-focused visual pooled features。
- language instruction embedding。
- proprioception。
- VLA action chunk。
- 最近一步执行误差或目标相对位姿估计。

输出：

- `z_rl`：64-256 维 compact RL-state token。

训练信号：

- correction residual imitation：预测 human correction 相对 VLA action 的 residual。
- intervention prediction：判断当前 state 是否即将需要接管。
- progress delta regression：预测任务进展变化。
- contrastive progress objective：成功进展片段靠近，停滞/退步片段远离。

### Phase gate

启用 residual actor 的条件：

- progress stagnation 持续 N 步。
- intervention probability 超阈值。
- VLA action uncertainty 或 chunk inconsistency 高。
- 进入预定义关键区域，例如末端距离目标小于阈值。

默认策略：训练早期用保守 gate，宁可少启用 residual，也不要让 RL 全程乱改 VLA。

### Residual actor-critic

最终动作：

`a_exec = a_vla + clip(delta_a, -epsilon, epsilon)`

训练：

- 离线 correction warmup。
- 在线 SAC/AWAC/TD3+BC 风格更新。
- Critic 使用 sparse success + progress critic + intervention penalty。
- Actor 加 reference regularization，防止偏离 VLA 太远。

## 7. 实验设计

### 任务

任务 1：peg/plug insertion。
任务 2：button/switch contact pressing。
任务 3：drawer handle alignment or stacking alignment。

任务选择原则：

- VLA 可以完成粗略接近。
- 失败集中在最后精细阶段。
- 成功可自动判定或半自动判定。
- reset 成本低。

### Baselines

1. Frozen VLA。
2. VLA SFT。
3. VLA + naive residual RL。
4. VLA + raw hidden-state RL。
5. VLA + manual phase residual RL。
6. Ours without correction advantage。
7. Ours without phase gate。
8. Ours full。

### Metrics

- Success rate。
- Completion time / throughput。
- Episodes to 80% success。
- Human interventions per 20 trials。
- Safety stops。
- Residual magnitude。
- Generalization to unseen object pose / object instance / lighting.

## 8. 预期贡献写法

贡献 1：

提出 post-hoc plug-in RL token，使冻结开源 VLA 无需原生 RL token 或端到端重训即可接入在线 RL。

贡献 2：

提出 correction-guided token learning，把人类接管前后的轨迹差异转化为 advantage/intervention supervision，提高小样本真实 RL 效率。

贡献 3：

提出 phase-gated residual adaptation，只在精细失败阶段修改 VLA action，保留 VLA 的粗粒度泛化能力并降低探索风险。

贡献 4：

在单臂夹爪真实精细操作任务上验证，展示相对 VLA SFT 和 naive residual RL 的成功率、样本效率和安全性提升。

## 9. 主要风险与规避

风险：开源 VLA 中间特征不好提取。
规避：先用 action chunk + visual encoder features + proprioception 做 token，再逐步接 hidden states。

风险：真实 RL 样本不够。
规避：先做 correction warmup 和离线 replay；在线只微调 residual actor。

风险：奖励不稳定。
规避：优先用可验证 sparse success 和人工 intervention labels，VLM progress reward 只作为辅助。

风险：任务太难导致 VLA 基线完全失败。
规避：任务必须让 VLA 能完成接近阶段，只让 RL 解决最后精细瓶颈。

风险：贡献被认为只是 RLT 工程版。
规避：实验必须证明 native token 不可用场景下，post-hoc token adaptor 优于 raw hidden state、naive residual RL 和 VLA-guided RL。

## 10. 下一步执行路线

1. 用 SmolVLA/LeRobot 跑通一个单臂任务的 VLA SFT。
2. 加 hooks 导出视觉特征、语言特征、action chunk。
3. 收集 20-50 条示教和 20-50 条 correction 数据。
4. 训练 token adaptor + phase gate。
5. 离线训练 residual actor，再上线小步 RL。
6. 做 3 个任务和完整消融。
