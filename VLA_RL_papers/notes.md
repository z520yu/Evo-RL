# VLA+RL 文献笔记与方向整理

更新时间：2026-04-28
目标：围绕单臂夹爪、1 张 GPU、少量示教、真实在线交互，寻找区别于 RLT/RECAP 复现的可发表方向。

## 0. 核心判断

VLA+RL 的热点已经从“能不能 RL 微调 VLA”转向三个更具体的问题：

1. 大模型 VLA 如何从真实经验持续变强：RECAP、RLT、RL-100、GR-RL 已经给了强证据。
2. 奖励/critic 如何替代人工奖励工程：VLAC、SOLE-R1、VLA-RL 都在抢这个位置。
3. 普通实验室如何在开源 VLA 上低成本做真实机器人后训练：这里仍然空缺最大。

因此最有价值的题目不是“复现 RLT/RECAP”，而是：

**Correction-Guided Plug-in RL Tokens for Open-Source Vision-Language-Action Policies**

一句话版本：不用改 VLA 预训练，也不用端到端重训大模型，只在冻结的 OpenPI/SmolVLA/OpenVLA 外面后接一个可训练的 RL-state token 和 residual actor-critic，让单臂夹爪在精细失败阶段通过少量人类 correction 和真实交互快速变好。

## 1. PI 主线：RECAP 与 RLT 到底做了什么

### pi*0.6 / RECAP

文件：`pdfs/pistar06_recap_2511.14759.pdf`
链接：https://arxiv.org/abs/2511.14759

RECAP 的关键不是简单在线 RL，而是把三类数据统一进一个 advantage-conditioned policy：

- Demonstrations：先让 VLA 学会基本策略。
- Corrections：运行当前策略，人在失败状态接管，提供“政策实际会到达的错误状态”上的监督。
- Autonomous experience：机器人自己反复执行，根据 outcome/reward 学 value function。

核心机制：

- 训练 value function 预测任务进度或到成功的距离。
- 用 value change 得到 advantage。
- 把 advantage 作为条件输入给 VLA，训练时保留好轨迹和坏轨迹，推理时要求高 advantage 动作。

对我们方向的启发：

- Correction 比普通 demonstration 更值钱，因为它覆盖模型真实失败分布。
- Advantage 条件比“只模仿成功轨迹”更适合从坏数据中学习。
- 但 RECAP 依赖 PI 自有大模型、数据、训练系统，不适合作为普通实验室复现目标。

### RLT

文件：`pdfs/rlt_pi_rl_token.pdf`
链接：https://www.pi.website/research/rlt

RLT 针对的是长程任务中的精细瓶颈阶段，例如插网线、拧螺丝、插电源。它的核心是：

- 冻结 VLA 主体。
- 让 VLA 输出一个 RL token，作为小 actor/critic 的紧凑状态表示。
- 小 actor/critic 在线 RL 更新，用很少真实数据提升速度和精度。
- Actor 不是从零控制，而是编辑 VLA action chunk。

RLT 的关键限制：

- RL token 是 native token，需要模型方在训练或适配阶段提供内部接口。
- 适配对象是 PI 的 pi0.6，不是任意开源 VLA。
- 论文主要证明 PI 模型内部接口有效，没有回答“开源黑盒/半黑盒 VLA 怎么办”。

我们的切口：

**Native RLT asks: can a VLA trained to emit an RL token support online RL?**
**Plug-in RLT asks: can an existing frozen VLA be post-hoc tokenized for online RL without native token support?**

这个问题更贴近普通实验室，也更有独立性。

## 2. VLA 后训练 RL：热，但直接卷不划算

### VLA-RL

文件：`pdfs/vla_rl_2505.18719.pdf`
链接：https://arxiv.org/abs/2505.18719

贡献：

- 把自回归 VLA 训练表述为 trajectory-level RL。
- 用 process reward model 缓解稀疏奖励。
- 在 LIBERO 40 个任务上提升 OpenVLA。

局限：

- 偏大规模模拟环境和系统工程。
- 核心创新在 VLA RL 训练框架，不适合单臂真实小实验直接超越。

### SimpleVLA-RL

文件：`pdfs/simplevla_rl_2509.09674.pdf`
链接：https://arxiv.org/abs/2509.09674

贡献：

- 基于 veRL 做 VLA-specific trajectory sampling、并行渲染和优化。
- 在 OpenVLA-OFT、LIBERO、RoboTwin 上展示 RL 能超过 SFT。

对我们方向的用处：

- 可以引用为“RL post-training is becoming the standard path”。
- 但我们的定位应避开大规模训练框架，强调真实小样本与开源 VLA plug-in。

### CO-RFT

文件：`pdfs/co_rft_2508.02219.pdf`
链接：https://arxiv.org/abs/2508.02219

贡献：

- 面向 action chunk 的 offline RL。
- 30-60 条示教后再用 chunked TD 做离线强化。

对我们方向的压力：

- 它已经覆盖了“少示教 + action chunk + offline RL”。
- 所以我们不能只做 chunked offline RL；必须突出 human correction、phase gate 和 post-hoc RL token。

### VLA-RFT / World-Env

文件：`pdfs/vla_rft_2510.00406.pdf`、`pdfs/world_env_2509.24948.pdf`

贡献：

- 用 world model/virtual environment 代替真实交互。
- 降低真实机器人 reset 和安全成本。

局限：

- world model 本身是大坑，容易把论文变成另一个不可控系统。
- 单臂夹爪真实平台反而是优势，没必要绕远路做完整虚拟环境。

## 3. Reward / Critic 主线：可用，但不宜作为主贡献

### VLAC

文件：`pdfs/vlac_2509.15937.pdf`
链接：https://arxiv.org/abs/2509.15937

贡献：

- 输入 pairwise observations + language goal，输出 progress delta 和 done signal。
- 用异步真实 RL loop + human-in-loop 改善样本效率。

对我们方向的意义：

- Progress critic 是合理辅助模块。
- 但如果我们主打“VLA critic/reward model”，会和 VLAC 正面撞车。

### SOLE-R1

文件：`pdfs/sole_r1_2603.28730.pdf`
链接：https://arxiv.org/abs/2603.28730

贡献：

- 用 video-language reasoning model 直接作为 dense reward。
- 强调普通 VLM reward 在 partial observability 和 distribution shift 下会被 RL reward hacking。

对我们方向的意义：

- 强化了“纯 VLM reward 不可靠”的论点。
- 我们可以用 progress critic，但必须加入 correction/intervention 信号和安全约束，不能只靠 VLM 打分。

## 4. 真实机器人精细强化：最贴近单臂夹爪

### RL-100

文件：`pdfs/rl100_2510.14830.pdf`
链接：https://arxiv.org/abs/2510.14830

贡献：

- imitation -> offline RL -> online RL 的三阶段框架。
- 在线 RL 用来消除残余 failure modes。
- 展示高成功率和高吞吐真实任务。

对我们方向的意义：

- 说明真实机器人在线 RL 的价值已经被认可。
- 但 RL-100 不是 VLA plug-in，给我们留下“VLA 表征如何服务真实 RL”的空间。

### GR-RL

文件：`pdfs/gr_rl_2512.01801.pdf`
链接：https://arxiv.org/abs/2512.01801

贡献：

- 把 generalist VLA 转成 long-horizon dexterous specialist。
- 用 progress filtering、symmetry augmentation 和 latent-space online RL。

对我们方向的压力：

- 它已经强调 VLA -> precise specialist。
- 我们必须把创新点落在“post-hoc token interface + correction advantage + phase gating”，而不是泛泛说精细操作。

### RPD 与 VLAJS

文件：`pdfs/rpd_vla_to_rl_experts_2503.05833.pdf`、`pdfs/vlajs_2604.13733.pdf`

RPD：用 VLA teacher 指导小 RL student，student 可超过 teacher。
VLAJS：用 VLA sparse guidance jump-start PPO，之后逐渐退火，让 RL 超过 VLA。

对我们方向的区分：

- RPD/VLAJS 把 VLA 当 teacher 或 regularizer。
- Plug-in RLT 把 VLA 当 frozen foundation policy，同时学习一个 compact token 来决定如何 residual-correct VLA action。
- 如果实验只证明“VLA 指导 RL 有用”，会被 RPD/VLAJS 吃掉；必须证明 token/adaptor/correction 的必要性。

## 5. 评测与部署相关

### ManipArena / vla-eval / RobotArena infinity

文件：`pdfs/maniparena_2603.28545.pdf`、`pdfs/vla_eval_2603.13966.pdf`、`pdfs/robotarena_inf_2510.23571.pdf`

意义：

- VLA 论文越来越重视可比评测，而不是只放 demo。
- 即使我们做真实单臂，也要报告固定 reset、固定 unseen splits、失败类型统计、接管次数和安全停止次数。

### FocusVLA / StreamingVLA

文件：`pdfs/focusvla_2603.28740.pdf`、`pdfs/streamingvla_2603.28565.pdf`

意义：

- FocusVLA 提醒我们：视觉 token 的选择会直接影响动作质量。Plug-in token adaptor 不应简单平均 hidden states，而应比较 task-focused pooling。
- StreamingVLA 提醒我们：action chunk、推理延迟和执行异步会影响真实控制。Residual actor 必须和 action chunk 对齐。

## 6. 方向 Battle

### A. 复现 RECAP

优点：热，和 PI 对齐。
问题：需要完整 VLA 训练、value model、大量真实部署数据和内部模型接口。
结论：淘汰。最多作为 related work。

### B. 复现 RLT

优点：非常贴合单臂精细操作。
问题：原生 RL token 需要模型内部训练支持；直接复现会变成“把 RLT 换一个机械臂跑”。
结论：不能作为主贡献，但可以作为方法灵感。

### C. VLA Progress Critic

优点：容易讲，和 RL 结合自然。
问题：VLAC、SOLE-R1 已经占位，而且 reward model 训练本身需要数据。
结论：作为辅助 reward/diagnostic，不作为主贡献。

### D. VLA-guided RL / Distillation

优点：实现相对直接。
问题：RPD、VLAJS 已经覆盖 VLA teacher -> RL student 的路线。
结论：作为 baseline，不作为主线。

### E. Plug-in RL Token + Correction Advantage + Phase Gate

优点：

- 和 RLT/RECAP 关系强，热点足。
- 不依赖 PI 私有模型，适合普通实验室。
- 真实单臂小样本能做出结果。
- 和现有工作区分清楚：不是 native RL token，不是 full VLA RL fine-tune，不是 pure VLM reward。

风险：

- 需要能稳定提取开源 VLA 中间特征。
- 真实机器人实验要设计得够窄，否则系统复杂度失控。

结论：主推。

## 7. 建议论文方案

### 题目

**Correction-Guided Plug-in RL Tokens for Open-Source Vision-Language-Action Policies**

### 核心假设

冻结 VLA 内部包含足够的任务语义和动作先验，但这些信息没有被组织成适合在线 RL 的紧凑状态。通过后验 token adaptor，可以把 VLA 表征压缩为 RL-friendly state token，再用 human correction 和少量真实交互训练 residual actor-critic。

### 系统结构

1. Frozen VLA policy
   输入 observation、language goal、proprioception，输出 action chunk。

2. Plug-in token adaptor
   输入 VLA hidden states、视觉 token、语言 token、proprioception、VLA action chunk、最近执行误差，输出低维 `z_rl`。

3. Phase gate
   根据 progress stagnation、intervention probability、action uncertainty、task phase embedding 判断是否启用 residual actor。

4. Residual actor-critic
   Actor 输出 `delta_a`，最终动作为 `a = a_vla + clip(delta_a)`。Critic 学 `Q(z_rl, delta_a)` 或 progress value。

5. Correction advantage
   用接管前后轨迹、任务进度变化、success/failure、intervention event 构造训练信号。

### 训练流程

Stage 0：VLA SFT 或直接使用开源 checkpoint。
Stage 1：收集每任务 20-50 条示教和少量失败轨迹。
Stage 2：运行 VLA，人在明显失败时接管，记录 pre-intervention state、robot action、human correction、post-correction outcome。
Stage 3：训练 token adaptor 和 phase gate：

- intervention prediction loss
- progress delta regression loss
- correction residual behavior cloning loss
- optional contrastive loss：成功进展 token 拉近，停滞/退步 token 拉远

Stage 4：在线 RL 只更新 residual actor-critic，不动 VLA 主干。
Stage 5：可选离线回放，把成功 residual 轨迹蒸馏回 adaptor/actor。

### 最小实验任务

选择 3 个单臂夹爪精细失败阶段：

- 插入/对孔：peg-in-hole、USB/Type-C mock plug、圆柱插孔。
- 按压/接触：按钮按压、开关拨动、触点压合。
- 拉取/对齐：抽屉把手拉取、堆叠对齐、工具尖端对准。

每个任务要有：

- 基础阶段 VLA 能大致完成，例如接近目标。
- 精细阶段 VLA 经常失败，例如最后 1-3 cm 对不准。
- 成功判定可自动化或半自动化，避免全靠人工主观评分。

## 8. 必做消融

1. `VLA SFT`：只微调或直接运行基座。
2. `VLA + naive residual RL`：不用 plug-in token，只用原始状态或视觉 embedding。
3. `VLA + raw hidden-state RL`：直接拼接 hidden states，不训练 token adaptor。
4. `VLA + manual phase RLT-style`：人工指定关键阶段，不用 phase gate。
5. `Ours w/o correction advantage`：只用 sparse success reward。
6. `Ours w/o phase gate`：全程 residual correction。
7. `Ours full`：plug-in token + correction advantage + phase gate。

## 9. 论文卖点

主张不要过大：

- 不是说超过 PI RLT。
- 不是说解决通用 VLA 后训练。
- 不是说 reward model 全自动。

主张应该是：

1. Existing open-source VLAs can be post-hoc converted into RL-adaptable policies through a compact plug-in token interface.
2. Human corrections provide a practical advantage signal for learning this interface under small real-robot budgets.
3. Phase-gated residual RL improves precision bottlenecks while preserving the broad competence of the frozen VLA.

## 10. 最低可发表标准

实验上至少满足：

- 3 个真实任务，至少 2 个任务显著提升。
- 相比 VLA SFT，成功率提升 15%-25%。
- 相比 naive residual RL，真实交互 episode 减少 30%-50%。
- 人工接管次数随训练下降。
- 安全停止次数不高于 naive RL。

如果真实实验规模不足，可以加模拟补充，但真实单臂结果必须作为主结果。
