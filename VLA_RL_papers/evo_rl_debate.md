# Evo-RL 子智能体 Battle 后的方向重构

更新时间：2026-04-28
远程仓库：https://github.com/MINT-SJTU/Evo-RL
本地仓库：`E:\Desktop\科研\VLA_RL_papers\repos\Evo-RL`
关联文献库：`papers.csv`、`notes.md`、`direction_battle.md`

## 0. 一句话结论

不要把论文主线停在 **Correction-Guided Plug-in RL Tokens** 这个层级。这个说法能讲通，但容易被审稿人打成“RLT 降级版 + Evo-RL/HIL-SERL 接管数据 + ACP 标签工程”。

更强的题目应该收窄成：

**Counterfactual Boundary Tokens for Residual Reinforcement of Chunked VLAs**

中文可以写成：

**面向冻结 VLA 的反事实失败边界 Token 与阶段门控残差强化**

核心问题改成：

> PI 的 RLT 依赖 native RL token；Evo-RL 已经有接管记录、Pi*0.6 value 和 ACP 标签。普通实验室真正缺的是：如何从人类接管中学习“VLA 即将失败但仍可修复”的 failure boundary / recoverability，并只在这个边界窗口编辑 action chunk，而不是全程微调或复现 RLT。

## 1. 子智能体辩论结论

### A. Evo-RL 代码审计员

Evo-RL 已经实现了四个很有用的底座：

1. **Human-in-loop takeover 数据采集**
   `lerobot-human-inloop-record` 支持 policy 和 teleop 切换，记录 executed `action`、`complementary_info.policy_action`、`complementary_info.is_intervention`、`complementary_info.state`、`collector_policy_id`。
   关键入口：
   - `src/lerobot/scripts/lerobot_human_inloop_record.py`
   - `src/lerobot/scripts/recording_loop.py`
   - `src/lerobot/scripts/lerobot_record.py`

2. **Pi*0.6 value / progress 建模**
   `pistar06` 在仓库里是 value policy，不是 action VLA。它用视觉编码器、语言模型和 MLP value head 预测离散 value bins。
   关键入口：
   - `src/lerobot/values/pistar06/configuration_pistar06.py`
   - `src/lerobot/values/pistar06/modeling_pistar06.py`
   - `src/lerobot/scripts/lerobot_value_infer.py`

3. **ACP：Advantage-conditioned prompt**
   当前 ACP 不是 native token，也不是 RLT；它只是把二值标签拼进 task 文本：`Advantage: positive/negative`。
   关键入口：
   - `src/lerobot/rl/acp_tags.py`
   - `src/lerobot/rl/acp_hook.py`
   - `src/lerobot/scripts/recording_hil.py`

4. **RTC / chunk correction**
   RTC 在推理期对 action chunk 做 overlap consistency correction，适合拿来研究“chunk 后半段或边界窗口该不该被编辑”。
   关键入口：
   - `src/lerobot/policies/rtc/modeling_rtc.py`
   - `src/lerobot/policies/pi05/modeling_pi05.py`
   - `src/lerobot/policies/pi0/modeling_pi0.py`

明确没看到的东西：

- 没有 PI RLT 那种 native RL token。
- 没有 VLA base action + residual actor + critic 的统一训练栈。
- SAC/HILSERL 是独立 RL policy，不是接在 SmolVLA/PI05 后面的 residual RLT。
- 没有把 `action - policy_action` 显式转成 correction residual / counterfactual label 的模块。

### B. 创新性挑刺者

原题 **Correction-Guided Plug-in RL Tokens** 的风险：

- 容易被说成“RLT 去掉 native token 后的工程平替”。
- 接管、value、advantage、ACP 都已被 Evo-RL 类系统预占。
- hidden-state adaptor 不是贡献，很多开源 VLA 本来就能 hook 中间表征。
- 如果没有证明 token 学到的是特殊东西，raw hidden-state RL、naive residual RL、residual BC 都会成为强反驳。

真正值得押的创新点不是“能不能加 token”，而是：

- token 到底学什么；
- correction 提供了什么 reward/value 之外的信息；
- residual RL 应该何时获得控制权；
- action chunk 应该被整体改、局部改，还是只改失败边界的尾部窗口。

### C. 真实机器人可行性评估员

资源约束下，最稳的是：

- 基座优先 `SmolVLA`，其次 `PI05/OpenPI`。
- 任务选 VLA 能完成粗接近、RL 只补最后几厘米的精细任务。
- 不要第一阶段做 Evo1 stage2、OpenVLA 7B 全参、大规模 world model、完整 RECAP/RLT 复现。

建议任务：

1. 按钮按压 / 拨杆开关。
2. peg-in-hole / 假 USB 插入 / 对孔插接。
3. 抽屉把手勾住后拉开小行程。

建议数据预算：

- 每任务 `30-50` 条成功示教。
- 每任务 `20-40` 条接管/纠错轨迹。
- 每任务 `100-200` 个在线 episodes。
- 每任务至少 `20` 次正式评测，外加 3 组未见物体位姿。

## 2. 最终 Battle 排名

### 方案 1：Counterfactual Boundary Tokens + Boundary-Gated Chunk Residual

推荐级别：**最高，作为主论文方向。**

核心思想：

不是学普通 RL-state token，而是从接管数据中学一个“反事实失败边界表征”：

- 如果不接管，VLA 大概率会怎么失败？
- 当前状态是否仍可由小修正救回来？
- 需要修正 action chunk 的哪一段？
- 需要的最小 residual 是什么方向和幅度？

这让论文问题从“给开源 VLA 接一个 token”升级为：

> 如何从人类接管中识别 chunked VLA 的可修复失败边界，并把在线 RL 限制在这个边界窗口内。

#### 方法设计

基座：

- `SmolVLA` 作为第一实现目标。
- `PI05/OpenPI` 作为第二验证目标。
- 冻结 VLA 主干，最多允许 PEFT/LoRA 作为对照，不作为主贡献。

数据来源：

- VLA rollout 时记录 `policy_action`。
- 人类接管时记录 executed `action`。
- 用 `delta = action - policy_action` 得到 correction residual。
- 用 `is_intervention`、接管开始/结束、episode success/failure、value/advantage 生成监督信号。

Token / 表征学习目标：

- `boundary_label`：正常、接近失败边界、接管中、恢复中。
- `recoverability`：接管后是否成功恢复，或者 value 是否显著上升。
- `correction_magnitude`：none / small / medium / large。
- `correction_direction`：末端位姿或关节空间中的主要修正方向。
- `residual_target`：接管帧上的 `action - policy_action`。
- `advantage_delta`：接管前后 value/progress 的变化。

控制结构：

```text
observation, task, proprio, a_vla_chunk, VLA features
        |
counterfactual boundary token adaptor
        |
boundary gate  ---- no ----> execute VLA action
        |
       yes
        |
residual actor/editor
        |
a_exec = a_vla + clip(delta_a)
```

关键限制：

- residual actor 只在 boundary gate 开启时工作。
- 优先编辑 chunk 的尾部或局部窗口，不全程覆盖 VLA。
- residual 有幅度 clipping、action smoothness、reference regularization。
- critic/reward 用 sparse success、progress value、intervention penalty，不把 VLM reward 当主依赖。

#### 为什么比原题更强

和 RLT 的区别：

- RLT 是 PI 内部 native RL token；这里是 post-hoc，从开源 VLA 和接管数据中学习。
- RLT 默认知道精细阶段；这里自动学习 failure boundary 和 recoverability。
- RLT 强调小 actor-critic；这里强调 correction counterfactual 和 chunk-local control authority。

和 Evo-RL/ACP 的区别：

- Evo-RL 当前 ACP 只是 `Advantage: positive/negative` 文本标签。
- 本方案不是简单二值 advantage，而是学习接管暴露出的失败边界、恢复性和最小修正。
- Evo-RL 没有 VLA residual actor-critic；本方案把 residual editor 接到 frozen VLA action chunk 后。

和 naive residual RL 的区别：

- naive residual 全程探索，危险且样本效率低。
- 本方案只在“高 recoverability、低探索风险”的边界窗口启用。
- 需要用实验直接证明 gate/token 比 raw hidden-state residual 更省样本、更少危险停止。

#### 最小实现路径

第一阶段不硬上完整 actor-critic，先做可验证的 ACP++：

1. 从 HIL 数据生成 correction labels。
2. 把 `Advantage: positive/negative` 扩展成多粒度文本标签，例如：
   - `Boundary: normal/failure/recovering`
   - `Recoverability: low/high`
   - `Correction: none/small/large`
3. 复用 `acp_hook.py` 和 `recording_hil.py`，先验证多粒度 correction token 是否优于二值 ACP。

第二阶段加 residual editor：

1. 训练小 MLP/Transformer adaptor 输出 `z_boundary`。
2. 用接管帧做 residual BC warmup。
3. 用 TD3+BC / SAC / AWAC 风格在线微调 residual actor。
4. gate 控制 residual 权限，只改 chunk 局部窗口。

#### 主要代码入口

- 标签生成：`src/lerobot/scripts/lerobot_value_infer.py`
- 接管数据字段：`src/lerobot/scripts/recording_loop.py`
- 训练 prompt hook：`src/lerobot/rl/acp_hook.py`
- token 字符串定义：`src/lerobot/rl/acp_tags.py`
- 闭环推理 ACP/CFG：`src/lerobot/scripts/recording_hil.py`
- PI05 action chunk：`src/lerobot/policies/pi05/modeling_pi05.py`
- SmolVLA policy：`src/lerobot/policies/smolvla/`
- RTC chunk correction：`src/lerobot/policies/rtc/modeling_rtc.py`
- 独立 SAC 可参考但不能直接当 VLA residual：`src/lerobot/policies/sac/`

### 方案 2：ACP++ Correction Tokens

推荐级别：**最容易落地，适合做第一篇实验的 MVP 或主方案的第一阶段。**

核心思想：

把 Evo-RL 已有的二值 ACP：

```text
Advantage: positive / negative
```

升级成 correction-conditioned prompt：

```text
Advantage: positive
Boundary: failure
Recoverability: high
Correction: small-z-down
```

优点：

- 改动集中在 `acp_tags.py`、`acp_hook.py`、`lerobot_value_infer.py`。
- 不需要新增完整 actor-critic 栈。
- 能快速验证 correction labels 是否真的比二值 advantage 有效。

风险：

- 容易被说成 prompt engineering。
- 如果不接 residual RL，和 RLT 的关系会变弱。
- 必须用干净 ablation 证明多粒度 correction label 不是噪声。

适合定位：

- 作为主方案的 warmup。
- 作为论文中的 “cheap post-hoc RLT interface” baseline。
- 如果时间不够，ACP++ 可以单独写成 workshop / 小论文方向。

### 方案 3：RTC-aware Correction Curriculum

推荐级别：**中等，适合作为辅助贡献。**

核心思想：

RTC 已经在推理期对 chunk overlap error 做 correction，但它没有反哺训练。可以把 RTC correction norm、overlap inconsistency、chunk tail error 当作自监督信号：

- 哪些状态下 chunk 执行不稳定？
- 哪些时刻需要更短 horizon 或更强 residual？
- RTC correction 是否能预测未来接管？

优点：

- 和 action chunk / flow matching VLA 结合紧。
- 不完全依赖人工标签。
- 可作为 boundary gate 的输入特征。

风险：

- 独立成文偏工程。
- 需要能稳定导出 RTC debug 信息。
- 不如人类接管的 counterfactual 信号直观。

建议：

不要单独做主线，把它并入方案 1 的 gate 特征。

### 方案 4：Universal Post-hoc RL Interface for Open-Source VLAs

推荐级别：**学术味更足，但不适合作为第一阶段。**

核心思想：

在 SmolVLA、PI05/OpenPI、OpenVLA 上学习统一 `z_ctrl` 接口，使不同 VLA 都能被 residual RL 控制。

优点：

- 比单模型方案更像“接口协议”贡献。
- 能更强地回答“普通实验室如何给任意开源 VLA 接 RLT-style RL”。

风险：

- 实验量明显增加。
- 单 GPU 和真实机器人时间不一定撑得住。
- 不同 backbone 差异可能掩盖方法贡献。

建议：

只作为后续扩展，不作为当前小论文主线。

### 方案 5：Intervenability-Aware RL

推荐级别：**概念不错，适合作为 critic/gate 的辅助目标。**

核心思想：

把接管数据看成“可介入性先验”，而不是简单 reward：

- 当前状态是否值得探索？
- 接管后是否能救回来？
- 预计需要多大 correction？
- 人类接管成本多高？

这可以塑形 residual RL 的探索空间，让 actor 优先访问 historically recoverable states。

风险：

- 概念抽象，不好量化。
- 如果没有真实机器人安全收益，容易被说成 intervention predictor 换名。

建议：

作为方案 1 的 `recoverability` head 和 gate 约束，不单独成题。

## 3. 最推荐论文版本

### 题目

英文：

**Counterfactual Boundary Tokens for Residual Reinforcement of Chunked Vision-Language-Action Policies**

备用标题：

**Learning Recoverable Failure Boundaries for Plug-in RL on Frozen VLAs**

中文：

**面向冻结 VLA 残差强化的反事实失败边界学习**

### 核心贡献写法

贡献 1：

提出 counterfactual boundary token，把人类接管数据转化为 failure boundary、recoverability、correction residual 三类监督，而不是只做二值 advantage。

贡献 2：

提出 boundary-gated residual editor，只在可恢复的精细失败边界窗口编辑 action chunk，降低真实机器人在线 RL 的探索成本和安全风险。

贡献 3：

基于 Evo-RL/LeRobot 的开源实现，在 SmolVLA/PI05 上验证：无需 native RLT token、无需 VLA 端到端重训，也能获得 RLT-style 的小样本真实机器人强化收益。

## 4. 实验计划

### 主任务

优先顺序：

1. 按钮按压 / 拨杆开关。
   成功判定简单，reset 成本低，适合第一阶段。
2. peg-in-hole / 假 USB 插入。
   最贴近 RLT 的精细接触卖点，但失败率和 reset 成本更高。
3. 抽屉把手拉开小行程。
   适合展示视觉定位和接触阶段分离。

任务必须满足：

- VLA 能完成粗接近。
- 失败集中在最后精细阶段。
- 成功可以自动或半自动判定。
- reset 不要太贵。

### Baselines

必须有：

1. `VLA BC / PEFT only`
2. `VLA + binary ACP`
3. `VLA + ACP++ correction prompt`
4. `VLA + naive residual RL`
5. `VLA + raw hidden-state residual RL`
6. `VLA + manual phase residual`
7. `Ours: counterfactual boundary token + gated residual editor`

可选：

8. `HIL-SERL SAC from demos`，作为无 VLA 先验的 RL 对照。
9. `Ours without recoverability head`
10. `Ours without chunk-local edit`

### Metrics

核心指标：

- 成功率。
- 到达 70% / 80% 成功率所需真实 episodes。
- 每 20 次 rollout 的人工接管次数。
- 危险停止次数。
- 平均完成时间。
- residual magnitude。
- 对未见物体位姿、未见物体实例、轻微光照变化的泛化。

论文最低目标：

- 3 个真实精细任务里至少 2 个任务有稳定提升。
- 相比 VLA BC / PEFT 成功率提升 15%-25%。
- 相比 naive residual RL 真实交互样本减少 30%-50%。
- 接管次数和危险停止次数不高于 naive residual RL。

## 5. 2-3 个月执行路线

第 1-2 周：

- 跑通 Evo-RL 的 HIL recording 和 dataset report。
- 先用 SmolVLA/PI05 在一个按钮或拨杆任务上得到 VLA baseline。
- 验证能稳定写入 `policy_action`、`is_intervention`、episode success。

第 3 周：

- 写 correction label generator：
  - `delta = action - policy_action`
  - takeover onset / release
  - correction magnitude
  - recoverability
  - boundary window
- 先生成离线统计图：接管前后 value、delta norm、成功率变化。

第 4 周：

- 实现 ACP++ prompt tokens。
- 对比 binary ACP vs ACP++。
- 如果 ACP++ 完全无收益，立刻回到 residual-only 路线，不继续堆 prompt。

第 5-6 周：

- 实现 boundary gate 和 residual BC warmup。
- residual editor 先不用复杂 RL，先做 supervised correction residual。
- 对比 raw hidden-state residual BC。

第 7-8 周：

- 加小规模在线 RL：SAC / TD3+BC / AWAC 风格均可，重点是 residual action clipping 和 gate。
- 只在一个任务上调到稳定。

第 9-10 周：

- 扩展到第二、第三个任务。
- 做 ablation：无 gate、无 recoverability、无 correction residual、全程 residual、manual phase。

第 11-12 周：

- 整理图表、失败案例、消融。
- 写论文主线：不要写“复现 RLT”，写“RLT-style capability without native RLT tokens”。

## 6. 审稿人攻击与防线

攻击 1：这不就是 RLT 的开源降级版？

防线：

- RLT 需要 VLA native token，本方案不需要。
- 本方案的 token 不是普通 state，而是从接管反事实中学习 failure boundary 和 recoverability。
- 必须用 raw hidden-state residual baseline 证明 post-hoc tokenization 不是接口工程。

攻击 2：这不就是 Evo-RL ACP 加了更多 prompt？

防线：

- ACP++ 只是第一阶段；主方法还包含 boundary gate 和 residual action editor。
- 需要证明 correction token 可以预测接管、恢复性和 residual，而不只是提高 imitation loss。

攻击 3：这不就是 HIL-SERL？

防线：

- HIL-SERL 学独立 RL policy；本方案冻结 VLA，只做边界窗口 residual 编辑。
- 对比 HIL-SERL SAC from demos，展示 VLA prior 带来的样本效率。

攻击 4：phase gate 是 heuristic。

防线：

- gate 要作为学习模块，用 intervention onset、recoverability、value stagnation 监督。
- 对比 manual phase gate、always-on residual、no gate。

攻击 5：任务太简单，raw residual 也能做。

防线：

- 至少做两个需要精细接触的任务。
- 报告 sample efficiency、安全停止、接管次数，而不只看最终成功率。

## 7. 当前最应该做的下一步

不要直接开写完整 residual actor-critic。先做一个能最快验证创新信号的 MVP：

1. 选一个低 reset 成本任务：按钮 / 拨杆。
2. 用 Evo-RL HIL 流程收集：
   - 30 条成功示教；
   - 30 条 VLA rollout + 人类接管轨迹；
   - 每条记录保留 `policy_action` 和 executed `action`。
3. 生成 `delta = action - policy_action` 和 boundary/recoverability 标签。
4. 做 binary ACP vs ACP++ correction token 的第一轮实验。
5. 如果 ACP++ 有效，再加 boundary-gated residual editor；如果无效，直接把重点转向 residual editor 和 gate，不在 prompt token 上继续投入。

最终定位：

**Plug-in RLT 仍然是动机，但论文创新点应落在 counterfactual boundary learning + gated chunk residual，而不是“给 VLA 接一个 token”。**
