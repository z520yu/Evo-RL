# Value-Guided RTC 实验设计

## 1. 研究问题与核心假设

Value-Guided RTC 的目标是在 action chunk 执行过程中，不再固定执行完整 chunk，而是利用 chunk 内的 Q/value profile 判断：

- 当前 chunk 还值得继续执行时，减少 re-observe/replan，保留开环 chunk 的速度优势。
- 当前 chunk 的未来价值明显下降或不确定性升高时，提前 re-observe/replan，避免错误动作持续累积。

核心假设：

1. VLA 或 diffusion policy 的 action chunk 内部存在可预测的价值形态，例如价值单调衰减、局部断崖、末端接触不确定性升高。
2. 用 critic 估计每个 chunk step 的 Q/value、advantage、uncertainty，可以比固定 stride 或固定频率 RTC 更好地决定 replan 时机。
3. Value-Guided RTC 的收益主要来自两类场景：长时程多阶段任务中的早期纠错，以及接触丰富任务中的错误动作截断。

## 2. 方法定义

给定观测 `o_t`，策略输出长度为 `H` 的 action chunk：

```text
a_{t:t+H-1} = pi(o_t)
```

同时训练 critic 估计 chunk 内每一步继续执行当前 chunk 后的价值：

```text
V_i = V(o_t, a_{t:t+i}) 或 Q_i = Q(o_t, a_{t:t+i}, a_{t+i})
```

执行时维护一个 replan gate。若满足以下任一条件，则提前停止当前 chunk，重新观测并规划：

- `Q_i < tau_abs`：绝对价值低于阈值。
- `Q_i - Q_0 < -tau_drop`：相对起始价值下降过大。
- `Q_i - Q_{i-1} < -tau_slope`：局部价值斜率出现断崖。
- `U_i > tau_unc`：ensemble critic 或 dropout critic 的不确定性过高。
- `A_i = Q_i - baseline(o_{t+i}) < tau_adv`：继续执行当前 chunk 相对重新规划的优势为负。

推荐论文主方法：

```text
replan if sigmoid(alpha * drop_i + beta * unc_i + gamma * contact_i) > p_replan
```

其中 `drop_i = max(0, Q_0 - Q_i)`，`unc_i` 来自 critic ensemble 方差，`contact_i` 是可选的接触阶段特征。最小实验阶段可以先使用简单阈值 gate。

## 3. 仿真实验

### 3.1 任务选择

优先使用三个层次的任务，避免只在简单 pick-place 上得到局部结论。

**最小可行仿真实验：LIBERO**

- 推荐任务：`LIBERO-Spatial`、`LIBERO-Object`、`LIBERO-Goal` 中选择 8 到 12 个任务。
- 任务类型：pick-place、open/close drawer、turn on/off object、object arrangement。
- 选择理由：任务多样、成功率评估成熟、演示数据可直接使用，适合快速验证 value gate 是否优于固定 chunk stride。

**中等规模仿真实验：ManiSkill**

- 推荐任务：`PickCube-v1`、`StackCube-v1`、`PegInsertionSide-v1`、`TurnFaucet-v1`、`OpenCabinetDrawer-v1`。
- 任务类型：抓取、堆叠、插孔、旋转、抽屉接触。
- 选择理由：状态和视觉观察都可用，便于训练 state critic 与 visual critic 做上界/下界比较。

**理想论文实验：LIBERO + RoboTwin + ManiSkill**

- LIBERO：验证语言条件、多任务泛化。
- RoboTwin：验证双臂或复杂长时程操作中的 replan 时机，尤其是协同、遮挡、接触切换。
- ManiSkill：提供标准化模拟器控制和可解释状态，用于分析 value profile 与真实物理失败模式的对应关系。

任务分组建议：

| 分组 | 代表任务 | 主要检验点 |
| --- | --- | --- |
| 短程刚体操作 | PickCube, LIBERO pick-place | 是否减少不必要 replanning |
| 接触丰富操作 | PegInsertion, TurnFaucet, OpenDrawer | 是否提前截断错误接触动作 |
| 多阶段长程任务 | LIBERO-Goal, RoboTwin long-horizon | 是否在阶段转换处自适应 re-observe |
| 分布偏移任务 | 目标位置扰动、物体形状扰动、相机扰动 | value gate 是否提升鲁棒性 |

### 3.2 数据生成

训练数据需要包含成功轨迹、失败轨迹和“局部偏离后恢复”的轨迹，否则 critic 很难学到 chunk 中间的价值下降。

**数据来源**

1. 专家演示：使用 LIBERO/RoboTwin/ManiSkill 原始 demo，训练基础 chunk policy。
2. 策略 rollout：用训练好的 chunk policy 在模拟器中 rollout，收集成功和失败轨迹。
3. 扰动 rollout：在执行 chunk 时加入扰动，制造价值下降样本。
4. Recovery rollout：在扰动后允许高频 replan 或专家接管，记录能否恢复。

**扰动设计**

- 观测扰动：相机噪声、遮挡、目标位置随机化。
- 状态扰动：物体位置偏移 1 到 5 cm，姿态偏移 5 到 20 度。
- 动作扰动：chunk 内随机 step 注入 action noise，或跳过某个微动作。
- 执行扰动：随机延迟 gripper close/open，模拟低成本硬件时序误差。

**数据规模建议**

最小可行：

- 每个任务 50 条专家 demo。
- 每个任务 200 到 500 条 policy rollout。
- 每个任务 200 条扰动 rollout。
- 总计 8 个任务，约 4k 到 8k 条轨迹。

理想论文：

- 每个任务 100 到 200 条专家 demo。
- 每个任务 1k 到 3k 条 policy rollout。
- 每个任务 1k 条扰动/recovery rollout。
- 总计 30 到 60 个任务，约 60k 到 180k 条轨迹。

### 3.3 Critic 训练

**输入形式**

推荐从简单到复杂分三档：

1. State critic：输入 simulator state、task id/language embedding、chunk prefix。
2. Visual critic：输入 RGB-D 或 RGB、language embedding、chunk step index、action prefix。
3. Policy-attached critic：复用 VLA/diffusion policy encoder，在 action chunk head 旁接 value head。

**监督信号**

可并行比较三类 critic target：

- Monte Carlo return：`G_t = sum gamma^k r_{t+k}`，用于稀疏成功任务时可解释性强。
- Success probability：预测从当前 chunk step 继续执行后的最终成功概率。
- Recovery value：预测“继续当前 chunk”相对于“立即 replan”的成功率差值。

推荐主 target 使用 success probability，因为机器人任务的 reward 常稀疏，最终成功率更贴近 RTC 决策。

**chunk 内标签构造**

对每条轨迹，在每个观测点 `t` 采样策略 chunk，并对 chunk step `i` 构造标签：

```text
y_{t,i} = episode_success
```

更强的版本使用 counterfactual rollout：

```text
y_continue(t,i) = 从 t 执行 chunk 前 i+1 步后继续低频执行的成功率
y_replan(t,i) = 从 t 执行 chunk 前 i+1 步后立即重新观测规划的成功率
A_replan(t,i) = y_replan(t,i) - y_continue(t,i)
```

最小实验可不做昂贵 counterfactual，只用 rollout 结果训练 `P(success | o_t, chunk, i)`，再用 value drop 作为 gate。

**损失函数**

- 二分类成功概率：binary cross entropy。
- 标量 return：Huber loss 或 MSE。
- 排序关系：pairwise ranking loss，让成功 chunk prefix 的 value 高于失败 chunk prefix。
- 不确定性：训练 3 到 5 个 critic ensemble，使用均值做 value、方差做 uncertainty。

**防止 critic 只学任务难度**

- 输入中显式包含 chunk step index 和 action prefix。
- 对同一初始观测采样多个 chunk，包括 expert chunk、policy chunk、noisy chunk。
- 使用 hard negative：看起来接近专家但 gripper 时序错误、接触方向错误、末端偏移过大的 chunk。
- 评估 value calibration：按预测 success probability 分桶，计算实际成功率。

### 3.4 Baseline

至少包含以下对照组：

1. Full chunk：固定执行完整 chunk，不提前 replan。
2. Fixed stride RTC：每 `k` 步 re-observe/replan，例如 `k=1,2,4,8`。
3. Confidence RTC：用 policy action variance、diffusion score、VLA token logprob 或 entropy 判断 replan。
4. Heuristic RTC：接触阶段、距离目标、gripper 状态变化等手工规则。
5. Oracle RTC：用模拟器 ground-truth success/recovery rollout 选择 replan，上界对照。
6. Value-Guided RTC：本文方法。

如果 compute 允许，增加：

- Learned replan classifier：直接预测是否 replan，不显式建模 value profile。
- State-value only：只用 `V(o_t)`，不看 chunk prefix，用于证明 chunk 内 profile 的必要性。
- Always high-frequency：每一步 replan，作为成功率上界和速度下界。

### 3.5 Metric

主指标：

- Success rate：每任务 50 到 100 episodes。
- Normalized execution time：完成任务所需环境 step 或真实秒数。
- Replan count：每 episode 平均 re-observe/replan 次数。
- Action waste：失败前仍继续执行低价值 chunk 的步数。
- Contact failure rate：碰撞、推偏、夹爪空抓、插孔卡死等。

综合指标：

```text
Efficient Success = SuccessRate - lambda * NormalizedReplanCount
```

或报告 Pareto 曲线：x 轴为平均 replan 次数，y 轴为成功率。

critic 诊断指标：

- AUC：value 是否区分成功/失败 chunk prefix。
- ECE：success probability calibration error。
- Spearman correlation：预测 value 与 counterfactual success rate 的相关性。
- Replan timing error：与 oracle replan step 的平均绝对误差。

### 3.6 Ablation

核心消融：

1. 无 uncertainty，只用 value drop。
2. 无 value drop，只用 uncertainty。
3. 只用 chunk 起点 value，不用 step-wise profile。
4. 不使用扰动 rollout 训练 critic。
5. 不使用 hard negative。
6. critic ensemble 数量：1、3、5。
7. chunk 长度：`H=4,8,16,32`。
8. gate 阈值：不同 `tau_drop` 形成成功率-重规划频率曲线。
9. target 类型：return、success probability、recovery advantage。
10. policy 类型：BC transformer、diffusion policy、VLA policy。

关键结论应证明：

- step-wise value profile 比单点 confidence 更有效。
- value-guided gate 在相同 replan budget 下成功率更高。
- 在相同成功率下，value-guided gate 需要更少 replanning。
- 扰动/recovery 数据显著提升 critic 对早期失败的敏感性。

### 3.7 统计显著性

每个任务至少 50 episodes，论文实验建议 100 episodes。每个方法使用 3 个随机种子，报告 mean 和 95% confidence interval。

推荐检验：

- Success rate：paired bootstrap across tasks，或按 episode 做 Wilson confidence interval。
- Replan count / execution time：paired t-test 或 Wilcoxon signed-rank test。
- 多 baseline 比较：Holm-Bonferroni 校正。
- 跨任务汇总：每个任务先算相对提升，再对任务维度做 bootstrap。

报告格式：

```text
Value-Guided RTC vs Fixed stride k=4:
Success +8.3 pp, replan -22.5%, p < 0.01, 95% CI [+4.1, +12.6]
```

### 3.8 预期曲线与图表

论文中建议包含以下图：

1. Success vs Replan Count Pareto 曲线：Value-Guided RTC 应位于固定 stride 曲线的左上方。
2. Value profile 可视化：成功 chunk 平滑或保持高值，失败 chunk 在接触前后出现断崖。
3. Oracle gap：Value-Guided RTC 接近 oracle RTC，明显优于 confidence RTC。
4. Calibration plot：预测 success probability 与实际成功率基本单调一致。
5. Threshold sweep：随着 `tau_drop` 增大，replan count 上升，成功率先升后趋于饱和。
6. Failure truncation histogram：失败 episode 中，Value-Guided RTC 更早触发 replan。

预期定量趋势：

- 简单 pick-place：成功率提升较小，主要减少不必要 replan。
- 接触任务：成功率提升最大，尤其是插孔、开抽屉、旋转水龙头。
- 长程任务：replan 次数不一定最少，但 replan 更集中在阶段转换和低价值时刻。

## 4. 真机实验

### 4.1 平台选择

最小可行真机平台：

- 单臂 6/7 DoF 机械臂，夹爪末端。
- 低成本平台可用 ALOHA 单臂、Dobot、UFACTORY xArm、Franka、UR5、WidowX 或 SO-100/SO-101 类低成本臂。
- 传感器：1 到 2 个 RGB 相机，推荐一个腕部相机加一个第三视角相机。
- 控制频率：policy chunk 以 5 到 10 Hz 输出低层目标，chunk 长度 4 到 16。

推荐硬件优先级：

1. 已稳定运行的现有平台优先，不为了方法验证更换硬件。
2. 有腕部相机优先，因为接触失败和抓取偏差更容易被 critic 观察到。
3. 夹爪开合状态、末端位姿、力矩或电流读数应记录，即使主方法只用视觉。

### 4.2 真机任务选择

最小可行任务选择 3 个：

1. Pick and place：从桌面抓取积木放入碗或指定区域。
2. Drawer open/close：拉开或推上小抽屉。
3. Insertion or placement with constraint：将圆柱插入杯口、将物体放入窄槽、将 USB-like 插片插入宽容槽。

理想论文任务选择 6 到 8 个：

- 桌面 pick-place，目标位置随机。
- 物体分类放置，语言指定目标。
- 抽屉打开/关闭。
- 旋钮或水龙头旋转。
- 插孔或窄槽放置。
- 叠块或堆叠。
- 遮挡后抓取。
- 多阶段任务：打开抽屉后放入物体再关闭。

任务应覆盖三类失败：

- 视觉定位失败：抓空、抓偏。
- 接触执行失败：卡住、推偏、插不进去。
- 时序失败：过早闭合夹爪、打开太早、阶段转换迟滞。

### 4.3 数据量预算

最小可行真机数据：

- 每个任务 50 条人工遥操作 demo。
- 每个任务 100 到 150 条 policy rollout。
- 每个任务 50 条带人工 correction 的失败恢复轨迹。
- 总计约 600 到 750 条真机轨迹。

理想论文真机数据：

- 每个任务 100 到 200 条 demo。
- 每个任务 300 到 500 条 policy rollout。
- 每个任务 150 到 300 条 correction/recovery 轨迹。
- 6 到 8 个任务，总计约 3k 到 8k 条轨迹。

数据记录字段：

- RGB/RGB-D、相机时间戳。
- 末端位姿、关节角、夹爪宽度、控制命令。
- action chunk、实际执行 action、replan step。
- 人工 correction 标记、reset 标记。
- episode success/failure、失败类型。
- 安全停止或越界原因。

### 4.4 人工 correction 与 reset 方案

真机不应只收成功 demo。需要主动收集“将失败截断并恢复”的数据。

**correction 方案**

- 机器人自动执行 chunk。
- 操作者按键触发 correction，例如键盘、脚踏开关或手柄按钮。
- 触发后记录当前 step 为 `human_replan_needed=1`。
- 操作者短程遥操作 2 到 5 秒，将系统带回可恢复状态。
- 策略继续执行或 episode 结束。

**reset 方案**

- 桌面贴定位标记，reset 位置固定在几个离散区域。
- 每 10 到 20 次 episode 做一次相机外参和工作区检查。
- 失败后不立即丢弃轨迹，先标注失败原因和失败发生 step。
- 对插孔/抽屉类任务，设计可快速复位的 3D 打印 fixture。

**correction 标签用途**

- 训练 replan classifier：人类触发点作为正样本。
- 训练 recovery advantage：触发 correction 前后的成功概率差。
- 评估 Value-Guided RTC 是否能提前预测人类 correction 点。

### 4.5 安全边界

必须设置独立于学习策略的安全层：

- 工作区边界：末端 `x/y/z` 限制，超出立即停止。
- 速度限制：末端线速度和角速度上限。
- 力/电流阈值：检测卡住、撞击、夹爪过载。
- 自碰和桌面碰撞保护：最低高度、禁入区域。
- gripper 保护：夹爪闭合力或电流限制。
- emergency stop：操作者可随时硬停。

Value-Guided RTC 只决定 replan 时机，不允许绕过安全层。所有方法共享同一安全边界，避免对照不公平。

### 4.6 真机评测 protocol

每个任务固定 3 到 5 个随机化维度：

- 初始物体位置。
- 目标位置。
- 物体姿态。
- 轻微光照变化。
- 干扰物是否出现。

评测时每个方法每个任务至少 20 episodes，理想论文为 50 episodes。方法顺序随机化，避免硬件热漂移、操作者疲劳或环境变化造成偏差。

成功判定：

- Pick-place：物体稳定位于目标区域 2 秒。
- Drawer：抽屉位移超过阈值，例如打开超过 8 cm。
- Insertion：物体进入槽内并稳定停留。
- 多阶段：所有子目标按顺序完成。

记录指标：

- Success rate。
- 平均完成时间。
- 平均 replan 次数。
- 人工 intervention 次数。
- 安全停止次数。
- 失败类型分布。
- 每次 replan 前后的 value profile。

### 4.7 真机对照组

至少包含：

1. Full chunk。
2. Fixed stride RTC，建议 `k=1,4,8`。
3. Policy confidence RTC。
4. Value-Guided RTC。

如果真机时间允许，增加：

- Human-triggered RTC upper bound：操作者只按 replan，不直接遥操作。
- Sim-only critic：只用仿真训练 critic，评估 sim-to-real 迁移。
- Fine-tuned critic：少量真机 correction 数据微调 critic。

关键比较：

- 相同成功率下，Value-Guided RTC 是否减少 replan 和完成时间。
- 相同 replan 次数下，Value-Guided RTC 是否减少失败和人工 intervention。
- 低成本平台执行噪声更大时，value-guided gate 是否比固定 stride 更鲁棒。

### 4.8 失败模式记录

每个失败 episode 必须人工标注一个主失败类型，必要时加次失败类型：

- `perception_miss`：物体定位错误、遮挡、目标识别错误。
- `grasp_empty`：夹爪闭合但未抓住。
- `grasp_slip`：抓住后滑落。
- `contact_jam`：插入、拉抽屉或旋转时卡住。
- `push_away`：末端将物体推离可操作区域。
- `timing_error`：夹爪开合或阶段切换时机错误。
- `unsafe_stop`：越界、速度过高、力阈值触发。
- `planner_loop`：频繁 replan 但无法推进任务。

额外记录：

- 失败前最后 10 个 action。
- 失败前 value profile 和 uncertainty。
- gate 是否触发过 replan。
- 若未触发，属于 false negative；若频繁触发但无收益，属于 false positive。

## 5. 最小可行实验

目标是在 2 到 4 周内验证方向是否值得扩展。

### 5.1 仿真 MVP

环境：

- LIBERO 选择 8 个任务。
- 基础策略使用已有 BC transformer、diffusion policy 或轻量 VLA policy。
- chunk 长度 `H=8` 或 `H=16`。

数据：

- 每任务 50 条专家 demo。
- 每任务 300 条 policy rollout。
- 每任务 200 条 noisy rollout。

critic：

- 先训练 state critic 或冻结视觉 encoder 的 visual critic。
- target 使用最终 success label。
- 训练 3 个 ensemble critic，输出 success probability 和 uncertainty。

方法：

- Full chunk。
- Fixed stride `k=1,4,8`。
- Confidence RTC。
- Value drop gate。
- Value drop + uncertainty gate。

评测：

- 每任务每方法 50 episodes，3 seeds。
- 报告 success、replan count、execution steps、failure truncation。

MVP 成功标准：

- 在至少 5/8 个任务上，Value-Guided RTC 的 success-replan Pareto 优于 fixed stride。
- 接触类任务成功率提升至少 5 到 10 个百分点。
- critic AUC 大于 0.70，value drop 与失败 step 有可解释对应关系。

### 5.2 真机 MVP

平台：

- 单臂夹爪，1 个第三视角相机即可；有腕部相机更好。

任务：

- Pick-place。
- Drawer open。
- Cup/slot insertion。

数据：

- 每任务 50 demo。
- 每任务 100 policy rollout。
- 每任务 50 correction/recovery。

训练：

- 用仿真或历史数据预训练 policy。
- 真机数据微调 policy。
- critic 可先用 frozen encoder + MLP，输入视觉 embedding、proprio、chunk step、action prefix。

评测：

- 每任务每方法 20 episodes。
- 对照 Full chunk、Fixed stride `k=4`、Value-Guided RTC。

MVP 成功标准：

- 至少 2/3 个任务上，Value-Guided RTC 比 fixed stride 达到相同或更高成功率，同时 replan 次数不增加超过 20%。
- 在 drawer/insertion 中，能明显减少“卡住后继续执行”的动作步数。

## 6. 理想论文实验

理想版本应证明方法具有跨任务、跨策略、跨平台的普适性。

### 6.1 仿真论文实验

环境：

- LIBERO：至少 20 个任务，覆盖 Spatial/Object/Goal/Long。
- RoboTwin：至少 10 个任务，包含长时程和双臂任务。
- ManiSkill：至少 10 个任务，覆盖抓取、插入、旋转、开关节物体。

策略：

- Diffusion Policy chunk。
- Transformer BC chunk。
- VLA policy chunk。

critic：

- visual-language-action critic，复用 policy encoder。
- ensemble 规模 5。
- target 包含 success probability 和 recovery advantage。
- 部分任务做 counterfactual rollout，构造 oracle replan 标签。

评测：

- 每任务每方法 100 episodes，3 seeds。
- 报告 per-task、per-suite 和 aggregate 结果。
- 所有阈值只在 validation tasks 上调，不在 test tasks 上调。

论文主表：

- Success rate。
- Replan count。
- Execution time。
- Efficient Success。
- Failure truncation。

论文主图：

- Pareto frontier。
- value profile case study。
- oracle gap。
- ablation heatmap。
- sim-to-real transfer curve。

### 6.2 真机论文实验

平台：

- 单臂夹爪平台作为主实验。
- 若资源允许，增加一个低成本平台或另一种机械臂，证明平台无关性。

任务：

- 6 到 8 个真实桌面任务。
- 至少 2 个接触丰富任务。
- 至少 2 个长时程多阶段任务。
- 至少 1 个语言条件任务。

数据：

- 总计 3k 到 8k 条真机轨迹。
- 每任务都有 demo、policy rollout、correction/recovery。

评测：

- 每任务每方法 50 episodes。
- 方法顺序随机化。
- 每天开始和结束都跑固定 sanity task，检查硬件状态漂移。

对照组：

- Full chunk。
- Fixed stride RTC。
- Confidence RTC。
- Human-triggered RTC upper bound。
- Value-Guided RTC。

预期论文级结果：

- 平均成功率相对 fixed stride 提升 5 到 15 个百分点。
- 平均 replan 次数相对 high-frequency RTC 降低 30% 到 60%。
- 接触任务中 unsafe stop 或 contact jam 明显减少。
- 人工 correction 点与 value drop 的时间误差小于固定 stride。

## 7. 主要风险与缓解

**风险 1：critic 只学到任务难度，不学 chunk 内变化。**

缓解：同一观测下采样多种 chunk，加入 noisy chunk 和 hard negative，使用 pairwise ranking loss。

**风险 2：value drop 触发太频繁，退化成 high-frequency control。**

缓解：引入 replan penalty，在 validation 上选择 Pareto 最优阈值，设置最小执行步数 `min_steps=2`。

**风险 3：critic 在真机上误判，导致危险动作继续执行。**

缓解：安全层独立运行；uncertainty 高时保守 replan；真机初期只允许 gate 提前 replan，不允许延长超过 baseline chunk。

**风险 4：仿真收益无法迁移真机。**

缓解：用少量真机 correction 微调 critic；采用视觉 augmentation；评估 sim-only、real-finetuned 两档。

**风险 5：固定 stride baseline 已经很强。**

缓解：报告 Pareto 曲线而非单点；加入长程和接触任务；强调相同 replan budget 下的成功率。

## 8. 最快出结果方案：1 张 GPU + 一台真机

最快路径应先拿到“value profile 能预测失败并改善 RTC”的证据，而不是追求大模型完整闭环。

### 第 1 周：仿真快速闭环

- 选 LIBERO 6 到 8 个任务，chunk 长度 `H=8`。
- 用已有 demo 训练或复用一个 BC/diffusion chunk policy。
- rollout 每任务 300 条，其中 1/3 加动作噪声或目标扰动。
- 训练轻量 critic：冻结视觉 encoder，MLP 输入 visual embedding、proprio、chunk step、action prefix。
- 跑 Full chunk、Fixed stride `k=4`、Value-Guided RTC。
- 产出第一版 Pareto 曲线和 value profile case study。

### 第 2 周：真机小数据验证

- 只做 2 个任务：pick-place 和 drawer open。
- 每任务收 50 demo、80 policy rollout、30 correction。
- 真机 critic 从仿真 critic 初始化，或只用真机数据训练小 MLP head。
- 评测每任务每方法 20 episodes。
- 重点看 drawer：卡住或拉偏前 value 是否下降，Value-Guided RTC 是否提前 replan。

### 第 3 到 4 周：补强论文信号

- 增加 insertion/slot placing 作为第三个真机任务。
- 做 ablation：value only、uncertainty only、value+uncertainty。
- 做阈值 sweep，画 success-replan Pareto。
- 记录失败模式表，突出 fixed stride 的 late replan 和 full chunk 的 error accumulation。

### 资源受限时的取舍

- 不训练完整 VLA critic，先冻结 encoder 训练小 critic。
- 不做 RoboTwin，先用 LIBERO + 真机。
- 不做 counterfactual rollout，先用 final success label + noisy rollout。
- 不追求 8 个真机任务，先把 drawer/insertion 这种接触任务做扎实。
- 真机评测少但要严格随机化和完整失败标注。

最快可发表雏形：

```text
LIBERO 8 tasks + real robot 3 tasks
Value-Guided RTC improves success-replan Pareto over fixed stride
step-wise value profile predicts contact failure and human correction timing
```
