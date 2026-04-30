# Value-Guided RTC 当前 Evo-RL 版本分析与实验计划

日期：2026-04-30

## 结论

读完 `value_guided_rtc_extensions.md` 后，当前最合理的推进方式不是马上实现完整的 chunk value profile critic，也不是直接在真机上做动态 RTC。

当前版本应该收敛成三步：

1. **先做证据**：证明固定 chunk / 固定 RTC 在不同任务阶段确实有不同最优执行节奏。
2. **先做 shadow-mode switcher**：只记录 Value-Guided RTC 如果在线运行会怎样切换，不先控制机器人。
3. **最后再接 RTC 底层参数**：把 value/contact/progress 信号接到 `execution_horizon`、`guidance_weight`、accept/reject 和 action-dim mask。

论文叙事也应收敛：

> RTC 解决 action chunk 的时间连续性；VG-RTC 解决什么时候应该继续相信旧 chunk 的时间承诺，什么时候应该付出切换代价，让新 observation/new chunk 接管。

不要写成：

> value 低了就重新切 chunk。

这个太像普通 adaptive replanning。

## 当前文档状态

| 文件 | 当前作用 | 判断 |
|---|---|---|
| `value_guided_rtc_extensions.md` | 创新点扩展，讨论 uncertainty、switch cost、latency、contact 等 | 思路完整，但离实验入口偏远 |
| `value_guided_rtc_algorithm.md` | 完整算法长稿，定义 value profile、option-value switcher、训练/运行时框架 | 可作为论文方法草稿，但第一版实现不能全做 |
| `value_guided_rtc_experiments.md` | 仿真实验设计，覆盖 LIBERO/ManiSkill/RoboTwin、critic 训练、baseline、metric | 适合作为中长期实验矩阵 |
| `value_guided_rtc_evo_rl_experiment_readme.md` | 当前仓库落地说明，最接近可执行路线 | 应作为近期主 README |
| `idea_progress_producing_chunk_trust.md` | 和 VG-RTC 最相关的早期 idea：chunk 是否产生 progress | 可并入 VG-RTC 的 critic/switcher 信号 |
| `idea_event_conditioned_rtc.md` | event/contact 条件化 RTC | 可作为 VG-RTC 的低成本前置版本 |
| `teleop_audit/README.md` | Piper 真机底层日志审计命令 | 支撑真机 physical evidence，不是 VG-RTC 主实现 |

当前文档的问题是：`extensions` 和 `algorithm` 比较像论文构想，`evo_rl_experiment_readme` 才是应该执行的工程路线。后续不要继续扩大 idea 面，而应开始把实验闭环写清楚。

## 当前代码状态

### RTC 已有能力

关键文件：

```text
src/lerobot/policies/rtc/configuration_rtc.py
src/lerobot/policies/rtc/modeling_rtc.py
src/lerobot/policies/rtc/action_queue.py
examples/rtc/eval_dataset.py
examples/rtc/eval_with_real_robot.py
```

当前 `RTCConfig` 只有少数核心参数：

```text
enabled
prefix_attention_schedule
max_guidance_weight
execution_horizon
debug
debug_maxlen
```

`RTCProcessor.denoise_step()` 的核心是：

```python
weights = get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
err = (prev_chunk_left_over - x1_t) * weights
result = v_t - guidance_weight * correction
```

这说明 VG-RTC 最干净的插入点有四个：

1. 动态 `execution_horizon`
2. 动态 `max_guidance_weight`
3. 动态 prefix weights
4. 后续新增 action-dim weights，例如夹爪维度和手臂维度分开调

### Pi0.5 / SmolVLA 的 RTC 接口

Pi0/Pi0.5/SmolVLA 都通过 `predict_action_chunk(..., execution_horizon=...)` 把 per-call RTC 参数传到 `RTCProcessor`。

重要限制：

```text
policy.select_action() 不支持 RTC
```

Pi0.5 里明确有断言：

```python
assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"
```

所以：

```text
lerobot-eval 可用于 fixed h sweep；
但不能直接用于 RTC success-rate sweep。
```

RTC 实验必须写自己的 rollout loop，使用：

```text
policy.predict_action_chunk(...)
ActionQueue.get_left_over()
ActionQueue.merge(...)
```

### ActionQueue 已有能力

`ActionQueue` 已经能做：

```text
RTC enabled: 用新 chunk 替换 queue，并跳过 inference delay 对应的动作
RTC disabled: 追加新 chunk
get_left_over(): 返回旧 chunk 剩余动作给 RTC guidance
```

当前缺少的是：

```text
accept/reject 新 chunk
动态 request timing
动态 chunk_size_threshold
记录 boundary jump / jerk / replan_count
value/contact/progress 的 shadow-mode decision log
```

第一版 VG-RTC 不应该先改 `ActionQueue` 核心逻辑，而是在 `examples/vg_rtc` 的 eval loop 外面包一层 scheduler。

### 当前 value pipeline

已有：

```text
src/lerobot/scripts/lerobot_value_train.py
src/lerobot/scripts/lerobot_value_infer.py
src/lerobot/configs/value_train.py
src/lerobot/values/pistar06/modeling_pistar06.py
```

但现在只支持：

```text
--value.type=pistar06
```

它是 frame/state value，输入视觉和语言，输出 normalized return-to-go。它不是 chunk critic，因为它不显式输入候选 action chunk。

因此当前 value pipeline 的合理用法是：

```text
短期：作为 value annotation / progress proxy / offline oracle 分析
中期：为 chunk critic 提供监督标签
长期：新增 Q_chunk(o_t, A_t, k)
```

不要在论文里把现有 `pistar06` 说成 VG-RTC 的核心 critic。

### Piper audit 代码

当前新增脚本：

```text
src/lerobot/scripts/lerobot_piper_audit.py
src/lerobot/scripts/lerobot_piper_audit_plot.py
```

它们的作用是：

```text
记录 Piper 遥操作时的 raw_observation / sent_action / piper_sdk / target_minus_obs / obs_velocity
画 gripper effort、电机 current、tracking residual、physical evidence overview
```

这部分适合支撑 real-robot physical evidence，但不是 VG-RTC 主实验入口。近期用途是验证：

```text
grasp evidence
progress/stall evidence
motor load evidence
```

它能帮助真机阶段定义 S2/S3：

```text
S2 = grasp/contact established + progress
S3 = grasp/contact established + no progress + load/error high
```

## 当前最小实验闭环

### Stage 0：固定 h 动机实验

目的：证明 action chunk 的固定执行长度对成功率、速度和失败模式有明显影响。

这一步不需要 RTC，不需要 value，也不需要新代码。

命令：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
export MUJOCO_GL=egl

for H in 2 4 8 10 16; do
  lerobot-eval \
    --policy.path=lerobot/pi05_libero_finetuned \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids='[0,1,2]' \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --policy.n_action_steps=$H \
    --policy.device=cuda \
    --output_dir=outputs/eval/vg_rtc_fixed_h/libero10_h${H} \
    --env.max_parallel_tasks=1
done
```

必须输出：

```text
success_rate vs H
episode_steps vs H
每个 task_id 的 best H
失败 case 的视频或状态截图
```

如果不同 H 没差异，VG-RTC 动机不成立，要换任务。

### Stage 1：RTC fixed sweep

目的：先测 vanilla RTC 本身，证明 `execution_horizon` 和 `max_guidance_weight` 不是一组全局最优固定参数。

需要新增：

```text
examples/vg_rtc/eval_libero_rtc_sweep.py
```

核心 loop：

```python
if action_queue.empty() or action_queue.qsize() <= threshold:
    prev = action_queue.get_left_over()
    actions = policy.predict_action_chunk(
        obs,
        prev_chunk_left_over=prev,
        inference_delay=real_delay,
        execution_horizon=eh,
    )
    action_queue.merge(raw_actions, processed_actions, real_delay)

action = action_queue.get()
env.step(action)
```

第一组 sweep：

```text
execution_horizon: 4, 8, 10, 12, 16
max_guidance_weight: 2, 5, 10, 20
prefix_attention_schedule: LINEAR, EXP
```

必须记录：

```text
replan_count
inference_delay
queue_size
boundary_jump_l2
action_jerk_l2
rtc_debug correction norm
success/failure
```

### Stage 2：AAC-lite baseline

目的：避免 VG-RTC 被说成只是 adaptive chunking。AAC-lite 用 action variance/entropy 决定执行多少步。

需要新增：

```text
examples/vg_rtc/eval_libero_aac_lite.py
```

方法：

```python
chunks = sample N chunks from the policy
step_var = chunks.var(dim=0).mean(dim=-1)
h_star = first large rise / argmax entropy slope
execute h_star steps
```

它是强 baseline，因为它也在做 adaptive horizon，但不用 value。

### Stage 3：VG-RTC-lite shadow mode

这一步是当前最关键的近期版本。

先不要让 scheduler 真的控制 policy，只记录它会怎么决策：

```text
当前 fixed/RTC rollout 中，
如果有 value/contact/progress switcher，
它会在哪些时刻 replan、keep、shorten horizon、strengthen gripper continuity。
```

需要新增：

```text
src/lerobot/policies/rtc/value_scheduler.py
examples/vg_rtc/eval_libero_vg_rtc_shadow.py
```

第一版 scheduler 输入：

```text
state/frame value from pistar06 or oracle simulator progress
current chunk index
prev_leftover length
chunk age
RTC correction norm
gripper/contact evidence if real robot
boundary discontinuity cost
inference latency
```

输出：

```text
decision: keep / replan / reject_new / shorten_horizon / strengthen_gripper
suggested_execution_horizon
suggested_guidance_weight
score_continue
score_replan
switch_cost
uncertainty
```

shadow mode 要回答：

```text
这些 decision 是否集中出现在真实失败前、contact/stall 前、value drop 前？
```

### Stage 4：VG-RTC-lite execution

只有 Stage 3 的 shadow decision 合理，才让它真的控制执行：

```text
value/profile 低或 drop 大 -> 提前 replan
switch cost 高 -> 保持旧 chunk
uncertainty 高 -> 降级到 fixed RTC
```

第一版控制对象只做：

```text
execute_horizon / replan timing
```

不要马上动态改 `guidance_weight`。

### Stage 5：VG-RTC-full

full 版本再控制 RTC 底层：

```text
execution_horizon
max_guidance_weight
accept/reject new chunk
contact-aware action-dim weights
gripper continuity vs arm replanning split
```

这一步才需要改 `RTCProcessor.denoise_step()`，例如增加：

```python
guidance_weight_override: float | Tensor | None
prefix_weights_override: Tensor | None
action_dim_weights: Tensor | None
```

## 真机实验怎么接

真机先不要直接跑动态 VG-RTC。当前 Piper 阶段最应该做的是数据审计和条件验证。

### 真机 Stage A：收对照数据

对 drawer task 至少收三类：

```text
1. 成功：打开抽屉、放入物体、关闭抽屉
2. 失败：打开抽屉但不放物体，或夹住后拉不动
3. 自由空间：做类似动作但不接触抽屉/物体
```

用：

```bash
python -m lerobot.scripts.lerobot_piper_audit ...
python -m lerobot.scripts.lerobot_piper_audit_plot <run_dir>
```

目标不是训练，而是证明这些信号有区分度：

```text
gripper effort + gripper mismatch 是否检测 grasp/contact
motor current residual 是否检测 load/stall
target_minus_obs + obs_velocity 是否检测 tracking failure
wrist video progress 是否区分 S2/S3
```

### 真机 Stage B：定义 S2/S3

先用规则，不用学习：

```text
grasp_score = f(gripper_pos - gripper_cmd, abs(gripper_effort), abs(gripper_vel))
intent_score = f(action_delta, target_minus_obs)
progress_score = visual drawer/object progress 或 task-axis end_pose progress
load_score = motor_current residual
tracking_score = joint target_minus_obs
```

判断：

```text
S2 stable-grasp + progress =
  grasp_score high
  AND progress_score high
  AND slip_score low

S3 grasped-but-stalled / jam =
  grasp_score high
  AND intent_score high
  AND progress_score low
  AND (load_score high OR tracking_score high)
```

这一步只做分析图和 shadow decision，不控制机器人。

### 真机 Stage C：接到 RTC

只有当 S2/S3 能在成功/失败/自由空间对照里分开，才接 RTC：

```text
S2: gripper 维度 continuity 强，arm 维度正常或略强
S3: gripper 维度 continuity 强，arm 维度 continuity 弱，允许 recovery/replan
free/search: 正常 RTC 或弱 RTC，允许重新对准
```

这比“夹住就调 RTC”强，因为它体现的是：

```text
物理接触承诺改变了 action chunk 的哪些维度该保持、哪些维度该重规划。
```

## 当前应提交的版本边界

本版本只应提交：

```text
1. Value-Guided RTC 文档与当前实验计划
2. Piper audit 脚本和绘图脚本
3. teleop_audit README
4. 忽略生成数据的 .gitignore
```

不要提交：

```text
teleop_audit/runs/
teleop_audit/datasets/
__pycache__/
根目录 README 删除/重命名
URDF 或 gravity compensation 的无关修改
```

## 下一步具体 TODO

优先级从高到低：

1. 跑 `fixed h sweep`，确认不同 `H` 是否真的影响 LIBERO 成功率。
2. 新增 `examples/vg_rtc/eval_libero_rtc_sweep.py`，让 RTC 能在 LIBERO 中评估。
3. 在 RTC eval loop 里保存 per-step CSV，而不是只保存 episode 平均值。
4. 做 `AAC-lite`，拿 entropy adaptive horizon baseline。
5. 做 `VG-RTC shadow scheduler`，只记录不控制。
6. 如果 shadow decision 和失败/接触窗口对齐，再做 execution 版本。
7. 真机继续用 audit 收 S2/S3 对照数据，不急着在线控制。

## 一句话版本

当前最稳的版本是：

> 先证明 action chunk 的最优执行承诺不是固定的，再用 value/contact/progress 证据做 shadow-mode option switcher，最后才把 switcher 接到 RTC 的 horizon、guidance 和 accept/reject 机制。这样 VG-RTC 不是“AAC 的 Q-value 版”，也不是“夹爪 if-else 调参”，而是把 RTC 的固定时间承诺改成 value-aware、latency-aware、contact-aware 的在线 option selection。
