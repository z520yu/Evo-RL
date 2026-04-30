# VG-RTC 在 Evo-RL 中启动实验的 README

本文档目标：把 **Value-Guided RTC** 从研究想法落到当前 `/home/lenovo/Evo-RL` 代码库里，明确第一批实验怎么开始、用哪些现有能力、哪些地方需要新写代码，以及怎么把 AAC 作为 baseline 引入。

核心判断：

```text
不要先完整复现 AAC 再做改动。
主线应基于 Evo-RL 现有 Pi0.5 + RTC + value pipeline 做 VG-RTC；
AAC 只作为 entropy-based adaptive chunk baseline。
```

原因是 AAC 的贡献点是 `adaptive execution horizon`，即“每次预测完整 action chunk，但动态选择执行前多少步”。你的贡献应该避开“把 entropy 换成 Q”的弱叙事，主打：

```text
把 RTC 的 replan timing、overlap length、guidance strength 和 accept/reject 机制变成 value-aware。
```

---

## 1. 当前 Evo-RL 里已经有什么

### 1.1 可直接复用的 VLA/RTC 能力

当前仓库基于 LeRobot 0.4.4，Pi0/Pi0.5/SmolVLA 已经有 RTC 支持。

关键文件：

```text
src/lerobot/policies/rtc/configuration_rtc.py
src/lerobot/policies/rtc/modeling_rtc.py
src/lerobot/policies/rtc/action_queue.py
examples/rtc/eval_dataset.py
examples/rtc/eval_with_real_robot.py
docs/source/rtc.mdx
```

当前 `RTCConfig` 主要控制：

```text
enabled
execution_horizon
max_guidance_weight
prefix_attention_schedule
debug
```

`RTCProcessor.denoise_step()` 的核心逻辑是：

```text
1. 用 prev_chunk_left_over 表示旧 chunk 还没执行完的部分
2. 对新 chunk 的 denoising 加 prefix guidance
3. 用 execution_horizon 决定 overlap 区域长度
4. 用 max_guidance_weight 控制新 chunk 多强地贴旧 chunk
```

这正好是 VG-RTC 可以插入的地方。

### 1.2 可直接复用的 value pipeline

Evo-RL 已经有 value 训练和推理命令：

```text
lerobot-value-train
lerobot-value-infer
```

关键文件：

```text
src/lerobot/scripts/lerobot_value_train.py
src/lerobot/scripts/lerobot_value_infer.py
src/lerobot/configs/value_train.py
src/lerobot/values/pistar06/modeling_pistar06.py
```

当前 value 训练只支持：

```text
--value.type=pistar06
```

它训练的是 **state/frame value**，输入是视觉和语言，输出当前 frame 的 normalized return-to-go。它还不是 chunk critic，因为它没有显式输入候选 action chunk。

所以第一版实验可以复用它做：

```text
1. trajectory value annotation
2. failure/progress 分析
3. oracle value-drop study
4. 后续 chunk critic 的监督标签来源
```

但真正的 VG-RTC 需要新增一个：

```text
Q_chunk(o_t, A_t, k)
```

或者更轻量的：

```text
ValueDropHead(o_t, A_t) -> 每个未来 step 的 continue/replan 分数
```

### 1.3 当前不能直接用 `lerobot-eval` 跑 RTC success rate

这是一个重要坑。

`lerobot-eval` 的 rollout 调用的是：

```python
policy.select_action(observation)
```

但 Pi0.5 里有：

```python
assert not self._rtc_enabled(), (
    "RTC is not supported for select_action, use it with predict_action_chunk"
)
```

也就是说：

```text
普通 chunk-size sweep 可以先用 lerobot-eval。
RTC success-rate sweep 不能直接用 lerobot-eval。
```

RTC 实验必须走：

```text
policy.predict_action_chunk(...)
ActionQueue(...)
prev_chunk_left_over
inference_delay
action_queue.merge(...)
```

可参考：

```text
examples/rtc/eval_with_real_robot.py
```

第一批仿真实验需要新增一个 LIBERO 版本的 RTC eval loop。

---

## 2. 实验主线怎么设计

推荐四阶段推进。

```text
Stage 0: 环境和 baseline sanity check
Stage 1: fixed chunk / fixed RTC sweep
Stage 2: AAC-lite entropy baseline
Stage 3: VG-RTC-lite value-based switching
Stage 4: VG-RTC-full 控制 RTC 底层参数
```

### Stage 0: 先确认现有 Pi0.5 LIBERO 能跑

这一步不做创新，只确认环境、模型、LIBERO、GPU 都正常。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export MUJOCO_GL=egl

lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --output_dir=outputs/eval/vg_rtc_sanity_pi05_h10 \
  --env.max_parallel_tasks=1
```

如果这一步都不稳定，先不要碰 RTC 或 value。

### Stage 1A: 用现有 `lerobot-eval` 做普通 fixed h sweep

目的：证明固定 execution horizon/chunk size 对成功率有强影响。

先做非 RTC 版本：

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl
export MUJOCO_GL=egl

for H in 1 2 4 8 10 12 16; do
  lerobot-eval \
    --policy.path=lerobot/pi05_libero_finetuned \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids='[0,1,2]' \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --policy.n_action_steps=$H \
    --policy.device=cuda \
    --output_dir=outputs/eval/vg_rtc_fixed_h/h${H} \
    --env.max_parallel_tasks=1
done
```

这一步回答：

```text
h 太小是否 mode-jumping / 抖动？
h 太大是否反应慢 / 长程任务失败？
不同 task_id 的最优 h 是否不同？
```

建议先跑：

```text
libero_spatial: task_ids [0,1,2]
libero_object: task_ids [0,1,2]
libero_10: task_ids [0,1,2]
```

不用一开始跑全 benchmark。

### Stage 1B: 新增 RTC fixed-parameter sweep

这一步需要新增脚本，建议路径：

```text
examples/vg_rtc/eval_libero_rtc_sweep.py
```

它应该把 `lerobot_eval.py` 的环境 rollout 和 `examples/rtc/eval_with_real_robot.py` 的 action queue 逻辑合起来。

核心循环不是：

```python
action = policy.select_action(obs)
```

而是：

```python
if action_queue.qsize() <= threshold:
    prev_actions = action_queue.get_left_over()
    actions = policy.predict_action_chunk(
        obs,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev_actions,
    )
    action_queue.merge(original_actions, postprocessed_actions, real_delay)

action = action_queue.get()
env.step(action)
```

第一版参数 sweep：

```text
execution_horizon: 4, 8, 10, 12, 16
max_guidance_weight: 2.0, 5.0, 10.0, 20.0
prefix_attention_schedule: LINEAR, EXP
```

先不加 value。

目标是证明：

```text
RTC 自身也不是固定参数最优；
overlap 太长会过度服从旧 chunk，反应变慢；
guidance 太强会平滑但可能坚持错误策略；
guidance 太弱会更灵活但边界不连续。
```

### Stage 2: AAC-lite baseline

不要一开始完整复现 AAC，只做核心 baseline。

AAC-lite 逻辑：

```text
1. 当前观测 o_t
2. 并行采样 N 个完整 action chunks，长度为 H
3. 计算每个 future step 的 action variance / entropy
4. 找 entropy curve 的突增点 h*
5. 执行前 h* 步
6. 重新观测
```

建议新增：

```text
examples/vg_rtc/eval_libero_aac_lite.py
```

对 Pi0.5 来说，可以先用连续动作方差近似 entropy：

```python
chunks = policy.predict_action_chunk(repeated_obs)  # [N, H, action_dim]
step_var = chunks.var(dim=0).mean(dim=-1)           # [H]
```

然后：

```python
prefix_entropy[h] = step_var[:h].mean()
h_star = argmax(prefix_entropy[h+1] - prefix_entropy[h])
h_star = max(h_star, min_h)
h_star = min(h_star, max_h)
```

这不是 full AAC，但足够作为实验对照：

```text
entropy-based adaptive h
vs
value-based RTC switching
```

AAC-lite 的输出必须记录：

```text
h_star
entropy_curve
replan_count
success
steps_to_success
jerk / boundary discontinuity
```

### Stage 3: VG-RTC-lite

这一步先不要动 RTC denoising 的内部 guidance，只做 value-based execution scheduler。

问题形式：

```text
当前有旧 chunk 剩余动作 R_t。
是否继续执行，还是重新观测生成新 chunk？
```

最小决策：

```text
switch if Q_replan(o_t) - Q_continue(o_t, R_t) > margin + switch_cost
```

第一版可以简化为：

```text
VLA 生成完整 chunk A_t
critic 预测 chunk 内 value profile
选择 value drop 前的 h*
执行前 h* 步
```

建议新增：

```text
src/lerobot/policies/rtc/value_scheduler.py
examples/vg_rtc/eval_libero_vg_rtc_lite.py
```

`value_scheduler.py` 里先放一个独立小模块：

```python
class ValueGuidedRTCScheduler:
    def decide(self, obs, current_chunk, prev_leftover, candidate_chunk):
        ...
        return {
            "should_replan": bool,
            "execute_horizon": int,
            "execution_horizon": int,
            "guidance_weight": float,
        }
```

第一版 critic 不要做太复杂。推荐两条路线：

#### 路线 A: oracle/offline value-drop 分析

先用仿真 rollout 真实后验算：

```text
如果我们知道未来哪一步会失败/掉 value，最佳切换点在哪里？
```

这个能证明 upper bound：

```text
value-aware switching 是有收益空间的。
```

#### 路线 B: 训练轻量 chunk critic

从固定 h / AAC / RTC rollouts 收数据：

```text
obs_t
candidate action chunk A_t
episode success
future frame value
failure phase
```

训练：

```text
Q_chunk(o_t, A_t) -> [q_1, q_2, ..., q_H]
```

标签可以先用：

```text
q_k = normalized return from executing prefix length k
```

如果没有多个 counterfactual rollout，先用 logged trajectory 的后验 value 近似：

```text
q_k ~= V(o_{t+k})
```

注意：当前 `pistar06` value 是 state value，不输入 action chunk。它适合作为标签来源或辅助 baseline，但不能直接替代 chunk critic。

### Stage 4: VG-RTC-full

这才是论文主贡献。

不是只决定：

```text
执行前多少步 h*
```

而是控制 RTC 底层：

```text
1. 什么时候提前 replan
2. 新 chunk 与旧 chunk overlap 多长
3. 新 chunk 多强地贴旧 chunk
4. 什么时候拒绝新 chunk，继续旧 chunk
```

建议算法：

```text
Q_continue = 继续旧 chunk 剩余动作的价值

Q_replan(L, lambda) =
    生成新 chunk 后的价值
    - switch_cost
    - latency_cost
    - discontinuity_cost

选择:
    (L*, lambda*) = argmax Q_replan(L, lambda)

若:
    Q_replan(L*, lambda*) > Q_continue + margin
则:
    接受新 chunk，并使用 L*, lambda*
否则:
    继续当前 chunk
```

和当前代码的对应关系：

```text
L        -> RTCConfig.execution_horizon / predict_action_chunk(execution_horizon=...)
lambda   -> RTCConfig.max_guidance_weight
replan   -> action queue 何时触发新 predict_action_chunk
accept   -> action_queue.merge(...) 还是丢弃新 chunk
```

当前 Pi0.5 已经支持 per-call `execution_horizon`：

```python
policy.predict_action_chunk(..., execution_horizon=dynamic_L)
```

但 `max_guidance_weight` 目前来自：

```text
self.rtc_config.max_guidance_weight
```

所以如果要做 per-call dynamic guidance weight，有两种方式：

```text
简单版：调用前临时改 policy.config.rtc_config.max_guidance_weight
干净版：给 RTCProcessor.denoise_step 增加 guidance_weight_override 参数
```

推荐论文实验用干净版。

---

## 3. 第一批实验矩阵

不要一开始铺太大。第一批只跑 3 个 suite，每个 3 个 task，每个 20 episodes。

```text
libero_spatial: task_ids [0,1,2]
libero_object:  task_ids [0,1,2]
libero_10:      task_ids [0,1,2]
```

方法对比：

```text
1. Fixed h=2
2. Fixed h=4
3. Fixed h=8
4. Fixed h=10
5. Fixed h=16
6. AAC-lite
7. RTC fixed: execution_horizon=10, guidance=10
8. VG-RTC-lite
9. VG-RTC-full
```

指标：

```text
success_rate
mean_steps_to_success
mean_episode_length
replan_count
mean_execute_horizon
chunk_boundary_jump
action_jerk
inference_latency_ms
failure_phase
collision_or_bad_contact_count
```

必须保存逐 episode 结果，不要只看平均值。

建议 CSV schema：

```text
method
env_task
task_id
seed
episode_id
success
episode_steps
total_reward
mean_h
h_histogram
replan_count
mean_inference_latency_ms
mean_boundary_jump_l2
mean_jerk_l2
failure_phase
notes
```

---

## 4. 怎么在当前 Evo-RL 里开始

### 4.1 第一天应该做什么

第一天只做两件事：

```text
1. 确认普通 LIBERO eval 能跑。
2. 跑 fixed h sweep，拿到 success vs h 曲线。
```

用现有命令即可，不要写 VG-RTC。

```bash
cd /home/lenovo/Evo-RL
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

预期产出：

```text
outputs/eval/vg_rtc_fixed_h/...
一张 success_rate vs h 的表
一张不同 task_id 最优 h 的表
```

如果这一步没有发现明显差异，先换任务，不要急着做 VG-RTC。

### 4.2 第二步新增 RTC 仿真 eval loop

新增：

```text
examples/vg_rtc/eval_libero_rtc_sweep.py
```

它要复用：

```text
src/lerobot/scripts/lerobot_eval.py 的 env/preprocessor/postprocessor
examples/rtc/eval_with_real_robot.py 的 ActionQueue 和 predict_action_chunk 逻辑
```

第一版命令设计成：

```bash
python examples/vg_rtc/eval_libero_rtc_sweep.py \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2]' \
  --eval.n_episodes=20 \
  --device=cuda \
  --rtc.execution_horizon=10 \
  --rtc.max_guidance_weight=10.0 \
  --rtc.prefix_attention_schedule=EXP \
  --output_dir=outputs/eval/vg_rtc_rtc_fixed/libero10_eh10_g10
```

这一步的目的不是创新，而是搭建能测 RTC success rate 的实验管线。

### 4.3 第三步做 AAC-lite

新增：

```text
examples/vg_rtc/eval_libero_aac_lite.py
```

命令设计：

```bash
python examples/vg_rtc/eval_libero_aac_lite.py \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2]' \
  --eval.n_episodes=20 \
  --device=cuda \
  --aac.num_samples=8 \
  --aac.min_h=2 \
  --aac.max_h=16 \
  --output_dir=outputs/eval/vg_rtc_aac_lite/libero10_n8
```

先用 `N=8`，不够再上 `N=16/20`。

### 4.4 第四步做 VG-RTC-lite

新增：

```text
src/lerobot/policies/rtc/value_scheduler.py
examples/vg_rtc/eval_libero_vg_rtc_lite.py
```

第一版先支持两种 scheduler：

```text
oracle_value_drop
learned_chunk_critic
```

命令设计：

```bash
python examples/vg_rtc/eval_libero_vg_rtc_lite.py \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2]' \
  --eval.n_episodes=20 \
  --device=cuda \
  --scheduler.type=value_drop \
  --scheduler.switch_cost=0.02 \
  --scheduler.margin=0.03 \
  --scheduler.min_h=2 \
  --scheduler.max_h=16 \
  --output_dir=outputs/eval/vg_rtc_lite/libero10_value_drop
```

### 4.5 第五步做 VG-RTC-full

新增/修改：

```text
src/lerobot/policies/rtc/configuration_rtc.py
src/lerobot/policies/rtc/modeling_rtc.py
src/lerobot/policies/rtc/value_scheduler.py
examples/vg_rtc/eval_libero_vg_rtc_full.py
```

推荐新增配置：

```python
@dataclass
class ValueGuidedRTCConfig:
    enabled: bool = False
    scheduler_type: str = "chunk_value"
    switch_cost: float = 0.02
    latency_cost: float = 0.01
    discontinuity_cost: float = 0.01
    margin: float = 0.03
    min_execution_horizon: int = 2
    max_execution_horizon: int = 16
    min_guidance_weight: float = 2.0
    max_guidance_weight: float = 20.0
```

不要一开始把它塞进所有 policy。先在 `examples/vg_rtc` 的 eval script 里做 wrapper，等结果成立再抽进 `src/lerobot/policies/rtc`。

---

## 5. 真机实验什么时候开始

真机不要太早做。建议满足以下条件再上：

```text
1. fixed h sweep 在仿真中有明显曲线差异
2. AAC-lite 能跑通并有合理 h 分布
3. VG-RTC-lite 至少在 2-3 个 LIBERO task 上优于 AAC-lite 或 best fixed h
4. 代码能完整记录每个 episode 的 h / replan / value / latency
```

当前本机已有 PiPER 人在环说明：

```text
lenovo_README_PI05_VALUE_ACP.md
```

已验证硬件映射：

```text
follower: can1
leader: can0
wrist camera: RealSense D405
serial: 409122274629
```

真机第一批只做 2-3 个任务：

```text
1. block into bowl
2. button press / precise contact
3. pick-place-close drawer 或类似长程任务
```

真机对比：

```text
baseline pi05
fixed RTC
AAC-lite
VG-RTC-lite
```

每个方法至少：

```text
20 trials / task
```

记录：

```text
success
intervention count
episode steps
replan count
bad contact / collision
failure phase
```

---

## 6. 和 AAC 的关系怎么写

实验里一定要放 AAC，但不要把自己写成 AAC 改版。

论文叙事：

```text
AAC decides how many actions to execute from a predicted chunk using action entropy.
VG-RTC decides when to commit, replan, inpaint, and accept a new chunk using task value.
```

对应中文：

```text
AAC 解决的是执行长度选择；
VG-RTC 解决的是 RTC 中的异步承诺、重规划和跨 chunk 拼接控制。
```

差异表：

| 方法 | 决策信号 | 控制对象 | 是否进入 RTC 底层 | 是否关注任务回报 |
| --- | --- | --- | --- | --- |
| Fixed h | 手工超参 | 执行步数 | 否 | 否 |
| AAC | action entropy | 执行步数 | 否 | 间接 |
| Fixed RTC | 手工超参 | overlap/guidance | 是 | 否 |
| VG-RTC-lite | chunk value | 执行步数/replan | 部分 | 是 |
| VG-RTC-full | Q_continue vs Q_replan | timing/overlap/guidance/accept | 是 | 是 |

---

## 7. 最推荐的近期执行顺序

按这个顺序最稳：

```text
1. 用 lerobot-eval 跑 fixed h sweep。
2. 写 eval_libero_rtc_sweep.py，让 RTC success rate 能在仿真里测。
3. 写 eval_libero_aac_lite.py，把 AAC 作为 entropy baseline。
4. 收集 fixed/AAC/RTC rollouts，保存 per-step h、action、value、success。
5. 做 oracle value-drop 分析，确认 value-guided switch 的上限。
6. 训练轻量 chunk critic。
7. 做 VG-RTC-lite。
8. 再把 value 控制接进 RTC 的 execution_horizon / guidance_weight / accept-reject。
```

第一周不要碰真机，也不要完整复现 AAC 官方训练流程。

---

## 8. 当前最小可交付结果

最小 paper-precheck 结果应该包括：

```text
Figure 1: fixed h sweep，证明不同 h 成功率不同
Figure 2: AAC-lite 的 h 分布，证明 entropy 会随阶段变化
Figure 3: value profile，证明失败前存在 value drop
Table 1: Fixed h vs AAC-lite vs Fixed RTC vs VG-RTC-lite
Case study: 固定 RTC 坚持旧 chunk 失败，VG-RTC 提前 replan 成功
```

如果这些都成立，再做 VG-RTC-full 和真机。

---

## 9. 一句话结论

当前 Evo-RL 最适合的起步方式是：

```text
先用现有 lerobot-eval 做 fixed h 动机实验；
然后新增 LIBERO RTC eval loop；
再把 AAC-lite 放进同一个 loop 当强 baseline；
最后实现 VG-RTC scheduler，逐步从 value-based h 选择扩展到 RTC-level overlap/guidance 控制。
```

这样可以最大化利用当前仓库已有的 Pi0.5、RTC、value training 和 PiPER 真机管线，同时避免把创新点降级成“AAC 的 Q-value 版本”。
