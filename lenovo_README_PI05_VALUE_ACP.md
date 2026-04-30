# [lenovo] Pi0.5 Piper: Value + ACP + VLA Retrain

本文档只针对当前这台机器上的 `pi05 piper` 流程，目的是把后续离线 RL 迭代固定成一套可重复执行的步骤。

适用前提：

- 你已经完成了 `pi05` 的基础 VLA 训练
- 你已经完成了分机推理验证
- 当前基础策略 checkpoint 为：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model
```

- 当前原始数据集目录为：

```bash
/home/lenovo/Evo-RL/piper_multitask_v1
```

---

## 1. 总体流程

每一轮都按下面顺序做：

1. 冻结一份数据集快照 `D_k`
2. 在 `D_k` 上训练 value function
3. 在 `D_k` 上跑 value inference，写回 `value / advantage / indicator`
4. 用带 `indicator` 的 `D_k` 重新训练 `pi05`
5. 拿新策略去部署、收新数据，再进入下一轮

注意：

- `lerobot-value-infer` 会直接原地改数据集 parquet，不是只在 `output_dir` 里存结果
- 所以不要直接在原始数据集 `piper_multitask_v1` 上跑 value inference
- 先复制一份数据集快照，再在快照上打标签

---

## 2. 当前仓库中的对应关系

- Value training: `lerobot-value-train`
- Value inference: `lerobot-value-infer`
- ACP VLA retrain: `lerobot-train --acp.enable=true`

当前实现约束：

- `value_train` 目前只支持 `--value.type=pistar06`
- `value_train` 依赖 episode 成功标签，默认字段是 `episode_success`
- `ACP` 重训时会把 `acp_indicator` 注入到 task 文本里训练 `pi05`

---

## 3. Iteration 1

### 3.1 先复制一份数据集快照

```bash
cd /home/lenovo/Evo-RL
cp -a /home/lenovo/Evo-RL/piper_multitask_v1 /home/lenovo/Evo-RL/piper_multitask_v1_iter1
```

### 3.2 统一变量

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

export DATASET_ID=piper_multitask_v1_iter1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_multitask_v1_iter1

export BASE_POLICY=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model

export VALUE_RUN=pi05_piper_value_iter1_bs16_32k
export TAG=iter1
export POLICY_RUN=pi05_piper_acp_iter1
```

### 3.3 训练 value function

```bash
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-value-train \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.use_gradient_checkpointing=true \
  --batch_size=16 \
  --steps=32000 \
  --value.scheduler_warmup_steps=2000 \
  --value.scheduler_decay_steps=32000 \
  --output_dir=outputs/value_train/$VALUE_RUN \
  --job_name=$VALUE_RUN \
  --wandb.disable_artifact=true \
  --wandb.enable=true
```

说明：

- 这条命令是按当前这张 `32G` 卡调整过的 value 训练配置
- 原始默认配方是 `batch_size=64, steps=8000`，总样本预算是 `64 x 8000 = 512000`
- 现在改成 `batch_size=16` 后，为了保持接近的总样本预算，需要把步数同步放大到 `32000`
- `value.use_gradient_checkpointing=true` 是为了把显存压到 32G 可用范围
- `wandb.disable_artifact=true` 是为了避免训练结束后卡在 checkpoint artifact 上传
- 这一步会在 `outputs/value_train/$VALUE_RUN` 下保存 value 模型
- 后面 `value_infer` 直接使用这个目录作为 `--inference.checkpoint_path`

### 3.4 运行 value inference，并把标签写回数据集

```bash
lerobot-value-infer \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --inference.checkpoint_path=outputs/value_train/$VALUE_RUN \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_$TAG \
  --acp.advantage_field=complementary_info.advantage_$TAG \
  --acp.indicator_field=complementary_info.acp_indicator_$TAG \
  --output_dir=outputs/value_infer/$VALUE_RUN \
  --job_name=$VALUE_RUN.infer
```

这一步结束后，数据集 `$DATASET_ROOT` 里会新增三列：

```bash
complementary_info.value_iter1
complementary_info.advantage_iter1
complementary_info.acp_indicator_iter1
```

### 3.5 用 ACP 标签重训 pi05

```bash
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=$BASE_POLICY \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=5000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_$TAG \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/train/$POLICY_RUN \
  --job_name=$POLICY_RUN \
  --wandb.enable=false
```

完成后，新策略通常从下面路径取：

```bash
outputs/train/$POLICY_RUN/checkpoints/<STEP>/pretrained_model
```

如果你训练跑满并使用最后一个 checkpoint，也可以直接用最后一轮对应的 `pretrained_model`。

补充：

- 这条命令刻意对齐你之前能稳定跑通的 `pi05 bs32` baseline，再额外叠加 `ACP` 参数
- `policy.gradient_checkpointing=true` 和 `policy.train_expert_only=true` 是这台 `32G` 卡上能跑 `batch_size=32` 的关键
- `ACP` 本身主要只是改 task 文本，不是这次爆显存的主因
- `policy.push_to_hub=false` 是因为这个仓库默认会要求推 Hugging Face Hub；如果只是本地训练，不关掉就会报 `policy.repo_id` 缺失
- 如果后面要重新开 `wandb`，建议同时加 `--wandb.disable_artifact=true`，避免训练结束后卡在 checkpoint artifact 上传

---

## 4. Iteration 2 及之后

下一轮重复同样流程，但一定要换：

- 数据集快照目录
- `VALUE_RUN`
- `TAG`
- `POLICY_RUN`
- `BASE_POLICY`

示例：

```bash
cp -a /home/lenovo/Evo-RL/piper_multitask_v1 /home/lenovo/Evo-RL/piper_multitask_v1_iter2

export DATASET_ID=piper_multitask_v1_iter2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_multitask_v1_iter2

export BASE_POLICY=/home/lenovo/Evo-RL/outputs/train/pi05_piper_acp_iter1/checkpoints/<STEP>/pretrained_model

export VALUE_RUN=pi05_piper_value_iter2_bs16_32k
export TAG=iter2
export POLICY_RUN=pi05_piper_acp_iter2
```

然后继续执行：

1. `lerobot-value-train`
2. `lerobot-value-infer`
3. `lerobot-train --acp.enable=true`

---

## 5. 本机直连 CAN 的人在环推理与采集

如果后面不走两机 `async_inference`，而是直接在这台 GPU 机器上连 PiPER 的 CAN、相机、leader 做人在环推理，推荐直接用：

```bash
lerobot-human-inloop-record
```

这条链会同时完成：

- 本机加载 policy checkpoint
- 本机推理
- leader 跟随 policy
- `i` 键切到人工接管
- 录制新数据并写入 `is_intervention / policy_action / episode_success`

当前这台机器已验证过的单臂 PiPER 映射：

- follower: `can1`
- leader: `can0`
- wrist camera: RealSense D405
- camera serial: `409122274629`

### 5.1 先激活环境和 CAN

```bash

cd /home/lenovo/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000 1-6.2:1.0
bash can_activate.sh can1 1000000 1-6.4:1.0

cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

cd /home/lenovo/Evo-RL
lerobot-setup-can --mode=test --interfaces=can0,can1
```

### 5.2 可选：先做一次 teleop 联通性检查

```bash
lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can1 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --display_data=true
```

### 5.3 跑本机人在环推理

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_acp_iter1/checkpoints/030000/pretrained_model
# 如果你想先部署 baseline，而不是 ACP 重训后的模型，可以改成：
# export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model
export TASK_TEXT="Put the green block into the dark green bowl"
export DATASET_ID=local/eval_piper_hil_round1
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_piper_hil_round1

lerobot-human-inloop-record \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can1 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --dataset.single_task="$TASK_TEXT" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --policy.path=$POLICY_PATH \
  --acp_inference.enable=true \
  --acp_inference.use_cfg=false
```

如果这里部署的是 baseline 模型，而不是 ACP 重训后的模型，去掉最后两行：

```bash
  --acp_inference.enable=true \
  --acp_inference.use_cfg=false
```

说明：

- `TASK_TEXT` 继续用原始任务文本，不需要手动改任务名字
- ACP 推理时也不要手动把 task 改成别的名字；代码会在推理侧自动追加 positive tag
- 这台机器当前使用的 `can_activate.sh` 路径是 `/home/lenovo/piper_sdk/piper_sdk`
- 这台机器当前识别到的 USB 硬件地址是：`can0 -> 1-6.2:1.0`，`can1 -> 1-6.4:1.0`
- 这里默认新建一套独立的人在环数据：`local/eval_piper_hil_round1`
- 因为带 `policy.path` 的录制会触发仓库里的命名校验，所以数据集名必须以 `eval_` 开头
- 第一次新建 `eval_piper_hil_round1` 时不要加 `--resume=true`
- `play_sounds=false` 用来关闭录制过程里的语音播报
- `gravity_comp_tx_ratio=[1,1,1,1,1,1]` 是当前这套 MIT 接管里比较实用的保守上限；继续细调收益不高，可以先用这版采数
- 如果后面继续往这套人在环数据里追加，在完整命令最后再加：

```bash
  --resume=true
```

### 5.4 运行时热键

- `i`: policy 和人工接管之间切换
- `s`: 标记成功并结束当前 episode
- `f`: 标记失败并结束当前 episode
- `Right Arrow`: 提前结束当前 episode
- `Left Arrow`: 丢弃当前 episode 并重录
- `Esc`: 结束整轮采集

---

## 6. 复现时必须保留的东西

如果你想几周后还能完全复现某一轮，至少保留以下内容：

- 数据集快照目录，例如 `piper_multitask_v1_iter1`
- value 训练输出目录，例如 `outputs/value_train/pi05_piper_value_iter1_bs16_32k`
- policy 训练输出目录，例如 `outputs/train/pi05_piper_acp_iter1`
- 当轮的 tag 名称，例如 `iter1`
- 当轮使用的基础策略路径
- 当时仓库的 git commit

建议每轮都额外记一份：

- value run 名称
- policy run 名称
- `n_step`
- `positive_ratio`
- `indicator_dropout_prob`
- 实际用于部署的 checkpoint 路径

---

## 7. 数据要求检查

在开始 value 训练前，至少确认：

- 数据集里有 `episode_success`
- 数据集里的 `task` 文本是正常的
- `episode_index / frame_index / index` 完整存在

补充：

- `complementary_info.is_intervention` 如果存在，`value_infer` 会优先把 intervention 帧标成正样本
- 如果这个字段不存在，脚本会按全 0 处理，不会报错

---

## 8. 最简执行顺序

如果只是快速开始，按下面顺序照跑即可：

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

cp -a /home/lenovo/Evo-RL/piper_multitask_v1 /home/lenovo/Evo-RL/piper_multitask_v1_iter1

export DATASET_ID=piper_multitask_v1_iter1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_multitask_v1_iter1
export BASE_POLICY=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model
export VALUE_RUN=pi05_piper_value_iter1_bs16_32k
export TAG=iter1
export POLICY_RUN=pi05_piper_acp_iter1
```

然后依次执行：

1. `lerobot-value-train`
2. `lerobot-value-infer`
3. `lerobot-train`

---

## 9. 当前建议

第一次做这条链时，不要同时改 RTC、异步推理框架、数据格式。

先把下面这条闭环做通：

1. `baseline dataset`
2. `value train`
3. `value infer`
4. `ACP pi05 retrain`
5. `新 checkpoint 部署验证`

等这条闭环稳定之后，再去比较：

- baseline `pi05`
- ACP `pi05`
- ACP + RTC `pi05`
