# [lenovo] Pi0.5 PiPER v4 Training and Local Inference Notes

本文档只记录当前这台机器上的 `piper_multitask_v4` 流程。

## 1. 当前结果

仓库路径：

```bash
/home/lenovo/Evo-RL
```

Conda 环境：

```bash
conda activate evo-rl
```

数据集：

```bash
/home/lenovo/Evo-RL/piper_multitask_v4
```

数据集概要：

- robot type: `piper_follower`
- fps: `30`
- episodes: `50`
- frames: `43671`
- observation state: `7` 维
- action: `7` 维
- camera key: `observation.images.wrist`
- 已确认任务文本：
  - `Open the first dark green drawer, pick up the blue block from the table, place the block inside, and close the drawer`

当前训练输出：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_v4_bs32_30k_0505_1748
```

最终推理 checkpoint：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_v4_bs32_30k_0505_1748/checkpoints/030000/pretrained_model
```

推理只需要 `pretrained_model` 目录。`training_state` 是恢复训练用的，不需要复制到推理机器。

## 2. 已跑训练命令

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

RUN=pi05_piper_v4_bs32_30k_0505_1748

HF_DATASETS_CACHE=/tmp/hf/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=piper_multitask_v4 \
  --dataset.root=/home/lenovo/Evo-RL/piper_multitask_v4 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
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
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false
```

说明：

- `HF_DATASETS_CACHE=/tmp/hf/datasets` 是当前机器上最稳的缓存位置。
- `--dataset.video_backend=pyav` 是当前数据集已验证可用的后端。
- `--policy.train_expert_only=true` 只训练动作专家。
- `--policy.push_to_hub=false` 是必须的，否则不传 `policy.repo_id` 会报错。

## 3. 本机推理前准备

### 3.1 激活 CAN

当前这台机器上先把 PiPER 的 CAN 拉起来：

```bash
cd /home/lenovo/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000 1-7.2:1.0
```

本机纯 policy 推理只需要 follower 的 `can0`。如果后面要接 leader 做 teleop 或人在环检查，再额外激活 `can1`：

```bash
cd /home/lenovo/piper_sdk/piper_sdk
bash can_activate.sh can1 1000000 1-7.4:1.0
```

然后回到 Evo-RL 里做一次检查：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
lerobot-setup-can --mode=test --interfaces=can0
```

如果同时激活了 leader 的 `can1`，检查命令改成：

```bash
lerobot-setup-can --mode=test --interfaces=can0,can1
```

### 3.2 检查相机

```bash
lerobot-find-cameras realsense
```

因为训练数据只有一个相机 key：`wrist`，推理时相机名也必须保持一致。这里沿用你其他命令里的 RealSense 写法：

```bash
--robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}'
```

注意：

- `409122274629` 先按你现有命令写死；如果这台机子的 RealSense 序列号不同，就换成实际值。
- `lerobot-find-cameras realsense` 如果没扫到，先检查 RealSense 的 USB、供电和 `pyrealsense2` 环境。
- 这里不要再用 OpenCV 路径，腕部相机是 RealSense。

## 4. 本机 rollout

这里直接用 `lerobot-human-inloop-record` 跑本机 rollout。它同时带上 follower policy 和 leader teleop，命令上最贴近真机验证流程。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v4_bs32_30k_0505_1748/checkpoints/last/pretrained_model
export TASK_TEXT="Open the first dark green drawer, pick up the blue block from the table, place the block inside, and close the drawer"
export DATASET_ID=local/eval_piper_v4_hil_rollout
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_piper_v4_hil_rollout

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
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
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --policy.device=cuda \
  --policy.path=$POLICY_PATH \
  --resume=true
```

这条命令只做一轮本机 rollout。想更短一点就把 `--dataset.episode_time_s=120` 改成 `20`。

## 5. 备注

- `checkpoints/last` 当前指向 `030000`。
- `wrist` 这个 key 不能改，否则预处理对不上训练数据。
- `can_activate.sh` 路径是 `/home/lenovo/piper_sdk/piper_sdk`。
- 如果后面 CAN 口或 USB 口映射变了，把 `can0` 和 `1-6.2:1.0` 换成实际值。
