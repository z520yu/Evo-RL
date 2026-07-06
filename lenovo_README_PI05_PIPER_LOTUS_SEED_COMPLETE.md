# [lenovo] Piper 莲子到左侧嘴巴：采集、训练、远端 LoRA、本地推理完整命令

当前任务：

```text
从莲子盒里夹起一个莲子，送到左边的嘴巴目标里。
```

固定 task text，采集、训练、推理都保持一致：

```bash
export TASK_LOTUS="Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"
```

常用路径：

```bash
export EVO_ROOT=/home/lenovo/Evo-RL
export DATASET_ID=local/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2
```

硬件约定：

```text
follower / robot: can1
leader / teleop: can0
wrist RealSense: 409122274629
front RealSense: 352122272924
```

如果实际 CAN 口相反，只改 `ROBOT_PORT` 和 `TELEOP_PORT`。

## 1. 激活环境和 CAN

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
```

激活两个 CAN 口。当前这台机器优先用 `can_activate.sh` 绑定 USB 地址，不直接用 `ip link set can0 ...`：

```bash
cd /home/lenovo/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000 1-10:1.0
bash can_activate.sh can1 1000000 1-4:1.0
```

这里的当前映射是：

```text
can0 -> 1-10:1.0
can1 -> 1-4:1.0
```

如果 USB 重新插拔导致地址变化，先重新确认：

```bash
cd /home/lenovo/piper_sdk/piper_sdk
bash find_all_can_port.sh
```

然后只改 `can_activate.sh` 的第三个参数。

检查：

```bash
cd /home/lenovo/Evo-RL
lerobot-setup-can --mode=test --interfaces=can0,can1
```

预期 follower 对应的口能扫到 `8/8` 个电机。扫不到就不要继续采集或推理。

## 2. 采集数据示例命令

第一次创建新数据集时不要加 `--resume=true`。继续往已有数据集追加时加 `--resume=true`。

下面示例是往 `piper_lotus_seed_pick_to_left_mouth_dual_rs_v2` 追加 25 条：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export TELEOP_PORT=can0
export DATASET_ID=local/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2
export TASK_LOTUS="Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"

lerobot-record \
  --robot.type=piper_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK_LOTUS" \
  --dataset.num_episodes=25 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --resume=true
```

采集时只保留成功样本：夹偏、掉落、碰翻盒子、没送进嘴巴目标，都丢弃重录。

## 3. 本地 lerobot 版本训练命令

本地 5090 推荐 `batch_size=32`、`30000` 步，从 `lerobot/pi05_base` 重新训练：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_$(date +%Y%m%d_%H%M)
LOG=/home/lenovo/Evo-RL/outputs/train/logs/${RUN}.log
mkdir -p /home/lenovo/Evo-RL/outputs/train/logs

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/tmp/hf/datasets \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=local/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2 \
  --dataset.root=/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2 \
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
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false 2>&1 | tee "$LOG"
```

已有可测 lerobot checkpoint 示例：

```bash
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026/checkpoints/025000/pretrained_model
```

## 4. 远端 OpenPI LoRA 训练命令

先把本地数据集同步到远端：

```bash
rsync -avh --info=progress2 \
  /home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2/ \
  L20_5:/home/zhangyu/datasets/piper_lotus_seed_pick_to_left_mouth_dual_rs_v2/
```

远端启动 LoRA 训练：

```bash
ssh L20_5
```

```bash
cd /home/zhangyu/datasets/openpi-lotus-v2-workspace

export PYTHONPATH=/home/zhangyu/datasets/openpi-lotus-v2-workspace/src
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CONFIG=pi05_piper_lotus_seed_v2_lora
EXP=official_pi05_piper_lotus_seed_v2_lora_from_base_$(date +%Y%m%d_%H%M%S)
CKPT_BASE=/home/zhangyu/rfm_runs_local/lotus_seed_v2_lora_from_base_30k/checkpoints
LOG_DIR=/home/zhangyu/rfm_runs_local/lotus_seed_v2_lora_from_base_30k/logs
mkdir -p "$LOG_DIR"

/home/zhangyu/venvs/openpi-baseline/bin/python scripts/train.py "$CONFIG" \
  --exp-name "$EXP" \
  --overwrite \
  --seed 42 \
  --num-train-steps 30000 \
  --batch-size 16 \
  --num-workers 8 \
  --fsdp-devices 2 \
  --checkpoint-base-dir "$CKPT_BASE" \
  --save-interval 5000 \
  --no-wandb-enabled 2>&1 | tee "$LOG_DIR/${EXP}.log"
```

如果从同一个 exp 断点续训，把 `EXP` 改成原来的名字，并使用：

```bash
/home/zhangyu/venvs/openpi-baseline/bin/python scripts/train.py "$CONFIG" \
  --exp-name "$EXP" \
  --resume \
  --seed 42 \
  --num-train-steps 30000 \
  --batch-size 16 \
  --num-workers 8 \
  --fsdp-devices 2 \
  --checkpoint-base-dir "$CKPT_BASE" \
  --save-interval 5000 \
  --no-wandb-enabled
```

当前本地 OpenPI 推理脚本默认使用的远端 exp：

```text
official_pi05_piper_lotus_seed_v2_lora_from_base_20260615_164629
```

## 5. 本地 lerobot 版本推理命令

这个方式直接用 `lerobot-record --policy.path`。首次创建 eval 数据集不要加 `--resume=true`。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026/checkpoints/025000/pretrained_model
export TASK_LOTUS="Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"
export EVAL_TAG=$(date +%Y%m%d_%H%M%S)
export DATASET_ID=local/eval_lotus_seed_lerobot_${EVAL_TAG}
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_lotus_seed_lerobot_${EVAL_TAG}

test -f "$POLICY_PATH/config.json" && test -f "$POLICY_PATH/model.safetensors" \
  && echo "policy ok" || { echo "policy path bad or incomplete"; exit 1; }

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
lerobot-record \
  --robot.type=piper_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK_LOTUS" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=10 \
  --reset_to_zero_action=true \
  --reset_gripper_pos=60 \
  --reset_before_first_episode=true \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --policy.device=cuda \
  --policy.path="$POLICY_PATH"
```

如果 reset 动作离人太近，先不要用自动 reset，把命令里的 reset 相关参数删掉或改成：

```bash
--dataset.reset_time_s=0
```

然后手动把机械臂放到安全起始位再开始每个 episode。

## 6. 本地 OpenPI 版本推理命令

OpenPI 推理是两终端方式：一个终端启动 JAX policy server，另一个终端运行机器人 client。

### 6.1 下载远端 LoRA checkpoint

例如下载最终的 `29999`：

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
bash scripts/download_lotus_seed_checkpoint.sh 29999
```

下载后的本地路径：

```text
/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_lotus_seed_v2_lora/official_pi05_piper_lotus_seed_v2_lora_from_base_20260615_164629/29999
```

### 6.2 终端 1：启动 OpenPI policy server

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

export CKPT=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_lotus_seed_v2_lora/official_pi05_piper_lotus_seed_v2_lora_from_base_20260615_164629/29999

./.venv/bin/python scripts/serve_piper_lotus_seed_policy.py \
  --checkpoint-dir "$CKPT" \
  --port 8000
```

第一次会 JAX 编译，可能慢几十秒。

### 6.3 终端 2：先 dry-run

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

python scripts/openpi_piper_eval_client.py \
  --host localhost \
  --port 8000 \
  --robot-port can1 \
  --num-episodes 1 \
  --episode-time-s 10 \
  --reset-time-s 0 \
  --dry-run
```

### 6.4 终端 2：真机 eval

```bash
python scripts/openpi_piper_eval_client.py \
  --host localhost \
  --port 8000 \
  --robot-port can1 \
  --num-episodes 5 \
  --episode-time-s 40 \
  --reset-time-s 12 \
  --reset-gripper-pos 60 \
  --max-reset-joint-delta-per-step 1 \
  --max-reset-gripper-delta-per-step 3 \
  --query-every 10
```

只执行慢速 reset，不推理：

```bash
python scripts/openpi_piper_eval_client.py \
  --robot-port can1 \
  --reset-only \
  --reset-time-s 12 \
  --reset-gripper-pos 60 \
  --max-reset-joint-delta-per-step 1 \
  --max-reset-gripper-delta-per-step 3
```

停止：

- 停止当前机器人 client：在 client 终端按 `Ctrl-C`。
- 停止 OpenPI server：在 server 终端按 `Ctrl-C`。
- 软件停止不是急停；旁边有人时优先使用硬件急停或断使能。
