# [lenovo] Pi0.5 PiPER Dual RealSense Spoon Scoop Collection and Training Notes

本文档记录当前这台机器上使用两个 RealSense D405、两个 PiPER CAN 口采集“拿起勺子，从黄色盒子里舀起食物，放到右边的碗里面”任务数据，并训练 Pi0.5 的命令。

## 1. 当前硬件映射

仓库路径：

```bash
/home/lenovo/Evo-RL
```

Conda 环境：

```bash
conda activate evo-rl
```

当前实测 RealSense：

- `wrist`: `409122274629`, USB3 `2-1`, D405
- `front`: `352122272924`, USB3 `2-2`, D405

当前实测 CAN 适配器：

- `can0 -> 1-10:1.0`
- `can1 -> 1-4:1.0`

当前命令约定：

- leader arm / 主臂: `can0`
- follower arm / 从臂: `can1`
- camera keys: `observation.images.wrist`, `observation.images.front`

注意：主臂是人手拖动的 teleop/leader，从臂是执行动作的 robot/follower。本文件以 `leader=can0`、`follower=can1` 为准；如果 USB 重新插拔，只改 `can_activate.sh` 的 USB 地址，不要把主从臂端口写反。

## 2. 任务文本

用户任务：

```text
拿起勺子，从黄色盒子里舀起食物，放到右边的碗里面
```

采集和训练统一使用英文任务文本：

```bash
export TASK_TEXT="Pick up the spoon, scoop food from the yellow box, then put the food into the bowl on the right"
```

同一个数据集内不要随意改任务文本。后续推理也使用同一条文本，避免 task embedding 和训练数据不一致。

## 3. 采集前检查

### 3.1 关闭 RealSense Viewer

如果开着 `realsense-viewer`，它会占用 `/dev/video*`，导致 LeRobot 代码打开相机失败。

```bash
pkill -f realsense-viewer || true
```

### 3.2 激活 CAN

```bash
cd /home/lenovo/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000 1-10:1.0
bash can_activate.sh can1 1000000 1-4:1.0
```

检查 CAN：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-setup-can --mode=test --interfaces=can0,can1
```

预期每个 PiPER CAN 口能扫到对应电机。如果扫不到，先检查机械臂供电、CAN 线、USB 口映射是否变化。

### 3.3 检查两个 RealSense

列出 RealSense：

```bash
rs-enumerate-devices
```

项目内检查：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-find-cameras realsense
```

如果 `lerobot-find-cameras realsense` 能列出设备但保存图片失败，通常是默认 profile 或已有进程占用问题。按采集配置做一次直接读帧检查：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

python -c 'from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig; cams={"wrist":"409122274629","front":"352122272924"}; opened=[];
try:
    for name, serial in cams.items():
        cam=RealSenseCamera(RealSenseCameraConfig(serial_number_or_name=serial,width=640,height=480,fps=30,warmup_s=1)); cam.connect(); img=cam.read(); print("{} {} shape={} dtype={}".format(name, serial, getattr(img, "shape", None), getattr(img, "dtype", None))); opened.append(cam)
finally:
    [c.disconnect() for c in opened]'
```

预期输出包含：

```text
wrist 409122274629 shape=(480, 640, 3) dtype=uint8
front 352122272924 shape=(480, 640, 3) dtype=uint8
```

## 4. 可选 teleop 检查

正式采集前先确认 leader 能控制 follower，两个相机能显示：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can0 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --display_data=true
```

## 5. 双 RealSense 采集命令

建议新建独立数据集，不要混到已有单相机数据集 `piper_multitask_v1` 或 `piper_multitask_v4` 里。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the spoon, scoop food from the yellow box, then put the food into the bowl on the right"
export DATASET_ID=local/piper_spoon_scoop_yellow_box_to_right_bowl_dual_rs_v2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_spoon_scoop_yellow_box_to_right_bowl_dual_rs_v2

lerobot-record \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can0 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --dataset.single_task="$TASK_TEXT" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false
```

继续往同一个数据集追加数据时，在完整命令末尾加：

```bash
  --resume=true
```

如果只是先试采一两条，把 episode 数量和时间改短：

```bash
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --reset_to_zero_action=true \
```

## 6. 采集按键

普通 `lerobot-record` 采集时：

- `Right Arrow`: 提前结束当前 episode，并保存这一条。
- `Left Arrow`: 提前结束当前 episode，丢弃这一条并重录。
- `Esc`: 结束整轮采集，会保存已经完成的 episode。

`lerobot-human-inloop-record` 带 policy 人在环采集时，除了上面三个键，还会启用：

- `i`: 在 policy 执行和人工接管之间切换。
- `s`: 标记当前 episode 为成功，并结束当前 episode。
- `f`: 标记当前 episode 为失败，并结束当前 episode。

注意：普通 `lerobot-record` 默认不写 `episode_success`，所以 `s/f` 只在 `lerobot-human-inloop-record` 或显式打开 episode outcome labeling 时有意义。

## 7. 采集后检查数据集

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-dataset-report --dataset local/piper_spoon_scoop_yellow_box_to_right_bowl_dual_rs_v1
```

检查 `meta/info.json` 中应包含两个视频 key：

```text
observation.images.wrist
observation.images.front
```

如果只有 `observation.images.wrist`，说明采集命令没有真正带上第二个相机，不能直接用于双相机训练。

## 8. 双相机训练命令

采集完成后训练 Pi0.5：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_spoon_yellow_box_bowl_dual_rs_bs32_30k

CUDA_VISIBLE_DEVICES=0 HF_DATASETS_CACHE=/tmp/hf/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=local/piper_spoon_scoop_yellow_box_to_right_bowl_dual_rs_v1 \
  --dataset.root=/home/lenovo/Evo-RL/piper_spoon_scoop_yellow_box_to_right_bowl_dual_rs_v1 \
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

当前推荐用于推理的本地 checkpoint 是从干净 `005000` resume 后保存的 `025000`：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_spoon_yellow_box_bowl_dual_rs_v2_bs32_30k_resume_from005000_local5090_20260528_1901/checkpoints/025000/pretrained_model
```

## 9. 推理或人在环采集

### 9.1 正常单独推理

训练完成后，用双相机 checkpoint 跑本机单卡 policy rollout。这个命令不带 leader teleop，只让 policy 控制 follower，同时把评估 episode 录到 `eval_` 数据集里：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/remote_downloads/pi05_piper_spoon_yellow_box_bowl_dual_rs_v2_bs32x4_7500_from_base_20260528_007500/pretrained_model
export TASK_TEXT="Pick up the spoon, scoop food from the yellow box, then put the food into the bowl on the right"
export DATASET_ID=local/eval_piper_spoon_yellow_box_bowl_dual_rs_policy_round2
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_piper_spoon_yellow_box_bowl_dual_rs_policy_round2

printf 'POLICY_PATH=<%s>\nDATASET_ID=<%s>\nDATASET_ROOT=<%s>\nTASK_TEXT=<%s>\n' \
  "$POLICY_PATH" "$DATASET_ID" "$DATASET_ROOT" "$TASK_TEXT"
test -f "$POLICY_PATH/config.json" && echo "policy ok" || { echo "policy path bad"; exit 1; }

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
lerobot-record \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK_TEXT" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=10 \
  --reset_to_zero_action=true \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --policy.device=cuda \
  --policy.path="$POLICY_PATH"
```

如果这里报 `Couldn't instantiate class RecordConfig using the given arguments.`，优先检查上面的
`policy ok` 是否打印出来。这个错误通常发生在命令行解析阶段，常见原因是 `$POLICY_PATH` 为空或目录里没有
`config.json`，导致 `lerobot-record` 没有拿到 policy，也没有 teleop。

首次创建这个 `eval_` 数据集时不要加 `--resume=true`。如果误加 `--resume=true` 且本地还没有完整的
`$DATASET_ROOT/meta/info.json`，LeRobot 会尝试去 Hugging Face Hub 查
`local/eval_...` 的 dataset refs；在 `HF_HUB_OFFLINE=1` 下会报
`OfflineModeIsEnabled`。

如果一次失败运行已经留下了只有 `meta/info.json`、没有 episode 数据的半成品 eval 目录，下一次首次运行
应先换一个新的 `DATASET_ID/DATASET_ROOT`，或确认不要该半成品后删除对应目录；不要直接对半成品目录加
`--resume=true`。

### 9.2 人在环推理

需要人工接管/标注成功失败时，用双相机 checkpoint 跑人在环 rollout：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_spoon_yellow_box_bowl_dual_rs_v2_bs32_30k_resume_from005000_local5090_20260528_1901/checkpoints/025000/pretrained_model
export TASK_TEXT="Pick up the spoon, scoop food from the yellow box, then put the food into the bowl on the right"
export DATASET_ID=local/eval_piper_spoon_yellow_box_bowl_dual_rs_hil_round2
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_piper_spoon_yellow_box_bowl_dual_rs_hil_round2

printf 'POLICY_PATH=<%s>\nDATASET_ID=<%s>\nDATASET_ROOT=<%s>\nTASK_TEXT=<%s>\n' \
  "$POLICY_PATH" "$DATASET_ID" "$DATASET_ROOT" "$TASK_TEXT"
test -f "$POLICY_PATH/config.json" && echo "policy ok" || { echo "policy path bad"; exit 1; }

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
lerobot-human-inloop-record \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can0 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK_TEXT" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --policy.device=cuda \
  --policy.path="$POLICY_PATH"
```

如果继续追加同一个 eval 数据集，在命令末尾加：

```bash
  --resume=true
```

## 10. 关键注意事项

- 两个相机 key 必须固定为 `wrist` 和 `front`；采集、训练、推理必须一致。
- 现有 `piper_multitask_v1` 和 `piper_multitask_v4` 都是单相机数据集，不能直接训练双相机模型。
- `realsense-viewer` 不能和采集同时运行。
- 当前 follower/leader 采用主从修正后的命令约定：`robot.port=can1`，`teleop.port=can0`。
- 如果 USB 重新插拔后 CAN 映射变化，先用 `/home/lenovo/piper_sdk/piper_sdk/find_all_can_port.sh` 或 `ip -details link show type can` 重新确认，再改 `can_activate.sh` 的第三个参数。
- 训练时 `--dataset.video_backend=pyav` 是当前本机已验证可用的后端。
- 本文训练和推理命令按本机单卡写法固定 `CUDA_VISIBLE_DEVICES=0`；如果换 GPU，只改这个环境变量。
- 第一次新建 `eval_` 数据集时不要加 `--resume=true`；只有继续追加同一个 eval 数据集时再加。
