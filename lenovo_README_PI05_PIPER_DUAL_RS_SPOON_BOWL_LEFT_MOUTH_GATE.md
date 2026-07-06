# [lenovo] Pi0.5 PiPER Dual RealSense Spoon Bowl to Left Mouth Gate Demo Notes

本文档记录当前这台机器上使用两个 RealSense D405、两个 PiPER CAN 口采集和训练这个任务：

```text
拿起勺子，从碗里舀起饭，倒到左边的嘴巴里；如果嘴巴闭合，就停在嘴巴前；如果嘴巴张开，就直接喂进去。
```

这个任务和普通放置任务不一样：闭嘴时停止是安全约束，不建议只靠 VLA 自己从 prompt 里学。推荐做法是：

1. policy 学会拿勺、舀饭、移动到嘴巴前的安全等待位、张嘴时完成喂入。
2. demo 时加一个 mouth gate：嘴巴闭合时禁止进入最终喂入口区域，嘴巴张开时才允许最后一段动作。
3. 第一版可以用人工人在环做 gate；稳定后再接视觉检测或外部开关做自动 gate。

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

- `wrist`: `409122274629`, D405
- `front`: `352122272924`, D405

当前常用 CAN 约定：

- leader arm / 主臂: `can0`
- follower arm / 从臂: `can1`
- camera keys: `observation.images.wrist`, `observation.images.front`

注意：最近 `can1` 出现过扫不到电机的问题。如果 `can1` 扫不到 8 个电机，不要继续用 `robot.port=can1` 跑 policy。先用下面的 CAN 检查确认哪个口能扫到 follower。

## 2. 任务文本

采集、训练、推理统一使用同一条英文任务文本：

```bash
export TASK_TEXT="Pick up the spoon, scoop rice from the bowl, feed the mouth on the left only if the mouth is open; if the mouth is closed, stop before the mouth"
```

同一个数据集内不要把 open-mouth 和 closed-mouth 写成两条不同任务文本。这里希望模型根据画面里的嘴巴状态做条件动作。

## 3. 采集前检查

### 3.1 关闭 RealSense Viewer

```bash
pkill -f realsense-viewer || true
```

### 3.2 检查 CAN

先激活环境：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
```

测试两个 CAN 口：

```bash
lerobot-setup-can --mode=test --interfaces=can0,can1
```

判断规则：

- `lerobot-record` 纯 policy 或采集 follower：`ROBOT_PORT` 对应的口必须能扫到 `8/8`。
- `lerobot-human-inloop-record` 或 teleop 采集：leader 和 follower 两个口都应该正常，总共能看到两条机械臂。
- 如果只有 `can0` 能扫到 8 个电机，先把 policy-only demo 的 `ROBOT_PORT` 改成 `can0`；但人在环 teleop 暂时不可用。

如果 CAN 口没有 UP，先尝试手动配置，不要加 `restart-ms`：

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

sudo ip link set can1 down || true
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up

ip -details link show can0
ip -details link show can1
```

### 3.3 检查两个 RealSense

```bash
rs-enumerate-devices
```

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-find-cameras realsense
```

按采集配置直接读帧检查：

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

## 4. 采集设计

先用假嘴或模型嘴做 demo，不要直接对真人嘴巴采集。勺子建议用软勺或轻量勺，饭量很少，速度调低。

推荐数据分布：

- 张嘴，正常舀饭并喂入：`40-60` 条。
- 闭嘴，舀饭后移动到嘴巴前安全等待位，停住 `3-5s`，不喂入：`40-60` 条。
- 先闭嘴等待，然后张嘴，再喂入：`20-30` 条。
- 本来张嘴，接近时突然闭嘴，停住或小幅后撤：`10-20` 条。

关键采集要求：

- `front` 相机必须清楚看到嘴巴开闭状态和嘴巴前等待位。
- `wrist` 相机必须清楚看到勺子、碗、饭和勺尖位置。
- 闭嘴样本不能只是 episode 失败结束，必须把“停在嘴巴前不动”作为成功演示。
- 所有样本都要经过相似的舀饭和接近嘴巴流程，否则模型可能把闭嘴理解成完全不执行任务。
- 嘴巴位置、碗位置、勺子初始姿态、饭量、光照都要有小幅变化。

建议分两轮采：

- pilot round：先采 `20-40` 条，训练一个小模型验证视觉和动作是否可学。
- full round：再补到 `120-160` 条，保证 closed-mouth 和 transition 样本足够。

## 5. 可选 teleop 检查

如果 CAN 检查确认 `can1` 是 follower、`can0` 是 leader：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export TELEOP_PORT=can0

lerobot-teleoperate \
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
  --display_data=true
```

如果主从臂端口和这里相反，只改 `ROBOT_PORT` 和 `TELEOP_PORT`，不要改相机 key。

## 6. 双 RealSense 采集命令

新建独立数据集：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export TELEOP_PORT=can0
export TASK_TEXT="Pick up the spoon, scoop rice from the bowl, feed the mouth on the left only if the mouth is open; if the mouth is closed, stop before the mouth"
export DATASET_ID=local/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1

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
  --dataset.single_task="$TASK_TEXT" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --resume=true
```

继续往同一个数据集追加数据时，在完整命令末尾加：

```bash
  --resume=true
```

试采时改短：

```bash
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --reset_to_zero_action=true \
```

## 7. 采集按键

普通 `lerobot-record` 采集时：

- `Right Arrow`: 提前结束当前 episode，并保存这一条。
- `Left Arrow`: 提前结束当前 episode，丢弃这一条并重录。
- `Esc`: 结束整轮采集，会保存已经完成的 episode。

`lerobot-human-inloop-record` 带 policy 人在环采集时，除了上面三个键，还会启用：

- `i`: 在 policy 执行和人工接管之间切换。
- `s`: 标记当前 episode 为成功，并结束当前 episode。
- `f`: 标记当前 episode 为失败，并结束当前 episode。

## 8. 采集后检查数据集

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-dataset-report --dataset local/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1
```

检查 `meta/info.json` 中应包含两个视频 key：

```text
observation.images.wrist
observation.images.front
```

如果只有 `observation.images.wrist`，说明采集命令没有真正带上第二个相机，不能直接用于双相机训练。

## 9. 本机单卡训练命令

这个新任务建议从 `lerobot/pi05_base` 重新训练，不建议从黄色盒子舀饭模型继续训。旧模型的动作习惯和目标区域不同，容易把“闭嘴停止”学偏。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_spoon_bowl_left_mouth_gate_dual_rs_bs32_30k_from_base

CUDA_VISIBLE_DEVICES=0 HF_DATASETS_CACHE=/tmp/hf/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=local/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1 \
  --dataset.root=/home/lenovo/Evo-RL/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1 \
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

本机 5090 上可以按之前 spoon 任务估计：`batch_size=32`、`30000` 步大约是一天量级。新任务如果数据集更大，数据读取可能略慢。

## 10. 远端四卡训练命令

如果远端代码和数据已经同步到：

```bash
/home/zhangyu/code/Evo-RL
/home/zhangyu/datasets/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1
```

四卡训练可以跑等样本量近似版本：每卡 `batch_size=32`，总 batch 约 `128`，步数用 `7500`。这和本机 `bs32 x 30000` 的样本数接近，但优化步数更少，不保证完全等价。

```bash
ssh -p 1141 zhangyu@10.103.92.120
```

远端执行：

```bash
cd /home/zhangyu/code/Evo-RL
source /home/zhangyu/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate evo-rl

RUN=pi05_piper_spoon_bowl_left_mouth_gate_dual_rs_bs32x4_7500_from_base

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_CACHE=/tmp/hf/datasets TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --main_process_port=29613 \
  "$(which lerobot-train)" \
  --dataset.repo_id=local/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1 \
  --dataset.root=/home/zhangyu/datasets/piper_spoon_scoop_bowl_to_left_mouth_gate_dual_rs_v1 \
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
  --steps=7500 \
  --log_freq=50 \
  --save_freq=2500 \
  --output_dir=/home/zhangyu/checkpoints/$RUN \
  --job_name=$RUN \
  --wandb.enable=false
```

如果远端速度仍然异常慢，先看 `updt_s`。粗略估计：

```text
预计训练时间 = steps * updt_s / 3600 小时
```

例如 `updt_s=2.6` 且 `steps=7500`，大约 `5.4` 小时；如果 `updt_s=12`，就是 `25` 小时。

## 11. 正常单独推理命令

训练完成后，先用假嘴跑 policy-only rollout。这个命令没有 leader teleop，不带人工接管。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_spoon_bowl_left_mouth_gate_dual_rs_bs32_30k_from_base/checkpoints/030000/pretrained_model
export TASK_TEXT="Pick up the spoon, scoop rice from the bowl, feed the mouth on the left only if the mouth is open; if the mouth is closed, stop before the mouth"
export DATASET_ID=local/eval_piper_spoon_bowl_left_mouth_gate_policy_round1
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_piper_spoon_bowl_left_mouth_gate_policy_round1

printf 'POLICY_PATH=<%s>\nDATASET_ID=<%s>\nDATASET_ROOT=<%s>\nTASK_TEXT=<%s>\n' \
  "$POLICY_PATH" "$DATASET_ID" "$DATASET_ROOT" "$TASK_TEXT"
test -f "$POLICY_PATH/config.json" && echo "policy ok" || { echo "policy path bad"; exit 1; }

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
lerobot-record \
  --robot.type=piper_follower \
  --robot.port="$ROBOT_PORT" \
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

首次创建 `eval_` 数据集时不要加 `--resume=true`。只有继续追加同一个 eval 数据集时再加。

## 12. 人在环 demo 命令

正式 demo 前建议先用人在环版本。闭嘴时人工接管让机械臂停在嘴巴前，张嘴时放开让 policy 完成最后喂入。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export TELEOP_PORT=can0
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_spoon_bowl_left_mouth_gate_dual_rs_bs32_30k_from_base/checkpoints/030000/pretrained_model
export TASK_TEXT="Pick up the spoon, scoop rice from the bowl, feed the mouth on the left only if the mouth is open; if the mouth is closed, stop before the mouth"
export DATASET_ID=local/eval_piper_spoon_bowl_left_mouth_gate_hil_round1
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_piper_spoon_bowl_left_mouth_gate_hil_round1

printf 'POLICY_PATH=<%s>\nDATASET_ID=<%s>\nDATASET_ROOT=<%s>\nTASK_TEXT=<%s>\n' \
  "$POLICY_PATH" "$DATASET_ID" "$DATASET_ROOT" "$TASK_TEXT"
test -f "$POLICY_PATH/config.json" && echo "policy ok" || { echo "policy path bad"; exit 1; }

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
lerobot-human-inloop-record \
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

## 13. Mouth gate 方案

纯 `lerobot-record --policy.path=...` 只能让 policy 自己判断嘴巴开闭；它没有内置安全门控。要做稳定 demo，建议按下面三层逐步做：

### 13.1 第一版：人工 gate

- 用 `lerobot-human-inloop-record`。
- 闭嘴时人工接管并保持在嘴巴前等待位。
- 张嘴时切回 policy 或人工完成最后一段喂入。
- 这个版本最适合先验证模型是否学会舀饭和接近嘴巴。

### 13.2 第二版：外部 open/closed 信号 gate

可以先用键盘、脚踏开关、按钮或简单视觉检测输出一个布尔量：

```text
mouth_open = true / false
```

控制规则：

```text
if mouth_open:
    allow policy action
else:
    block final feeding zone, hold at pre-mouth pose or retract slightly
```

这里的 pre-mouth pose 要在采集时固定成一个安全等待位：勺子在嘴巴前方几厘米，不接触嘴巴。

### 13.3 第三版：视觉检测 gate

如果要全自动，可以让 `front` 相机检测嘴巴状态：

- open: 嘴巴张开，允许最后喂入。
- closed: 嘴巴闭合，禁止进入最终喂入口区域。

检测器不需要很复杂，第一版可以是二分类器或阈值检测，但必须在假嘴上反复验证。安全逻辑要放在 action 发送前，而不是 episode 结束后。

## 14. 关键注意事项

- 这个任务不要直接拿真人嘴巴做首轮实验，先用假嘴或模型嘴。
- 闭嘴样本要作为“成功停止”采集，不要只采失败。
- 训练时从 `lerobot/pi05_base` 起，不要从旧黄色盒子 spoon 模型 resume。
- 两个相机 key 必须固定为 `wrist` 和 `front`；采集、训练、推理必须一致。
- 如果 `can1` 扫不到电机，先修 CAN 或把 `ROBOT_PORT` 换到能扫到 8 个电机的口。
- 第一次新建 `eval_` 数据集时不要加 `--resume=true`。
- policy-only demo 没有交互门控；真正演示闭嘴停止时，至少先用人在环版本。
