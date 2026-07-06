# [lenovo] Pi0.5 PiPER Dual RealSense White Box QR Reorientation Collection

本文档记录一个独立的白盒姿态调整任务：白色长方体初始已经位于绿色长方形托盘内，机械臂只需要在托盘内调整盒子姿态，使二维码所在面朝上并清晰可见。

这个任务不包含“把盒子搬进绿色托盘”或“把盒子搬到右侧成功区”。采集、训练和推理必须使用同一条任务文本，避免模型混入额外搬运语义。

## 1. 任务定义

推荐统一使用下面的英文任务文本：

```bash
export TASK_TEXT="Reorient the white rectangular box in the green tray so that the QR code side faces upward and is clearly visible"
```

推荐数据集：

```bash
export DATASET_ID=local/piper_white_box_reorient_qr_up_green_tray_dual_rs_v1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_reorient_qr_up_green_tray_dual_rs_v1
```

推荐先采集 60 条成功轨迹：

```bash
export NUM_EPISODES=60
```

如果盒子上实际是一维条码而不是二维码，必须在开始采集前把任务文本中的 `QR code` 统一改成 `barcode`。同一个数据集内不要混用两种表述。

## 2. 初始状态与成功标准

### 2.1 初始状态

- 白色长方体完整位于绿色托盘内。
- 二维码面不能已经朝上，避免采集大量无动作或近似无动作轨迹。
- 二维码面可以朝下或朝侧面，盒子的平面旋转角度和盘内位置可以适度变化。
- 盒子不能卡在托盘边缘，也不能处于机械臂无法安全夹取的姿态。
- 绿色托盘、相机和机械臂 reset 姿态在整轮采集中保持固定。

建议覆盖以下初始姿态：

1. 二维码面朝左侧。
2. 二维码面朝右侧。
3. 二维码面朝向或背向机械臂。
4. 二维码面朝下，需要重新夹取或翻转。
5. 盒子在托盘中心附近有少量位置和朝向变化。

不同初始状态都必须能够安全完成任务，不要为了增加多样性而加入明显不可解或高碰撞风险的摆放。

### 2.2 标准操作流程

1. 从 home 或固定安全起始位开始。
2. 移动到绿色托盘内的白盒上方。
3. 根据当前姿态夹取、推动或重新抓取白盒。
4. 在绿色托盘内旋转或翻转盒子。
5. 将盒子稳定放回托盘，使二维码面竖直朝上。
6. 完全松开夹爪并后撤，避免遮挡二维码。

不要求每条轨迹使用完全相同的调整方法。可以包含合理的重新夹取、推正和翻转，但不要加入与任务无关的长时间移动。

### 2.3 成功标准

- 白盒最终完整位于绿色托盘内，不压边、不掉出。
- 白盒稳定放置，没有倾倒、悬空或继续滑动。
- 二维码所在面朝上。
- 二维码没有被夹爪或托盘边缘遮挡，并且至少在一个相机视角中清晰可见。
- 机械臂末端已经后撤，不遮挡最终状态。

以下情况直接丢弃并重录：

- 二维码仍朝下或朝侧面。
- 白盒掉出托盘、压在托盘边缘或最终姿态不稳定。
- 夹爪没有完全松开，或者机械臂遮挡二维码。
- 操作中发生明显碰撞、掉落或失控。
- episode 开头二维码已经朝上，整条轨迹几乎没有有效动作。

## 3. 当前硬件映射

仓库与环境：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
```

RealSense：

- `wrist`: `409122274629`, D405
- `front`: `352122272924`, D405

PiPER CAN：

- leader arm / 主臂：`can0`
- follower arm / 从臂：`can1`
- `can0 -> 1-10:1.0`
- `can1 -> 1-4:1.0`

相机数据 key：

- `observation.images.wrist`
- `observation.images.front`

## 4. 采集前检查

关闭可能占用相机的 RealSense Viewer：

```bash
pkill -f realsense-viewer || true
```

激活两个 CAN 口：

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

检查 RealSense：

```bash
rs-enumerate-devices
lerobot-find-cameras realsense
```

正式采集前确认：

- `front` 能看到完整绿色托盘和白盒最终姿态。
- `wrist` 能看清夹取点、盒子侧面和二维码。
- 两个相机均已拧紧，采集过程中不再移动。
- leader 能稳定控制 follower，夹爪开合方向正确。

## 5. 可选 Teleop 检查

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

## 6. 正式采集命令

这是新数据集，第一次采集不要添加 `--resume=true`。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Reorient the white rectangular box in the green tray so that the QR code side faces upward and is clearly visible"
export DATASET_ID=local/piper_white_box_reorient_qr_up_green_tray_dual_rs_v1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_reorient_qr_up_green_tray_dual_rs_v1

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
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK_TEXT" \
  --dataset.num_episodes=100 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false
```

如果确认是在同一数据集上继续追加，且 `TASK_TEXT`、相机位置和场景布局均未改变，才在命令末尾添加：

```bash
  --resume=true
```

## 7. 采集按键

- `Right Arrow`：提前结束当前 episode，并保存。
- `Left Arrow`：提前结束当前 episode，丢弃并重录。
- `Esc`：结束整轮采集，并保存已经完成的 episodes。

普通 `lerobot-record` 默认不写 `episode_success`。本轮建议只保留满足成功标准的轨迹，失败样本直接丢弃重录。

## 8. 采集后检查

生成数据集报告：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-dataset-report \
  --dataset local/piper_white_box_reorient_qr_up_green_tray_dual_rs_v1
```

检查 `meta/info.json`，必须同时包含：

```text
observation.images.wrist
observation.images.front
```

人工检查每条 episode：

1. 第一帧中白盒是否已经位于绿色托盘内。
2. 第一帧二维码是否确实没有朝上。
3. 是否存在明确、连续的有效姿态调整动作。
4. 最后 3-5 秒中二维码是否朝上且清晰可见。
5. 白盒是否稳定留在托盘内，夹爪是否松开并后撤。
6. 相机视角、托盘位置和 reset 姿态是否在所有 episodes 中保持一致。

## 9. 训练前处理

训练前建议裁掉 episode 开头过长的静止或准备段：

1. 计算 action 或 state 的关节运动幅度。
2. 找到第一个明显开始运动的帧 `start_move`。
3. 保留 `start_move` 前 3-5 帧缓冲。
4. 删除更早的静止帧。
5. 重新生成 `frame_index`、`index` 和 `timestamp`，同时保持视频对齐。

处理前后都应生成 review 视频或 contact sheet。不要裁掉机械臂接近盒子、首次接触盒子或判断当前二维码朝向的关键画面。

## 10. 关键注意事项

- 本任务只学习绿盘内姿态调整，不学习从盘外搬入，也不学习搬到右侧区域。
- `TASK_TEXT` 在采集、训练和推理中必须完全一致。
- 第一轮只采成功轨迹，避免把二维码未朝上的失败终点作为监督。
- 初始姿态要有变化，但相机、托盘和 reset 姿态不要变化。
- 不要在同一个 dataset repo 中混入旧任务 A 或任务 B 的 episodes。
- 不要为了动作统一而强制每条轨迹使用同一种翻转方式，最终状态一致更重要。
- 当前固定使用 `robot.port=can1`、`teleop.port=can0`。
- `realsense-viewer` 不能与采集程序同时占用相机。
