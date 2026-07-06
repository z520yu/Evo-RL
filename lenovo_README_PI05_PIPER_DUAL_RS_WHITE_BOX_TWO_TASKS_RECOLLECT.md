# [lenovo] Pi0.5 PiPER Dual RealSense White Box Two-Task Recollection Notes

本文档记录当前这台机器上使用两个 RealSense D405、两个 PiPER CAN 口，重新采集两个白盒任务的数据。

两个任务是连续场景：

1. 任务 A：夹起白色盒子，放到中间绿色长方形托盘里，并调整盒子姿态，使条码面朝上且清晰可见。
2. 任务 B：从绿色托盘里夹起白色盒子，放到右侧成功区域。

这次重采的主要目的：让训练数据匹配当前相机位置、盘子位置、右侧成功区位置和 reset 姿态，减少之前真机推理中抓取整体偏左的问题。

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

当前实测 CAN 适配器：

- `can0 -> 1-10:1.0`
- `can1 -> 1-4:1.0`

当前命令约定：

- leader arm / 主臂: `can0`
- follower arm / 从臂: `can1`
- camera keys: `observation.images.wrist`, `observation.images.front`

注意：主臂是人手拖动的 teleop/leader，从臂是执行动作的 robot/follower。本文件以 `leader=can0`、`follower=can1` 为准；如果 USB 重新插拔，只改 `can_activate.sh` 的 USB 地址，不要把主从臂端口写反。

## 2. 任务定义

### 2.1 任务 A：白盒条码朝上放入绿盘

推荐采集、训练、推理统一使用下面这条英文任务文本：

```bash
export TASK_A_TEXT="Pick up the white box, place it in the center of the green rectangular tray, and orient the box so the barcode side faces upward and is clearly visible"
```

推荐数据集：

```bash
export TASK_A_DATASET_ID=local/piper_white_box_barcode_up_green_tray_dual_rs_v2
export TASK_A_DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_barcode_up_green_tray_dual_rs_v2
```

采集数量：

```bash
export TASK_A_EPISODES=60
```

这个任务更准确地说是一个 `scan-ready placement` 任务：机器人不直接完成扫码动作，而是把盒子摆到扫码器或相机可以读取条码的最终姿态。

不要写成简单的 `Scan the barcode`。当前流程里没有扫码器返回的成功信号，模型真正能学习的是：

1. 识别白色盒子和绿色长方形托盘。
2. 把盒子夹起并放到托盘中心区域。
3. 放置后继续调整盒子姿态。
4. 让条码所在面朝上，并且无遮挡、清晰可见。

### 2.2 任务 B：白盒从绿盘放到右侧成功区

推荐采集、训练、推理统一使用下面这条英文任务文本：

```bash
export TASK_B_TEXT="Pick up the white box from the green tray and place it in the success area on the right"
```

推荐数据集：

```bash
export TASK_B_DATASET_ID=local/piper_white_box_green_tray_to_right_success_area_dual_rs_v2
export TASK_B_DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_green_tray_to_right_success_area_dual_rs_v2
```

采集数量：

```bash
export TASK_B_EPISODES=50
```

任务 B 的初始状态应该接近任务 A 的成功终点：白盒已经在绿色托盘内，条码面朝上或接近朝上，盒子位置在托盘中心附近。

一旦开始采集同一个数据集，不要中途更换 task text，避免 task embedding 和数据动作语义不一致。

## 3. 场景设计和成功标准

桌面放置：

- 一个白色盒子，盒子某一面贴有条码。
- 一个绿色长方形托盘，放在桌面中间作为任务 A 的目标区域，也是任务 B 的起始区域。
- 右侧成功区域固定在绿色托盘右侧，建议用胶带或清晰边界标出来。
- 两个 RealSense 视角保持固定：`front` 看整体桌面、托盘和右侧成功区，`wrist` 看夹爪附近的盒子姿态。

这次采集中不要边采边调整相机。开始采集前先把 `front` 和 `wrist` 都拧紧，绿色托盘和右侧成功区位置也固定下来。

### 3.1 任务 A 标准流程

1. 从 home 或安全起始位开始。
2. 移动到白色盒子上方。
3. 夹住盒子，稳定抬起。
4. 移动到绿色长方形托盘中心上方。
5. 将盒子放入托盘中间区域。
6. 通过夹爪轻推、重新夹取或旋转，调整盒子姿态。
7. 最终松开盒子，使条码面朝上且在相机中清晰可见。
8. 机械臂后撤到不遮挡条码的位置。

任务 A 成功样本标准：

- 盒子完整进入绿色托盘，不压边、不掉出。
- 盒子稳定放置，没有倾倒或明显悬空。
- 条码所在面朝上。
- 条码无遮挡，至少在 `front` 或 `wrist` 视角中清晰可见。
- 机械臂末端后撤后不遮挡条码。

### 3.2 任务 B 标准流程

1. 从 home 或安全起始位开始。
2. 白盒初始放在绿色托盘内，尽量接近任务 A 的成功终点。
3. 移动到绿色托盘内的白盒上方。
4. 夹住盒子，稳定抬起。
5. 移动到右侧成功区域上方。
6. 将盒子放入右侧成功区域内。
7. 松开夹爪并后撤，避免遮挡最终状态。

任务 B 成功样本标准：

- 初始状态中白盒在绿色托盘内，不在托盘外。
- 盒子从托盘中被稳定夹起，中途不掉落。
- 盒子最终完整进入右侧成功区域。
- 盒子不压线、不停在托盘和成功区之间。
- 机械臂末端后撤后不遮挡盒子。

失败样本直接丢弃重录：

- 没夹起盒子或中途掉落。
- 盒子放在目标区域外、压边或明显偏离目标。
- 条码任务里条码朝下、朝侧面，或被夹爪/托盘边缘遮挡。
- 右侧成功区任务里盒子仍留在绿色托盘内cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box from the green tray and place it in the success area on the right"
export DATASET_ID=local/piper_white_box_green_tray_to_right_success_area_dual_rs_v2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_green_tray_to_right_success_area_dual_rs_v2

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
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --resume=true，或只移动到中间过渡区。
- 盒子最终姿态不稳定，松手后翻倒或滑出。

第一轮建议只采成功样本。这个任务的关键不是轨迹漂亮，而是最后状态必须清楚、稳定、可学习。

## 4. 采集前检查

### 4.1 关闭 RealSense Viewer

如果开着 `realsense-viewer`，它会占用 `/dev/video*`，导致 LeRobot 代码打开相机失败。

```bash
pkill -f realsense-viewer || true
```

### 4.2 激活 CAN

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

### 4.3 检查两个 RealSense

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

python -c 'from lerobot.cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box from the green tray and place it in the success area on the right"
export DATASET_ID=local/piper_white_box_green_tray_to_right_success_area_dual_rs_v2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_green_tray_to_right_success_area_dual_rs_v2

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
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --resume=truecameras.realsense import RealSenseCamera, RealSenseCameraConfig; cams={"wrist":"409122274629","front":"352122272924"}; opened=[];
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

## 5. 可选 teleop 检查

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

## 6. 双 RealSense 采集命令

两个任务建议采成两个独立数据集，不要混到同一个 repo 里。这样后续可以分别裁静止段、分别训练和评估，也方便排查任务 A 与任务 B 的失败原因。

### 6.1 任务 A：采集 100 条

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box, place it in the center of the green rectangular tray, and orient the box so the barcode side faces upward and is clearly visible"
export DATASET_ID=local/piper_white_box_barcode_up_green_tray_dual_rs_v2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_barcode_up_green_tray_dual_rs_v2

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
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false
```

### 6.2 任务 B：采集 80 条

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box from the green tray and place it in the success area on the right"
export DATASET_ID=local/piper_white_box_green_tray_to_right_success_area_dual_rs_v2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_white_box_green_tray_to_right_success_area_dual_rs_v2

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
  --dataset.num_episodes=90 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false  \
  --resume=true
```

如果只是先试采一两条，把 episode 数量和时间改短：

```bash
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=45 \
  --dataset.reset_time_s=10 \
  --reset_to_zero_action=true
```

继续往同一个数据集追加数据时，在完整命令末尾保留：

```bash
  --resume=true
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

注意：普通 `lerobot-record` 默认不写 `episode_success`，所以 `s/f` 只在 `lerobot-human-inloop-record` 或显式打开 episode outcome labeling 时有意义。

## 8. 采集后检查数据集

任务 A：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-dataset-report --dataset local/piper_white_box_barcode_up_green_tray_dual_rs_v2
```

任务 B：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

lerobot-dataset-report --dataset local/piper_white_box_green_tray_to_right_success_area_dual_rs_v2
```

检查 `meta/info.json` 中应包含两个视频 key：

```text
observation.images.wrist
observation.images.front
```

如果只有 `observation.images.wrist`，说明采集命令没有真正带上第二个相机，不能直接用于双相机训练。

也建议人工抽查每条 episode 的最后 3-5 秒：

- 任务 A：盒子是否在绿色托盘中心，条码面是否朝上，条码是否无遮挡。
- 任务 B：盒子是否完整放到右侧成功区域，是否离开绿色托盘，夹爪是否已经松开并后撤。
- 两个任务都要检查第一帧：相机视角、绿色托盘位置、白盒初始位置是否稳定。

## 9. 训练前处理建议

这次仍然建议训练前裁掉 episode 开头过长静止段：

1. 计算 action 或 state 的关节运动幅度。
2. 找到第一个明显开始动的帧 `start_move`。
3. 保留 `start_move` 前 3-5 帧缓冲。
4. 删除更早的静止/准备帧。
5. 重新生成 `frame_index`、`index`、`timestamp`，并保持视频对齐。

处理前后都生成 review 视频或 contact sheet。确认没有裁掉接近盒子的关键动作后，再同步远端训练。

## 10. 关键注意事项

- 采集、训练、推理必须使用完全一致的 `TASK_TEXT`。
- 两个相机 key 必须固定为 `wrist` 和 `front`；采集、训练、推理必须一致。
- `front` 相机这次尤其关键，采集中不要再拧动或移动。
- 绿色托盘和右侧成功区的位置不要中途变化。
- 任务 B 的初始状态要像任务 A 的成功终点，不要让任务 B 混入“从任意位置找盒子”的数据。
- 现有旧数据集可以作为参考，但不要和这次 v2 数据直接混在同一个 repo 里。
- `realsense-viewer` 不能和采集同时运行。
- 当前 follower/leader 命令约定：`robot.port=can1`，`teleop.port=can0`。
- 如果 USB 重新插拔后 CAN 映射变化，先用 `/home/lenovo/piper_sdk/piper_sdk/find_all_can_port.sh` 或 `ip -details link show type can` 重新确认，再改 `can_activate.sh` 的第三个参数。
