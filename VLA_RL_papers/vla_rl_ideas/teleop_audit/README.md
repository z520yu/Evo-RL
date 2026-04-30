# Piper Teleop Audit

这个审计走 Evo-RL/LeRobot 框架：入口在 `src/lerobot/scripts/lerobot_piper_audit.py`，使用
`RobotConfig`、`TeleoperatorConfig`、默认 processors 和 `LeRobotDataset`。这里的目录只放说明和输出日志。

目的：先做一次真实遥操作审计，记录底层到底哪些信号有变化、有稳定性、可能对后续 trust/progress/event idea 有用。

## 记录一次遥操作

先激活 Piper 的 CAN 口。

当前这台 Lenovo 机器在 `lenovo_README_PI05_VALUE_ACP.md` 里记录的是：

- follower / robot: `can0`
- leader / teleop: `can1`
- `can0 -> 1-6.2:1.0`
- `can1 -> 1-6.4:1.0`

注意：旧的 `gemini_PIPER_MULTITASK_DATA_COLLECTION.md` 记录过相反映射：follower=`can1`、leader=`can0`。不要混用两套命令。启动前先用普通 `lerobot-teleoperate` 确认当前接线。

```bash
cd /home/lenovo/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000 1-6.2:1.0
bash can_activate.sh can1 1000000 1-6.4:1.0

cd /home/lenovo/Evo-RL
ip -details link show can0
ip -details link show can1
lerobot-setup-can --mode=test --interfaces=can0,can1
```

如果不确定 CAN 口名字，先看：

```bash
ip link
```

先做一次普通遥操作联通性检查，不录审计：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

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
  --teleop.gravity_comp_tx_ratio='[1,1,1,1,1,1]' \
  --display_data=true
```

确认普通遥操作 OK 后，再跑审计：

```bash
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

cd /home/lenovo/Evo-RL

python -m lerobot.scripts.lerobot_piper_audit \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.speed_ratio=50 \
  --robot.cameras='{
    wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}
  }' \
  --teleop.type=piper_leader \
  --teleop.port=can1 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio='[1,1,1,1,1,1]' \
  --dataset.single_task="open drawer, put object in, close drawer" \
  --dataset.episode_time_s=50 \
  --dataset.fps=20
```

如果暂时不接相机，去掉 `--robot.cameras=...` 即可。

如果摄像头 index 不对，先用 Evo-RL/LeRobot 自带命令找：

```bash
lerobot-find-cameras realsense
```

如果只想记录标量，不保存图片：

```bash
python -m lerobot.scripts.lerobot_piper_audit \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.speed_ratio=50 \
  --teleop.type=piper_leader \
  --teleop.port=can1 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio='[1,1,1,1,1,1]' \
  --dataset.single_task="drawer audit" \
  --dataset.episode_time_s=50 \
  --dataset.fps=20 \
  --dataset.video=false
```

运行时会创建：

```text
VLA_RL_papers/vla_rl_ideas/teleop_audit/runs/<timestamp>/
  meta.json
  audit_steps.jsonl
  analysis_stats.json
  analysis_summary.md
```

同时会创建一个标准 LeRobot dataset，默认在：

```text
VLA_RL_papers/vla_rl_ideas/teleop_audit/datasets/local/piper_audit_<timestamp>/
```

`audit_steps.jsonl` 每一行是一帧，包含：

- `raw_observation`: Piper follower 的关节、夹爪、相机观测摘要
- `raw_teleop_action`: Piper leader 读到的人类遥操作目标
- `sent_action`: follower wrapper 实际发送给机器人 SDK 的目标
- `piper_sdk`: 直接从 Piper SDK 读取的状态快照
- `derived.target_minus_obs`: 目标位置减当前观测位置
- `derived.obs_velocity_per_s`: 相邻帧观测速度估计
- LeRobot dataset 视频：用于后续人工或 CV 标注抽屉/物体 progress

## 先看哪些字段

优先看：

- `derived.target_minus_obs.*`
- `derived.obs_velocity_per_s.*`
- `sent_action.*`
- `raw_observation.gripper.pos`
- `piper_sdk.GetArmLowSpdInfoMsgs`
- `piper_sdk.GetArmHighSpdInfoMsgs`
- 相机帧里的抽屉/物体 progress

这一步不预设 calibrated trust 是否有用，只看真实数据里哪些信号能支撑后续判断。
