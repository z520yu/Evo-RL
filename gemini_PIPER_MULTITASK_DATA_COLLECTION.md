# [gemini] PiPER Multi-Task Data Collection

This note records the exact commands for the current single-arm PiPER setup.

Current fixed mapping:

- follower arm: `can1`
- leader arm: `can0`
- wrist camera: Intel RealSense D405
- camera serial: `409122274629`

Current multi-task dataset plan:

- Task A: put the green block into the dark green bowl
- Task B: put the blue block into the dark green bowl
- episodes: `50 + 50`
- dataset id: `local/piper_multitask_v1`

## 1. Activate Environment

```bash
conda activate evo-rl
cd ~/Evo-RL
```

## 2. Activate CAN Interfaces

```bash
cd ~/piper_demo/piper_ros
bash can_activate.sh can1 1000000 1-2.2:1.0
bash can_activate.sh can0 1000000 1-2.4:1.0
```

## 3. Optional CAN Check

```bash
cd ~/Evo-RL
lerobot-setup-can --mode=test --interfaces=can0,can1
```

Expected result:

- `8/8 motors found` on each CAN interface
- `Total motors found: 16`

## 4. Optional Teleop Check

Use this before formal recording if you want to confirm arm control and wrist camera display.

```bash
cd ~/Evo-RL

lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can0 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --display_data=true
```

## 5. Record Task A

Task text:

`Put the green block into the dark green bowl`

```bash
cd ~/Evo-RL

lerobot-record \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can0 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --dataset.repo_id=local/piper_multitask_v1 \
  --dataset.single_task="Put the green block into the dark green bowl" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --resume=true
```

## 6. Record Task B

Task text:

`Put the blue block into the dark green bowl`

This round appends to the same dataset, so `--resume=true` is required.

```bash
cd ~/Evo-RL

lerobot-record \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can0 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --dataset.repo_id=local/piper_multitask_v1 \
  --dataset.single_task="Put the blue block into the dark green bowl" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --resume=true
```

## 7. Record Task C

Task text:

`Fold the towel in half`

This round appends to the same dataset, so `--resume=true` is required.

```bash
cd ~/Evo-RL

lerobot-record \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port=can0 \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --dataset.repo_id=local/piper_multitask_v1 \
  --dataset.single_task="Fold the towel in half" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --resume=true
```

## 8. Continue Recording Later

If you want to keep adding more data to the same dataset later, use the same command as Task A or Task B, but add:

```bash
--resume=true
```

Only change `--dataset.single_task="..."` to the task you are currently collecting.

## 9. Keyboard Shortcuts During Recording

- `Right Arrow`: end current episode immediately
- `Left Arrow`: discard current episode and re-record it
- `Esc`: stop the whole recording session

## 10. Check the Final Dataset

```bash
cd ~/Evo-RL
lerobot-dataset-report --dataset local/piper_multitask_v1
```

## 11. Check That All Tasks Are Inside the Same Dataset

```bash
cd ~/Evo-RL

/home/gemini/miniconda3/envs/evo-rl/bin/python -c "import pandas as pd; p='/home/gemini/.cache/huggingface/lerobot/local/piper_multitask_v1/meta/tasks.parquet'; print(pd.read_parquet(p).to_string())"
```

Expected outcome:

- three task rows
- one row for the green-block task
- one row for the blue-block task
- one row for the towel-fold task

## 12. Notes

- Do not use `--dataset.push_to_hub=true` unless you have already logged into Hugging Face.
- This setup uses one follower arm, one leader arm, and one wrist camera only.
