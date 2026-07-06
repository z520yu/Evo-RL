# OpenPI 白盒双任务 Eval 命令

这两个任务是连续场景：

1. 白盒条码任务：夹起白色盒子，放到绿色长方形盘中，并调整到条码朝上。
2. 右侧成功区任务：从绿色盘里夹起白色盒子，放到右侧成功区域。

两个任务共用同一套 CAN、相机和真机 client；单任务 eval 时只需要换 OpenPI server 的 checkpoint/config，以及 client 的 `--prompt`。

如果要连续执行“先扫码、再放右侧”，推荐同时启动两个 OpenPI server：任务 A 用 `8000`，任务 B 用 `8001`，再用 two-stage client 在一个真机进程里按两次回车切换阶段。这样相机和 CAN 只打开一次。

## 任务 A：白盒条码朝上

本地 checkpoint：

```bash
/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_barcode_green_tray_v2_lora/official_pi05_piper_white_box_barcode_green_tray_v2_lora_from_base_20260701_171130_gpu34_bs16_officiallr_save5k/25000
```

任务文本：

```text
Pick up the white box, place it in the center of the green rectangular tray, and orient the box so the barcode side faces upward and is clearly visible
```

OpenPI config：

```text
pi05_piper_white_box_barcode_green_tray_v2_lora
```

## 任务 B：白盒从绿色盘放到右侧成功区

本地 checkpoint：

25000：

```bash
/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_green_tray_to_right_success_area_v2_lora/official_pi05_piper_white_box_green_tray_to_right_success_area_v2_lora_from_base_20260630_171539_gpu34_bs16_officiallr_save5k/25000
```

29999：

```bash
/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_green_tray_to_right_success_area_v2_lora/official_pi05_piper_white_box_green_tray_to_right_success_area_v2_lora_from_base_20260630_171539_gpu34_bs16_officiallr_save5k/29999
```

任务文本：

```text
Pick up the white box from the green tray and place it in the success area on the right
```

OpenPI config：

```text
pi05_piper_white_box_green_tray_to_right_success_area_v2_lora
```

## 终端 1：激活 CAN

```bash
cd /home/lenovo/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000 1-10:1.0
bash can_activate.sh can1 1000000 1-4:1.0

cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
lerobot-setup-can --mode=test --interfaces=can0,can1
```

## 终端 2：启动 OpenPI Policy Server

每次只启动一个任务的 server。切换任务时先停掉当前 server，再启动另一个任务的命令。

第一次启动会 JAX 编译，可能慢几十秒。

启动前建议先确认 JAX 能看到 GPU。如果这里只看到 `CpuDevice`，policy 推理会退到 CPU，单次请求可能变成 10 秒级，真机会明显卡顿。

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

nvidia-smi
./.venv/bin/python -c "import jax; print(jax.devices())"
```

### 启动任务 A：白盒条码朝上

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

export CKPT=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_barcode_green_tray_v2_lora/official_pi05_piper_white_box_barcode_green_tray_v2_lora_from_base_20260701_171130_gpu34_bs16_officiallr_save5k/25000
export TASK_TEXT="Pick up the white box, place it in the center of the green rectangular tray, and orient the box so the barcode side faces upward and is clearly visible"

./.venv/bin/python scripts/serve_piper_white_box_barcode_policy.py \
  --checkpoint-dir "$CKPT" \
  --config pi05_piper_white_box_barcode_green_tray_v2_lora \
  --prompt "$TASK_TEXT" \
  --port 8000
```

### 启动任务 B：白盒放到右侧成功区

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

export CKPT_25000=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_green_tray_to_right_success_area_v2_lora/official_pi05_piper_white_box_green_tray_to_right_success_area_v2_lora_from_base_20260630_171539_gpu34_bs16_officiallr_save5k/25000
export CKPT_29999=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_green_tray_to_right_success_area_v2_lora/official_pi05_piper_white_box_green_tray_to_right_success_area_v2_lora_from_base_20260630_171539_gpu34_bs16_officiallr_save5k/29999
export CKPT=$CKPT_29999
export TASK_TEXT="Pick up the white box from the green tray and place it in the success area on the right"

./.venv/bin/python scripts/serve_piper_white_box_barcode_policy.py \
  --checkpoint-dir "$CKPT" \
  --config pi05_piper_white_box_green_tray_to_right_success_area_v2_lora \
  --prompt "$TASK_TEXT" \
  --port 8000
```

如果 8000 端口被占用，把 server 的 `--port` 和 client 的 `--port` 改成同一个新端口。

### 两阶段模式：同时启动两个 Server

两阶段回车切换需要两个 policy server 同时常驻。任务 A server 仍用 `8000`，任务 B server 改用 `8001`。两个 server 只占 GPU 显存，不会占相机或 CAN。

终端 2A 启动任务 A：

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

export CKPT=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_barcode_green_tray_v2_lora/official_pi05_piper_white_box_barcode_green_tray_v2_lora_from_base_20260701_171130_gpu34_bs16_officiallr_save5k/25000
export TASK_TEXT="Pick up the white box, place it in the center of the green rectangular tray, and orient the box so the barcode side faces upward and is clearly visible"

./.venv/bin/python scripts/serve_piper_white_box_barcode_policy.py \
  --checkpoint-dir "$CKPT" \
  --config pi05_piper_white_box_barcode_green_tray_v2_lora \
  --prompt "$TASK_TEXT" \
  --port 8000
```

终端 2B 启动任务 B：

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

export CKPT_25000=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_green_tray_to_right_success_area_v2_lora/official_pi05_piper_white_box_green_tray_to_right_success_area_v2_lora_from_base_20260630_171539_gpu34_bs16_officiallr_save5k/25000
export CKPT_29999=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_white_box_green_tray_to_right_success_area_v2_lora/official_pi05_piper_white_box_green_tray_to_right_success_area_v2_lora_from_base_20260630_171539_gpu34_bs16_officiallr_save5k/29999
export CKPT=$CKPT_29999
export TASK_TEXT="Pick up the white box from the green tray and place it in the success area on the right"

./.venv/bin/python scripts/serve_piper_white_box_barcode_policy.py \
  --checkpoint-dir "$CKPT" \
  --config pi05_piper_white_box_green_tray_to_right_success_area_v2_lora \
  --prompt "$TASK_TEXT" \
  --port 8001
```

两个 server 第一次启动后都要完成 JAX 编译。可以先跑下面的 two-stage dry-run 预热。

## 终端 3：先 Dry Run

这个命令只连 policy server，不下发动作给机械臂；会保存首帧输入图，优先用来确认相机图片确实传进去了。

### Dry Run：任务 A

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box, place it in the center of the green rectangular tray, and orient the box so the barcode side faces upward and is clearly visible"

python scripts/openpi_piper_eval_client.py \
  --host localhost \
  --port 8000 \
  --robot-port can1 \
  --num-episodes 1 \
  --episode-time-s 10 \
  --reset-time-s 0 \
  --dry-run \
  --prompt "$TASK_TEXT" \
  --log-policy-io \
  --debug-image-dir outputs/openpi_white_box_barcode_debug_images
```

### Dry Run：任务 B

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box from the green tray and place it in the success area on the right"

python scripts/openpi_piper_eval_client.py \
  --host localhost \
  --port 8000 \
  --robot-port can1 \
  --num-episodes 1 \
  --episode-time-s 10 \
  --reset-time-s 0 \
  --dry-run \
  --prompt "$TASK_TEXT" \
  --log-policy-io \
  --debug-image-dir outputs/openpi_white_box_right_success_debug_images
```

### Dry Run：两阶段回车切换

先确认任务 A server 在 `8000`、任务 B server 在 `8001`。这个命令只连 policy，不给机械臂下发动作；运行时会按两次回车：第一次跑任务 A dry-run，第二次跑任务 B dry-run。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

python scripts/openpi_piper_two_stage_eval_client.py \
  --host localhost \
  --port-a 8000 \
  --port-b 8001 \
  --robot-port can1 \
  --num-episodes 1 \
  --stage-a-time-s 10 \
  --stage-b-time-s 10 \
  --reset-time-s 0 \
  --dry-run \
  --query-every 50 \
  --log-policy-io \
  --debug-image-dir outputs/openpi_two_stage_debug_images
```

## 终端 3：真机 Eval

两个任务都直接使用 chunk 第 0 步动作，不要额外设置 `--action-start-index`。

正常 GPU 推理下，policy 请求通常很快，不需要异步预取。这里保留同步请求。当前模型的释放/张开夹爪动作经常出现在 chunk 后半段，所以推荐 `--query-every 50`，完整执行一个 50 步 action chunk；默认 `--control-hz 30` 下约 1.67 秒换一次。

真机 eval 的 policy action 默认原样发送，不做关节/夹爪限幅。`--max-joint-delta-per-step` 和 `--max-gripper-delta-per-step` 只有显式设置为正数时才会启用；下面命令不要加这两个参数。`--max-reset-*` 只影响 episode 开始前的慢速 reset。

第一次真机建议先把 `--num-episodes` 改成 `1` 观察完整轨迹，确认安全后再跑 5 条。

### 真机 Eval：任务 A

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box, place it in the center of the green rectangular tray, and orient the box so the barcode side faces upward and is clearly visible"

python scripts/openpi_piper_eval_client.py \
  --host localhost \
  --port 8000 \
  --robot-port can1 \
  --num-episodes 5 \
  --episode-time-s 100 \
  --reset-time-s 4 \
  --reset-gripper-pos 60 \
  --max-reset-joint-delta-per-step 5 \
  --max-reset-gripper-delta-per-step 15 \
  --query-every 50 \
  --prompt "$TASK_TEXT" \
  --log-policy-io
```

### 真机 Eval：任务 B

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export TASK_TEXT="Pick up the white box from the green tray and place it in the success area on the right"

python scripts/openpi_piper_eval_client.py \
  --host localhost \
  --port 8000 \
  --robot-port can1 \
  --num-episodes 5 \
  --episode-time-s 100 \
  --reset-time-s 4 \
  --reset-gripper-pos 60 \
  --max-reset-joint-delta-per-step 5 \
  --max-reset-gripper-delta-per-step 15 \
  --query-every 50 \
  --prompt "$TASK_TEXT" \
  --log-policy-io
```

### 真机 Eval：两阶段回车切换

运行前确认任务 A server 在 `8000`，任务 B server 在 `8001`。这个 client 只打开一次相机和 CAN：

1. 第一次回车：执行任务 A，把白盒放到绿色盘中并条码朝上。
2. 任务 A 跑完后暂停；扫码或人工确认。
3. 第二次回车：不 reset，直接执行任务 B，把白盒从绿色盘放到右侧成功区。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

python scripts/openpi_piper_two_stage_eval_client.py \
  --host localhost \
  --port-a 8000 \
  --port-b 8001 \
  --robot-port can1 \
  --num-episodes 1 \
  --stage-a-time-s 100 \
  --stage-b-time-s 100 \
  --reset-time-s 4 \
  --reset-gripper-pos 60 \
  --max-reset-joint-delta-per-step 5 \
  --max-reset-gripper-delta-per-step 15 \
  --query-every 50 \
  --log-policy-io
```

如果只想减少终端日志，可以去掉 `--log-policy-io`；它只影响诊断日志，不改变 policy 输出。日志里的 `rtt` 可以用来确认 policy 是否走 GPU，正常应该远小于 1 秒；如果接近 10 秒，优先检查 `nvidia-smi` 和 `jax.devices()`。

## 只执行慢速 Reset

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

python scripts/openpi_piper_eval_client.py \
  --robot-port can1 \
  --reset-only \
  --reset-time-s 4 \
  --reset-gripper-pos 60 \
  --max-reset-joint-delta-per-step 5 \
  --max-reset-gripper-delta-per-step 15
```
