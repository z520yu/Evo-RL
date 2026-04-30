# [gemini] Pi0.5 PiPER 客户端推理命令

本文档只记录这台客户端机器需要执行的关键命令。服务端已经在另一台机器运行。

## 当前配置

- 客户端仓库：`/home/gemini/Evo-RL`
- Conda 环境：`evo-rl`
- follower 机械臂：`can1`
- wrist 相机：RealSense D405
- 相机序列号：`409122274629`
- 服务端地址：`192.168.31.123:8080`
- 当前 Wi-Fi 需连接到 5GHz AP：`50:92:6A:28:9F:A7`
- 服务端 checkpoint：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model
```

注意：`--pretrained_name_or_path` 是服务端机器上的路径，不是客户端路径。

## 1. 进入环境

```bash
cd /home/gemini/Evo-RL
conda activate evo-rl
```

如果这台客户端还没装 async 依赖，先执行一次：

```bash
pip install -e ".[async]"
```

## 2. 检查服务端是否连通

先确认当前连的是 A7：

```bash
iwconfig wlo1 | grep "Access Point"
```

期望看到：

```text
Access Point: 50:92:6A:28:9F:A7
```

```bash
python - <<'PY'
import socket
socket.create_connection(("192.168.31.123", 8080), timeout=3).close()
print("server reachable")
PY
```

## 3. 激活并检查 CAN

```bash
cd /home/gemini/piper_demo/piper_ros
bash can_activate.sh can1 1000000 1-2.2:1.0
bash can_activate.sh can0 1000000 1-2.4:1.0
```

```bash
cd /home/gemini/Evo-RL
conda activate evo-rl
lerobot-setup-can --mode=test --interfaces=can1
```

期望看到：

```text
Summary: 8/8 motors found
Total motors found: 8
```

## 4. 检查相机

```bash
cd /home/gemini/Evo-RL
conda activate evo-rl
lerobot-find-cameras realsense
```

期望看到：

```text
Name: Intel RealSense D405
Id: 409122274629
```

## 5. 启动绿色任务推理

启动前确认机械臂周围安全，急停可用。

```bash
cd /home/gemini/Evo-RL
conda activate evo-rl

MPLCONFIGDIR=/tmp/matplotlib python -m lerobot.async_inference.robot_client \
  --server_address=192.168.31.123:8080 \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --task="Put the green block into the dark green bowl" \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=false \
  --fps=30
```

## 6. 启动蓝色任务推理

只改任务文本：

```bash
cd /home/gemini/Evo-RL
conda activate evo-rl

MPLCONFIGDIR=/tmp/matplotlib python -m lerobot.async_inference.robot_client \
  --server_address=192.168.31.123:8080 \
  --robot.type=piper_follower \
  --robot.port=can1 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --task="Put the blue block into the dark green bowl" \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=false \
  --fps=30
```

## 7. 停止

```bash
Ctrl+C
```

如果服务端报 checkpoint 不存在，说明 `--pretrained_name_or_path` 对服务端机器不正确，需要在服务端修路径。
