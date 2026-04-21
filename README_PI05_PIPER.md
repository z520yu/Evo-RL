# Pi0.5 PiPER Training and Two-Machine Inference Notes

本文档记录当前机器上已经跑通的 Pi0.5 + PiPER 微调命令，以及下一步把推理拆成两台电脑运行的启动方式。

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
/home/lenovo/Evo-RL/piper_multitask_v1
```

数据集概要：

- robot type: `piper_follower`
- fps: `30`
- episodes: `60`
- frames: `25117`
- observation state: `7` 维
- action: `7` 维
- camera key: `observation.images.wrist`
- 已确认任务文本：
  - `Put the green block into the dark green bowl`
  - `Put the blue block into the dark green bowl`

最终训练输出：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219
```

最终推理 checkpoint：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model
```

推理只需要 `pretrained_model` 目录。`training_state` 是恢复训练用的，不需要复制到推理机器。

## 2. 已跑训练命令

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

RUN=pi05_piper_v1_bs32_30k_0416_181219

TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=piper_multitask_v1 \
  --dataset.root=/home/lenovo/Evo-RL/piper_multitask_v1 \
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

- `TOKENIZERS_PARALLELISM=false` 用来消除 `huggingface/tokenizers: The current process just got forked...` 警告。
- `--dataset.video_backend=pyav` 是当前数据集已验证可用的后端；`torchcodec` 在本机缺 FFmpeg 动态库。
- `--policy.train_expert_only=true` 只训练动作专家，本次训练日志显示可训练参数约 `693M`，总参数约 `4B`。
- `--policy.push_to_hub=false` 是必须的，否则不传 `policy.repo_id` 会报错。

## 3. 推理总体方案

两台电脑建议按 async gRPC 推理拆分：

- 服务端电脑：有 GPU，加载 Pi0.5 checkpoint，运行 `lerobot.async_inference.policy_server`。
- 客户端电脑：接机械臂、CAN、相机，运行 `lerobot.async_inference.robot_client`，把观测发给服务端并接收动作。

Evo-RL 当前代码已经支持：

- policy: `pi05`
- robot: `piper_follower`
- async entrypoints:
  - `python -m lerobot.async_inference.policy_server`
  - `python -m lerobot.async_inference.robot_client`

当前服务端机器的 `evo-rl` 环境已安装 async 额外依赖。新机器正式跑前，服务端和客户端都建议执行：

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl
pip install -e ".[async]"
```

这会安装 async 所需的 `grpcio` 和 `matplotlib`。客户端不需要 CUDA，但为了少踩坑，建议两台电脑都用同一份 Evo-RL 代码和同一类 conda 环境；客户端可以保留 CPU 版或 CUDA 版 torch，推理计算实际在服务端执行。

当前服务端机器已确认：

- `grpcio` 可 import
- `matplotlib` 可 import
- `protobuf==6.31.1`

## 4. 服务端启动

服务端机器需要有最终 checkpoint。若 checkpoint 不在服务端，复制 `pretrained_model` 目录即可，例如：

```bash
rsync -av --progress \
  /home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model/ \
  <SERVER_USER>@<SERVER_IP>:/home/lenovo/Evo-RL/checkpoints/pi05_piper_v1_30k/
```

在服务端启动：

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=2
```

当前这台电脑已验证可用的完整启动命令：

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

/home/lenovo/miniconda3/envs/evo-rl/bin/python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=2
```

本机短时验证日志应出现：

```text
PolicyServer started on 0.0.0.0:8080
```

客户端连接当前这台服务端时可用：

```bash
--server_address=192.168.31.123:8080
```

如果客户端走 ZeroTier 网络，则用：

```bash
--server_address=10.249.101.53:8080
```

关键点：

- `--host=0.0.0.0` 表示允许局域网/异机客户端连接。
- 服务端刚启动时不会加载模型；模型路径、policy 类型和 device 会由客户端第一次握手发送。
- 客户端传的 `--pretrained_name_or_path` 是服务端本机路径，不是客户端路径。
- 防火墙需要放行 TCP `8080`。

可在客户端先测网络：

```bash
python -c "import socket; socket.create_connection(('<SERVER_IP>', 8080), timeout=3); print('ok')"
```

## 5. 客户端准备

客户端连接机械臂和 wrist 相机。先确认 CAN 和相机。

CAN：

```bash
lerobot-setup-can --mode=setup --interfaces=can0
lerobot-setup-can --mode=test --interfaces=can0
```

相机：

```bash
lerobot-find-cameras opencv
ls -l /dev/v4l/by-path/
```

因为训练数据只有一个相机 key：`wrist`，推理时客户端也必须用同名 key：

```bash
--robot.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}'
```

先确认普通遥操作还能正常控制 PiPER：

```bash
lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --teleop.type=piper_leader \
  --teleop.port=<LEADER_CAN_PORT> \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false
```

如果没有 leader 或只想先测 follower 连接，需要用项目里已有的单臂测试/遥操作方式确认机械臂能读状态、能安全下发动作后再跑 policy。

## 6. 客户端启动推理

把 `<SERVER_IP>`、`<WRIST_CAM_PATH>` 和服务端 checkpoint 路径替换成真实值：

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

python -m lerobot.async_inference.robot_client \
  --server_address=<SERVER_IP>:8080 \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
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

如果服务端 checkpoint 放在复制后的路径，比如 `/home/lenovo/Evo-RL/checkpoints/pi05_piper_v1_30k`，则客户端命令改成：

```bash
--pretrained_name_or_path=/home/lenovo/Evo-RL/checkpoints/pi05_piper_v1_30k
```

可替换任务文本：

```bash
--task="Put the blue block into the dark green bowl"
```

## 7. 参数调试建议

- `actions_per_chunk=50`：Pi0.5 常用 chunk 长度。网络或服务端慢时可先降到 `20` 或 `10`，更及时但请求更频繁。
- `chunk_size_threshold=0.5`：动作队列剩余到一半时请求新 chunk。想更频繁重规划可试 `0.6` 到 `0.7`；想降低服务端压力可试 `0.3` 到 `0.4`。
- `aggregate_fn_name=weighted_average`：默认更平滑。若需要更激进响应，可试 `latest_only`，但机械臂动作可能更抖。
- `fps=30`：保持和训练数据一致，不建议一开始改。
- `policy_device=cuda`：这是服务端推理 device；客户端即使没有 GPU 也可以保持 `client_device=cpu`。

## 8. 安全顺序

第一次真机推理建议按这个顺序：

1. 服务端先启动，确认监听 `0.0.0.0:8080`。
2. 客户端确认 `can0`、相机、普通 teleoperate 都正常。
3. 机械臂远离人、桌面边缘和易碰撞物，准备急停。
4. 用训练时出现过的任务文本启动，例如 `Put the green block into the dark green bowl`。
5. 先观察模型加载和动作队列日志，再让机械臂执行完整任务。

如果只在一台同时接 GPU 和机械臂的电脑上跑，也可以不用 async，改用同步 policy 方式。但两台电脑分工时，优先使用本文的 `policy_server` + `robot_client`。
