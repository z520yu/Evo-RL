# 本地 OpenPI 莲子 LoRA 推理

这个流程把两个环境隔离开：

- OpenPI server 环境：Python 3.11 + JAX，只负责加载 LoRA checkpoint 并提供 websocket policy server。
- Evo-RL 机器人环境：现有 `evo-rl` conda 环境，只负责 RealSense、CAN、Piper 发动作。

不要把 JAX/OpenPI 主环境装进 `evo-rl` 里。

## 1. 安装本地 OpenPI server 环境

第一次在本机配置 OpenPI 时执行：

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

# 当前工作区已经把 uv 安装在 .local/bin；source setup_env_local.sh 后可以直接用。
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

机器人侧只需要轻量 client 依赖，装到现有 `evo-rl` 环境：

```bash
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
pip install -e /home/lenovo/Evo-RL/openpi-lotus-v2-workspace/packages/openpi-client
```

## 2. 下载远端 checkpoint

等远端某个 checkpoint 已经同步到 public 盘后，例如下载最终的 `29999`：

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
bash scripts/download_lotus_seed_checkpoint.sh 29999
```

当前已经跑通 `29999` 的下载；后面要换成 `15000`、`20000` 或 `25000`，只需要把最后这个数字改掉。

下载后的默认目录：

```bash
/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_lotus_seed_v2_lora/official_pi05_piper_lotus_seed_v2_lora_from_base_20260615_164629/29999
```

## 3. 启动 OpenPI policy server

开一个终端：

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh

export CKPT=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_lotus_seed_v2_lora/official_pi05_piper_lotus_seed_v2_lora_from_base_20260615_164629/29999
./.venv/bin/python scripts/serve_piper_lotus_seed_policy.py \
  --checkpoint-dir "$CKPT" \
  --port 8000
```

第一次启动会编译 JAX，可能会慢一些；后面会走本地 `JAX_COMPILATION_CACHE_DIR`。

## 4. 启动 Piper 真机 eval client

另开一个终端，使用现有 Evo-RL 环境：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

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

默认相机序列号：

```bash
front=352122272924
wrist=409122274629
```

如果 CAN 口变了，只改 `--robot-port`。如果 policy server 和机器人不在同一台机器，改 `--host` 为 server IP。

## 5. 安全参数

客户端默认会限制每个控制步动作跳变：

```bash
--max-joint-delta-per-step 8
--max-gripper-delta-per-step 15
```

如果刚开始 eval，建议保留这个限制。模型如果输出异常大跳变，脚本会按当前关节位置做限幅，不会直接把整段 action 原样打给机械臂。

## 6. 停止和 reset

这个 OpenPI eval client 没有 `lerobot-record` 那种交互按钮。它按固定 episode 数运行：

```bash
--num-episodes 5
--episode-time-s 40
```

默认每个 episode 开始前会等待你按 ENTER；按 ENTER 后先执行 reset，再开始推理。reset 会先读取当前关节位置，然后分步插值回 0 位；夹爪到 `--reset-gripper-pos`：

```bash
--reset-time-s 12
--reset-gripper-pos 60
--max-reset-joint-delta-per-step 1
--max-reset-gripper-delta-per-step 3
```

中途停止当前 eval：在运行 client 的终端按 `Ctrl-C`。policy server 可以继续开着，下一次只需要重新运行 client。

只执行一次 reset，不推理：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

python scripts/openpi_piper_eval_client.py \
  --robot-port can1 \
  --reset-only \
  --reset-time-s 12 \
  --reset-gripper-pos 60 \
  --max-reset-joint-delta-per-step 1 \
  --max-reset-gripper-delta-per-step 3
```

如果 CAN 口是 `can0`，把 `--robot-port can1` 改成 `can0`。
