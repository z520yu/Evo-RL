# OpenPI 莲子 29999 Eval 三终端命令

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

## 终端 2：启动 OpenPI Server

```bash
cd /home/lenovo/Evo-RL/openpi-lotus-v2-workspace
source ./setup_env_local.sh
export CKPT=/home/lenovo/Evo-RL/openpi-lotus-v2-workspace/checkpoints/pi05_piper_lotus_seed_v2_lora/official_pi05_piper_lotus_seed_v2_lora_from_base_20260615_164629/29999
./.venv/bin/python scripts/serve_piper_lotus_seed_policy.py --checkpoint-dir "$CKPT" --port 8000
```

## 终端 3：启动 OpenPI Client

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
