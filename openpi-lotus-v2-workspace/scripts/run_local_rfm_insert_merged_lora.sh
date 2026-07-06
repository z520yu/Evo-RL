#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lenovo/Evo-RL/openpi-lotus-v2-workspace"
cd "$ROOT"

source ./setup_env_local.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

EXP_NAME="${EXP_NAME:-official_pi05_insert_phase1_expert_success_phase2_from_expert_hil29999_bs32_30k_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="$ROOT/logs/${EXP_NAME}.log"
mkdir -p "$ROOT/logs"

printf 'EXP_NAME=%s\nLOG_PATH=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$EXP_NAME" "$LOG_PATH" "$CUDA_VISIBLE_DEVICES"

./.venv/bin/python scripts/train.py \
  pi05_insert-mouse-battery_phase1_expert_success_phase2_lora \
  --exp-name "$EXP_NAME" \
  --overwrite \
  --no-wandb-enabled \
  2>&1 | tee "$LOG_PATH"
