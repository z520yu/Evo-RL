#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lenovo/Evo-RL/openpi-lotus-v2-workspace"
CONFIG="pi05_piper_lotus_seed_v2_lora"
EXP="official_pi05_piper_lotus_seed_v2_lora_from_base_20260615_164629"
STEP="${1:-29999}"

REMOTE_EXP="${REMOTE_EXP:-/public/robot/zhangyu/checkpoints/rfm_runs_local_backup_20260530/lotus_seed_v2_lora_from_base_30k/checkpoints/${CONFIG}/${EXP}}"
LOCAL_EXP="${LOCAL_EXP:-${ROOT}/checkpoints/${CONFIG}/${EXP}}"

mkdir -p "${LOCAL_EXP}"
mkdir -p "${LOCAL_EXP}/${STEP}"

for item in params assets; do
  rsync -avh --progress "L20_5:${REMOTE_EXP}/${STEP}/${item}/" "${LOCAL_EXP}/${STEP}/${item}/"
done

test -d "${LOCAL_EXP}/${STEP}/params"
test -d "${LOCAL_EXP}/${STEP}/assets"

echo "Downloaded checkpoint:"
echo "${LOCAL_EXP}/${STEP}"
du -sh "${LOCAL_EXP}/${STEP}/params" "${LOCAL_EXP}/${STEP}/assets"
echo
echo "Serve it with:"
echo "cd ${ROOT}"
echo "source ./setup_env_local.sh"
echo "OPENPI_PYTHON=\${OPENPI_PYTHON:-${ROOT}/.venv/bin/python}"
echo "\${OPENPI_PYTHON} scripts/serve_piper_lotus_seed_policy.py --checkpoint-dir ${LOCAL_EXP}/${STEP} --port 8000"
