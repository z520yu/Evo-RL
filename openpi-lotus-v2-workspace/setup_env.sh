#!/bin/bash

export RFM_ROOT="/home/zhangyu/datasets/posttraining_rfm_rss2026"
export PATH="${RFM_ROOT}/tools/bin:${PATH}"

export OPENPI_DATA_HOME="/home/zhangyu/.cache/openpi_rfm"
export HF_LEROBOT_HOME="${RFM_ROOT}/datasets/Challenge-phase1-dataset"
export HF_HOME="/tmp/zhangyu_hf_rfm"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export JAX_COMPILATION_CACHE_DIR="/tmp/zhangyu_jax_cache"

# Keep uv locks and the Python environment off the NFS dataset mount.
export UV_CACHE_DIR="/home/zhangyu/uv-rfm/cache"
export UV_PYTHON_INSTALL_DIR="/home/zhangyu/uv-rfm/python"
export UV_PROJECT_ENVIRONMENT="/home/zhangyu/venvs/openpi-baseline"

echo "Environment variables set:"
echo "RFM_ROOT: ${RFM_ROOT}"
echo "OPENPI_DATA_HOME: ${OPENPI_DATA_HOME}"
echo "HF_LEROBOT_HOME: ${HF_LEROBOT_HOME}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "JAX_COMPILATION_CACHE_DIR: ${JAX_COMPILATION_CACHE_DIR}"
echo "UV_CACHE_DIR: ${UV_CACHE_DIR}"
echo "UV_PROJECT_ENVIRONMENT: ${UV_PROJECT_ENVIRONMENT}"
