#!/usr/bin/env bash
_OPENPI_LOTUS_SHELL_OPTS="$(set +o)"
set -euo pipefail

export OPENPI_LOTUS_ROOT="/home/lenovo/Evo-RL/openpi-lotus-v2-workspace"
export OPENPI_DATA_HOME="${OPENPI_LOTUS_ROOT}/.openpi_cache"
export HF_HOME="${OPENPI_LOTUS_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export JAX_COMPILATION_CACHE_DIR="${OPENPI_LOTUS_ROOT}/.jax_cache"
export UV_CACHE_DIR="${OPENPI_LOTUS_ROOT}/.uv_cache"
export UV_PYTHON_INSTALL_DIR="${OPENPI_LOTUS_ROOT}/.uv_python"
export UV_PROJECT_ENVIRONMENT="${OPENPI_LOTUS_ROOT}/.venv"
export PATH="${OPENPI_LOTUS_ROOT}/.local/bin:${PATH}"
export PYTHONPATH="${OPENPI_LOTUS_ROOT}/src:${OPENPI_LOTUS_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

# Avoid JAX preallocating nearly all 5090 VRAM during policy serving.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p "${OPENPI_DATA_HOME}" "${HF_DATASETS_CACHE}" "${JAX_COMPILATION_CACHE_DIR}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

echo "OPENPI_LOTUS_ROOT=${OPENPI_LOTUS_ROOT}"
echo "OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"

eval "${_OPENPI_LOTUS_SHELL_OPTS}"
unset _OPENPI_LOTUS_SHELL_OPTS
