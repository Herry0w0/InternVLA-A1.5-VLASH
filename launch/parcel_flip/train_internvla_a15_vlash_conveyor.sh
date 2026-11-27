#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# InternVLA-A1.5 + VLASH 动态传送带翻包训练启动脚本（3 相机）
# ============================================================================
# 默认使用 4 卡 DDP（4 × BS=8 × grad_accum=4 = 全局 BS=128，对齐官方 Table 1 Posttrain，~15.3 epochs）：
#   bash launch/parcel_flip/train_internvla_a15_vlash_conveyor.sh
# 如需指定其他空闲卡或覆盖参数：
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
#     bash launch/parcel_flip/train_internvla_a15_vlash_conveyor.sh --steps=5 --wandb.mode=disabled
# SwanLab 云端同步请提前在终端导出环境变量：
#   export SWANLAB_API_KEY="<your-api-key>"
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_ROOT}"

ENV_PREFIX="${ENV_PREFIX:-/mnt/data3/jiangxirui/conda_envs/internvla_a15}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
ACCELERATE_BIN="${ENV_PREFIX}/bin/accelerate"

if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "ERROR: accelerate not found at ${ACCELERATE_BIN}" >&2
    exit 1
fi

export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/data3/jiangxirui/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 从外部环境变量读取 SWANLAB_API_KEY，严禁在脚本中硬编码明文密钥
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    echo "[INFO] SWANLAB_API_KEY not set in env; SwanLab will use cached login or local mode."
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
YAML_PATH="${YAML_PATH:-${PROJ_ROOT}/configs/parcel_flip/internvla_a15_vlash_conveyor_delta_3cam.yaml}"

DATA_ROOT="${DATA_ROOT:-/mnt/data3/jiangxirui/dynamic_rigid_parcel_flip_internvla_lerobot_v3_rel30}"
if [[ ! -f "${DATA_ROOT}/meta/info.json" ]]; then
    echo "ERROR: Dataset not ready at ${DATA_ROOT} (missing meta/info.json)" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data3/jiangxirui/train_outputs}"
mkdir -p "${OUTPUT_ROOT}"
RUN_NAME="$(date +%Y%m%d_%H%M%S)_${NUM_PROCESSES}gpu_internvla_a15_vlash_conveyor"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

ACCEL_ARGS=(
    --num_processes="${NUM_PROCESSES}"
    --mixed_precision="${MIXED_PRECISION}"
)
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    ACCEL_ARGS+=(
        --multi_gpu
        --num_machines=1
        --machine_rank=0
        --main_process_ip="${MASTER_ADDR:-127.0.0.1}"
        --main_process_port="${MASTER_PORT:-29515}"
    )
fi

echo "=== InternVLA-A1.5 + VLASH Training (3-Cam) ==="
echo "PROJ_ROOT            = ${PROJ_ROOT}"
echo "PYTHON_BIN           = ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES} (NUM_PROCESSES=${NUM_PROCESSES})"
echo "DATA_ROOT            = ${DATA_ROOT}"
echo "OUTPUT_DIR           = ${OUTPUT_DIR}"

exec "${ACCELERATE_BIN}" launch "${ACCEL_ARGS[@]}" \
    "${PROJ_ROOT}/src/lerobot/scripts/lerobot_train.py" \
    --yaml_path="${YAML_PATH}" \
    --dataset.root="${DATA_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    "$@"
