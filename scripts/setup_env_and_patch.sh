#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/mnt/data3/jiangxirui/conda_envs/internvla_a15}"
SITE_PACKAGES="${ENV_PREFIX}/lib/python3.11/site-packages"
TRANSFORMERS_DIR="${SITE_PACKAGES}/transformers"

if [[ ! -d "${TRANSFORMERS_DIR}" ]]; then
    echo "ERROR: transformers directory not found at ${TRANSFORMERS_DIR}" >&2
    exit 1
fi

echo "Applying InternVLA-A-series transformers_replace patches to ${TRANSFORMERS_DIR}..."
cp -r "${PROJ_ROOT}/src/lerobot/policies/pi0/transformers_replace/models" "${TRANSFORMERS_DIR}/"
cp -r "${PROJ_ROOT}/src/lerobot/policies/pi05/transformers_replace/models" "${TRANSFORMERS_DIR}/"
cp -r "${PROJ_ROOT}/src/lerobot/policies/internvla_a1_5/transformers_replace/models" "${TRANSFORMERS_DIR}/"

echo "Verifying imports in ${ENV_PREFIX}/bin/python..."
"${ENV_PREFIX}/bin/python" -c "
import torch, transformers, lerobot
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('transformers:', transformers.__version__)
print('lerobot:', lerobot.__file__)
"
echo "Done."
