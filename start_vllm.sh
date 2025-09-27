#!/bin/bash
set -euo pipefail

VLLM_ENV_PATH=".vllm_venv"
ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "[vLLM] ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [ -z "${MODEL_PATH:-}" ]; then
    echo "[vLLM] ERROR: MODEL_PATH is not set in .env"
    exit 1
fi

if [ -z "${VLLM_PORT:-}" ]; then
    echo "[vLLM] ERROR: VLLM_PORT is not set in .env"
    exit 1
fi

echo "[vLLM] Activating virtualenv..."
source "$VLLM_ENV_PATH/bin/activate"

export VLLM_CONFIGURE_LOGGING="0"

echo "[vLLM] Starting vLLM server on port $VLLM_PORT with model at $MODEL_PATH..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
    --max-num-batched-tokens "${VLLM_MAX_BATCH_TOKENS:-64000}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-32}"

