#!/usr/bin/env bash
# Launch Gemma 4 26B-A4B-IT (FP8) via SGLang on the GB10.
set -euo pipefail

echo "==> Gemma 4 26B-A4B-IT (FP8) + SGLang on NVIDIA GB10 (DGX Spark)"
echo "    model=${MODEL_ID}  quant=${QUANTIZATION}  max-concurrent=${MAX_RUNNING_REQUESTS}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "!! HF_TOKEN is not set. Gemma 4 is gated — accept the license at" >&2
  echo "   https://huggingface.co/${MODEL_ID} and pass --env HF_TOKEN=hf_xxx" >&2
  exit 1
fi
export HF_TOKEN

# Optional prefetch: downloads weights into the mounted HF_HOME volume so the
# download is a visible, cacheable step separate from server startup.
if [[ "${PREFETCH:-1}" == "1" ]]; then
  echo "==> Downloading ${MODEL_ID} into ${HF_HOME} (cached on the mounted volume)"
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')" || \
    echo "   (prefetch skipped/failed; SGLang will download on startup)"
fi

echo "==> Launching SGLang server on ${HOST}:${PORT}"
exec python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --quantization "${QUANTIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --context-length "${CONTEXT_LEN}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --tp-size 1 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --attention-backend triton \
  ${EXTRA_ARGS}
