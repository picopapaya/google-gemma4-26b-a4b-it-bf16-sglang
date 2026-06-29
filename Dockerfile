# Gemma 4 26B-A4B-IT served by SGLang with FP8 on-the-fly quantization, on the NVIDIA GB10 (DGX Spark).
#
# Loads google/gemma-4-26B-A4B-it (BF16 weights) and lets SGLang quantize to FP8
# at load time via its built-in --quantization fp8 path.
#
# Compare with ../gemma4-26b-a4b which uses RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
# (pre-quantized, per-channel scales, compressed-tensors format).
#
# Model architecture — Gemma 4 27B is a Mixture-of-Experts model:
#   - 26B total parameters, ~3.8B active per token ("26B-A4B" = 26B total, 4B active)
#   - 128 experts per MoE layer; router activates 2 per token
#   - Hybrid: attention layers are dense; FFN layers are MoE
#   - All 26B params must be in VRAM for routing, but compute cost ≈ a 4B dense model
#
# Quantization — why FP8 over NVFP4 on SM121a (GB10):
#   - NVFP4 has NO native FP4 GEMM kernel on SM12x; it falls back to Marlin, which
#     dequantizes FP4 → BF16 inside the kernel, losing the FP4 FLOPS advantage.
#   - FP8 uses the CUTLASS native matmul path on SM121a for dense/attention layers.
#   - Max concurrency capped at 4 via --max-running-requests.
#
# Same base image as the GB10 NVFP4 setup: CUDA 13.x is required for sm_121a,
# and Gemma 4 modeling support requires SGLang >= v0.5.11.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.12.post1-cu130
FROM --platform=linux/arm64 ${SGLANG_IMAGE}

ENV MODEL_ID="google/gemma-4-26B-A4B-it" \
    HOST="0.0.0.0" \
    PORT="30000" \
    QUANTIZATION="fp8" \
    KV_CACHE_DTYPE="fp8_e4m3" \
    CONTEXT_LEN="262144" \
    MEM_FRACTION="0.85" \
    MAX_RUNNING_REQUESTS="4" \
    EXTRA_ARGS="" \
    HF_HOME="/root/.cache/huggingface" \
    # Point Triton at CUDA 13.0's ptxas instead of PyTorch's bundled one.
    # The bundled ptxas predates SM_121 and rejects --gpu-name=sm_121a, causing
    # JIT compilation failures for attention and other kernels at runtime.
    # /usr/local/cuda/bin/ptxas (from CUDA 13.0 in this image) knows SM_121a natively.
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# GB10 (SM_121a) Triton MoE kernel config: the default "3072" batch-size entry
# uses BLOCK_SIZE_K=128 + num_stages=3 = 147456 bytes, exceeding the 101376-byte
# shared-memory limit. This hand-tuned config caps that entry at BLOCK_SIZE_K=64
# so every batch size stays under 74 KB.
#
# --quantization fp8 uses the plain fp8_w8a8 filename (no per_channel_quant suffix).
ARG CFG_DIR=/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0
COPY triton_moe_config.json ${CFG_DIR}/E=128,N=704,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
