# google-gemma4-26b-a4b-it-bf16-sglang

Docker image that runs **Google Gemma 4 26B-A4B-IT** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The model is downloaded in **BF16** from Hugging Face and **SGLang quantizes it to FP8 at load time** — no pre-quantized weights are used.

## What it is

Gemma 4 26B-A4B is a Mixture-of-Experts (MoE) language model: it has 26 billion total parameters but only activates about 4 billion of them per token, so it runs with the compute cost of a much smaller model while retaining the capacity of a large one.

This image serves the model using [SGLang](https://github.com/sgl-project/sglang). FP8 is the right quantization choice for the GB10 because the chip has native FP8 GEMM kernels (SM_121a / CUTLASS path) but no native FP4 path.

The image includes a hand-tuned Triton MoE kernel config that keeps shared memory usage within the GB10's hardware limits.

## BF16 weights + SGLang FP8 on-the-fly quantization

SGLang's `--quantization fp8` flag loads the official BF16 weights and converts them to FP8 in memory during model initialization. This is different from using pre-quantized weights (e.g. `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`).

**Pros**

- Uses the official Google weights — no third-party quantization artifacts or format conversions.
- Quantization parameters (per-tensor, per-channel) are computed from the actual weight distribution at load time, so they are always fresh and accurate for this exact checkpoint.
- More flexible: quantization behavior can be tuned or swapped at launch without re-downloading weights.
- SGLang uses the native CUTLASS FP8 matmul path on SM_121a for dense and attention layers, so inference throughput is the same as with pre-quantized weights.

**Cons**

- BF16 weights are roughly twice the download size of FP8 weights (~52 GB vs ~26 GB).
- Quantization adds to startup time (typically a few extra minutes on first launch; subsequent launches reuse the cached BF16 weights but still re-quantize in memory).
- Peak VRAM during load is briefly higher because BF16 and FP8 tensors coexist until the conversion is complete.
- If a carefully calibrated pre-quantized model is available, its per-channel scales may produce slightly lower quantization error than SGLang's on-the-fly per-tensor approach.

## Requirements

- NVIDIA GB10 / DGX Spark (SM_121a)
- Docker with NVIDIA Container Toolkit
- A Hugging Face token with access to [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (the model is gated — accept the license on the model page first)
- The `llm-net` Docker network: `docker network create llm-net`

## Usage

```bash
HF_TOKEN=hf_xxx docker compose up --build
```

The server starts on port **30000** and exposes an OpenAI-compatible API once the health check passes (allow up to 10 minutes for the first run while weights download).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face access token |
| `CONTEXT_LEN` | `65536` | Maximum context length in tokens |
| `MEM_FRACTION` | `0.60` | Fraction of VRAM reserved for the KV cache |
| `EXTRA_ARGS` | *(empty)* | Extra flags passed directly to `sglang.launch_server` |

## License

MIT — see [LICENSE](LICENSE).
