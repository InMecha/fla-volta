# fla_volta

Native CUDA kernels for [FLA](https://github.com/fla-org/flash-linear-attention) ops on NVIDIA Volta (sm_70) GPUs.

Replaces FLA's Triton-based kernels that fail to compile on V100, enabling Qwen 3.5 and other Gated DeltaNet models to run on V100 hardware through HuggingFace Transformers.

## Problem

FLA's Triton kernels for Gated DeltaNet (GDN) layers hang indefinitely during autotuner compilation on sm_70 (Volta). This blocks all Qwen 3.5 models from running on V100 GPUs through HuggingFace.

## Solution

Two hand-written CUDA kernels compiled for sm_70:

- **fused_rms_norm_gated** — fused RMSNorm + SiLU gate (replaces `fla.modules.FusedRMSNormGated`)
- **fused_recurrent_gated_delta_rule** — fused recurrent GDN (replaces `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`)

The recurrent kernel is adapted from [llama.cpp's gated_delta_net.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu).

## Requirements

- NVIDIA V100 (sm_70) GPU
- CUDA toolkit 12.x
- PyTorch 2.x with CUDA support
- HuggingFace Transformers with Qwen 3.5 support

## Install
```bash
git clone https://github.com/inmecha/fla-volta.git
cd fla-volta
pip install -e . --no-build-isolation
```

## Usage
```python
import fla_volta.monkey_patch  # BEFORE loading the model

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B",
    torch_dtype=torch.float16, device_map="cuda:0")
```

## Performance

Tested on V100-SXM2-32GB, Qwen3.5-2B, 200 tokens:

| Configuration | tok/s |
|---|---|
| FLA Triton kernels | Hangs indefinitely |
| PyTorch fallback (no FLA) | 11.5 |
| **fla_volta CUDA kernels** | **16.8** |

Kernel-level speedup is ~3x over PyTorch fallback. End-to-end improvement is limited by HuggingFace generate() Python overhead. For maximum inference speed on V100, use llama.cpp with GGUF.

Theoretically this should give you about 100tps on a Gated DeltaNet transformer model for a model that fits on a single V100 GPU 35GB. Realistically you will probably be CPU/python bound as we profiled that the V100 GPU with the modified CU code crunches the tokens so fast the TPS becomes CPU bound, like 10%/90% split (10% GPU and 90% CPU).

## License

The recurrent GDN kernel is adapted from llama.cpp (MIT License).

Thanks to Georgi Gerganov for his work.

Norm kernel is our original work.

AGPL-3.0 — INMECHA INC, 2026.

VALENTIN PETROV
