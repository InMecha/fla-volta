"""
Monkey-patches transformers' Qwen3.5 model to use fla_volta CUDA kernels
instead of FLA's Triton ops (which hang on Volta/sm_70).

Usage:
    import fla_volta.monkey_patch  # must import BEFORE loading the model
    
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", ...)
"""

import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
from fla_volta import fused_recurrent_gated_delta_rule, FusedRMSNormGatedVolta

# Replace FLA ops with our Volta CUDA kernels
qmod.FusedRMSNormGated = FusedRMSNormGatedVolta
qmod.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule

# Keep chunk_gated_delta_rule as the torch fallback for prefill
# (the Triton chunked kernel also hangs, and the torch fallback is
#  acceptable for prefill since it only runs once per sequence)
qmod.chunk_gated_delta_rule = None

# Update the fast path flag — we have fast recurrent + norm, slow chunk
# Setting False means the "fast path not available" warning prints once,
# but chunk_gated_delta_rule will use torch_chunk_gated_delta_rule fallback
# while norm and recurrent use our fast CUDA kernels
qmod.is_fast_path_available = False

print("[fla_volta] Patched Qwen3.5: recurrent GDN + RMSNormGated → Volta CUDA kernels")
