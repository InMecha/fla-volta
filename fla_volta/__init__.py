"""
fla_volta: Native CUDA kernels for FLA ops on Volta (sm_70).
Replaces Triton-based FLA kernels that fail to compile on V100 GPUs.
"""

import torch
import torch.nn as nn

from . import _C

__version__ = "0.1.0"


def fused_rms_norm_gated(x, gate, weight, eps=1e-6):
    """
    Fused RMSNorm + SiLU gate.
    
    Args:
        x: [N, D] input tensor
        gate: [N, D] gate tensor
        weight: [D] norm weight
        eps: epsilon for numerical stability
    
    Returns:
        [N, D] output = RMSNorm(x) * weight * SiLU(gate)
    """
    return _C.fused_rms_norm_gated_forward(x, gate, weight, eps)


def fused_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs  # absorb extra kwargs from FLA interface
):
    """
    Fused recurrent gated delta rule.
    Drop-in replacement for fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule
    
    Args:
        query: [B, T, H, K]
        key: [B, T, H, K]
        value: [B, T, H, V]
        g: [B, T, H]
        beta: [B, T, H]
        initial_state: [B, H, K, V] or None
        output_final_state: bool
        use_qk_l2norm_in_kernel: bool
    
    Returns:
        (output [B, T, H, V], final_state [B, H, K, V] or None)
    """
    if initial_state is None:
        initial_state = torch.empty(0, device=query.device)
    
    results = _C.fused_recurrent_gated_delta_rule_forward(
        query, key, value, g, beta, initial_state,
        output_final_state, use_qk_l2norm_in_kernel
    )
    
    output = results[0]
    final_state = results[1] if output_final_state else None
    return output, final_state


class FusedRMSNormGatedVolta(nn.Module):
    """
    Drop-in replacement for fla.modules.FusedRMSNormGated on Volta GPUs.
    Uses native CUDA kernel instead of Triton.
    """
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states, gate=None):
        if gate is None:
            # Fallback: just RMSNorm without gate
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight * hidden_states).to(input_dtype)
        
        return fused_rms_norm_gated(hidden_states, gate, self.weight, self.variance_epsilon)
