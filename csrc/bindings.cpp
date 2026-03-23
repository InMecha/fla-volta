/*
 * PyBind11 bindings for fla_volta CUDA kernels
 */
#include <torch/extension.h>

// Declarations from .cu files
torch::Tensor fused_rms_norm_gated_forward(
    torch::Tensor x, torch::Tensor gate, torch::Tensor weight, float eps);

std::vector<torch::Tensor> fused_recurrent_gated_delta_rule_forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor g, torch::Tensor beta, torch::Tensor initial_state,
    bool output_final_state, bool use_qk_l2norm_in_kernel);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rms_norm_gated_forward", &fused_rms_norm_gated_forward,
          "Fused RMSNorm + SiLU Gate forward (CUDA, sm_70)");
    m.def("fused_recurrent_gated_delta_rule_forward",
          &fused_recurrent_gated_delta_rule_forward,
          "Fused Recurrent Gated Delta Rule forward (CUDA, sm_70)");
}
