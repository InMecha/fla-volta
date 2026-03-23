/*
 * Fused RMSNorm + SiLU Gate kernel for Volta (sm_70)
 * Replaces FLA's Triton-based FusedRMSNormGated that fails to compile on V100.
 *
 * Computes: output = RMSNorm(x) * weight * SiLU(gate)
 * where RMSNorm(x) = x * rsqrt(mean(x^2) + eps)
 *
 * Each block processes one row (one token×head).
 * Block-level reduction computes the variance.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// Warp-level reduce
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void fused_rms_norm_gated_kernel(
    const float* __restrict__ x,       // [N, D]
    const float* __restrict__ gate,    // [N, D]
    const float* __restrict__ weight,  // [D]
    float* __restrict__ output,        // [N, D]
    int D,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* x_row = x + row * D;
    const float* g_row = gate + row * D;
    float* o_row = output + row * D;

    // Compute sum of squares for this row (partial)
    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    // Warp reduce
    sum_sq = warp_reduce_sum(sum_sq);

    // Cross-warp reduce via shared memory
    __shared__ float shared[32];
    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }

    // Broadcast rstd to all threads
    __shared__ float s_rstd;
    if (tid == 0) {
        s_rstd = rsqrtf(sum_sq / (float)D + eps);
    }
    __syncthreads();
    float rstd = s_rstd;

    // Fused: normed * weight * silu(gate)
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float x_val = x_row[i];
        float g_val = g_row[i];
        float w_val = weight[i];

        float normed = x_val * rstd * w_val;
        // SiLU(g) = g * sigmoid(g)
        float sigmoid_g = 1.0f / (1.0f + expf(-g_val));
        float silu_g = g_val * sigmoid_g;

        o_row[i] = normed * silu_g;
    }
}

// Entry point: accepts FP16 or FP32 tensors, computes in FP32
torch::Tensor fused_rms_norm_gated_forward(
    torch::Tensor x,       // [N, D]
    torch::Tensor gate,    // [N, D]
    torch::Tensor weight,  // [D]
    float eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    auto input_dtype = x.dtype();

    // Cast to FP32 for computation
    auto x_f32 = x.to(torch::kFloat32).contiguous();
    auto gate_f32 = gate.to(torch::kFloat32).contiguous();
    auto weight_f32 = weight.to(torch::kFloat32).contiguous();

    int N = x_f32.size(0);
    int D = x_f32.size(1);

    auto output = torch::empty({N, D}, x_f32.options());

    constexpr int BLOCK_SIZE = 256;
    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    fused_rms_norm_gated_kernel<BLOCK_SIZE><<<grid, block, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        x_f32.data_ptr<float>(),
        gate_f32.data_ptr<float>(),
        weight_f32.data_ptr<float>(),
        output.data_ptr<float>(),
        D,
        eps);

    // Cast back to input dtype
    return output.to(input_dtype);
}
