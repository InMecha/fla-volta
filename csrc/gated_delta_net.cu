/*
 * Fused Recurrent Gated Delta Net kernel for Volta (sm_70)
 * Adapted from llama.cpp's gated_delta_net.cu for PyTorch tensor interface.
 *
 * The recurrence per timestep:
 *   state = state * exp(g_t)
 *   kv = (state @ k_t)            // contract over K dimension
 *   delta = (v_t - kv) * beta_t
 *   state = state + outer(k_t, delta)
 *   output_t = state @ q_t        // contract over K dimension
 *
 * State is [B, H, K, V] — K rows, V columns.
 * Each warp owns one column of the state matrix.
 * Warp lanes hold shards of rows within that column.
 * State lives in registers across all timesteps — no global memory round-trips.
 *
 * Grid: (H, B, ceil(V / num_warps))
 * Block: (warp_size, num_warps)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cmath>

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum_gdn(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * Template parameters:
 *   S_K: key/state row dimension (e.g. 128)
 *   S_V: value/state column dimension (e.g. 128)
 */
template <int S_K>
__global__ void __launch_bounds__(128, 2)  // 32 lanes * 4 warps, 2 blocks/SM
fused_recurrent_gdn_kernel(
    const float* __restrict__ q,       // [B, H, T, K]
    const float* __restrict__ k,       // [B, H, T, K]
    const float* __restrict__ v,       // [B, H, T, V]
    const float* __restrict__ g,       // [B, H, T]
    const float* __restrict__ beta,    // [B, H, T]
    const float* __restrict__ state_in,// [B, H, K, V] or nullptr
    float* __restrict__ output,        // [B, H, T, V]
    float* __restrict__ state_out,     // [B, H, K, V] or nullptr
    int B, int H, int T, int V,
    float scale)
{
    const int h_idx = blockIdx.x;         // head index
    const int b_idx = blockIdx.y;         // batch index
    const int lane  = threadIdx.x;        // lane within warp (0..31)
    const int col   = blockIdx.z * blockDim.y + threadIdx.y;  // V column index

    if (col >= V) return;

    constexpr int WARP_SIZE = 32;
    static_assert(S_K % WARP_SIZE == 0, "S_K must be a multiple of 32");
    constexpr int ROWS_PER_LANE = S_K / WARP_SIZE;

    // Strides for [B, H, T, D] layout
    const int stride_bh = H * T;
    const int bh_offset_t = (b_idx * H + h_idx) * T;    // base offset to [b, h, 0]
    const int bh_offset_kv = bh_offset_t;

    // Strides for [B, H, T] layout (g, beta)
    const int gb_base = (b_idx * H + h_idx) * T;

    // State offset [B, H, K, V]
    const int state_base = (b_idx * H + h_idx) * S_K * V;

    // Load initial state column into registers
    float s_shard[ROWS_PER_LANE];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
        const int row = r * WARP_SIZE + lane;
        if (state_in != nullptr) {
            s_shard[r] = state_in[state_base + row * V + col];
        } else {
            s_shard[r] = 0.0f;
        }
    }

    // Process each timestep
    for (int t = 0; t < T; t++) {
        // Read q, k for this timestep — [B, H, T, K]
        const float* q_t = q + (bh_offset_t + t) * S_K;
        const float* k_t = k + (bh_offset_t + t) * S_K;
        // Read v — [B, H, T, V]
        const float* v_t = v + (bh_offset_kv + t) * V;
        // Read scalar g, beta — [B, H, T]
        const float g_val = expf(g[gb_base + t]);
        const float beta_val = beta[gb_base + t];

        // Cache k and q in registers
        float k_reg[ROWS_PER_LANE];
        float q_reg[ROWS_PER_LANE];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; r++) {
            const int row = r * WARP_SIZE + lane;
            k_reg[r] = k_t[row];
            q_reg[r] = q_t[row] * scale;
        }

        // Step 1: kv[col] = sum_i (g * state[i][col]) * k[i]
        //   = sum over K dimension of state column * k
        float kv_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; r++) {
            kv_partial += (g_val * s_shard[r]) * k_reg[r];
        }
        float kv_col = warp_reduce_sum_gdn(kv_partial);
        // kv_col is now valid in lane 0, but we need it broadcast
        kv_col = __shfl_sync(0xffffffff, kv_col, 0);

        // Step 2: delta[col] = (v[col] - kv[col]) * beta
        float v_col = v_t[col];
        float delta_col = (v_col - kv_col) * beta_val;

        // Step 3: Fused state update + output computation
        //   state[i][col] = g * state[i][col] + k[i] * delta[col]
        //   attn[col] = sum_i state[i][col] * q[i]
        float attn_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; r++) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        float attn_col = warp_reduce_sum_gdn(attn_partial);

        // Write output — only lane 0 has the reduced value
        if (lane == 0) {
            output[(bh_offset_kv + t) * V + col] = attn_col;
        }
    }

    // Write final state back to global memory
    if (state_out != nullptr) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; r++) {
            const int row = r * WARP_SIZE + lane;
            state_out[state_base + row * V + col] = s_shard[r];
        }
    }
}


// Launch helper
void launch_fused_recurrent_gdn(
    torch::Tensor q,            // [B, H, T, K] float32
    torch::Tensor k,            // [B, H, T, K] float32
    torch::Tensor v,            // [B, H, T, V] float32
    torch::Tensor g,            // [B, H, T] float32
    torch::Tensor beta,         // [B, H, T] float32
    torch::Tensor state_in,     // [B, H, K, V] float32 or empty
    torch::Tensor output,       // [B, H, T, V] float32
    torch::Tensor state_out,    // [B, H, K, V] float32 or empty
    float scale)
{
    int B = q.size(0);
    int H = q.size(1);
    int T = q.size(2);
    int K = q.size(3);
    int V = v.size(3);

    const int NUM_WARPS = 4;
    const int WARP_SIZE = 32;

    dim3 grid(H, B, (V + NUM_WARPS - 1) / NUM_WARPS);
    dim3 block(WARP_SIZE, NUM_WARPS);

    const float* state_in_ptr = state_in.numel() > 0 ? state_in.data_ptr<float>() : nullptr;
    float* state_out_ptr = state_out.numel() > 0 ? state_out.data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Dispatch based on K dimension
    switch (K) {
        case 64:
            fused_recurrent_gdn_kernel<64><<<grid, block, 0, stream>>>(
                q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                g.data_ptr<float>(), beta.data_ptr<float>(), state_in_ptr,
                output.data_ptr<float>(), state_out_ptr,
                B, H, T, V, scale);
            break;
        case 128:
            fused_recurrent_gdn_kernel<128><<<grid, block, 0, stream>>>(
                q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                g.data_ptr<float>(), beta.data_ptr<float>(), state_in_ptr,
                output.data_ptr<float>(), state_out_ptr,
                B, H, T, V, scale);
            break;
        case 256:
            fused_recurrent_gdn_kernel<256><<<grid, block, 0, stream>>>(
                q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                g.data_ptr<float>(), beta.data_ptr<float>(), state_in_ptr,
                output.data_ptr<float>(), state_out_ptr,
                B, H, T, V, scale);
            break;
        default:
            TORCH_CHECK(false, "Unsupported K dimension: ", K, ". Must be 64, 128, or 256.");
    }
}


// Python-facing function matching torch_recurrent_gated_delta_rule interface
// Inputs: [B, T, H, K/V] (pre-transpose, matching HuggingFace convention)
// Outputs: [B, T, H, V]
std::vector<torch::Tensor> fused_recurrent_gated_delta_rule_forward(
    torch::Tensor query,        // [B, T, H, K]
    torch::Tensor key,          // [B, T, H, K]
    torch::Tensor value,        // [B, T, H, V]
    torch::Tensor g,            // [B, T, H]
    torch::Tensor beta,         // [B, T, H]
    torch::Tensor initial_state,// [B, H, K, V] or empty
    bool output_final_state,
    bool use_qk_l2norm_in_kernel)
{
    auto input_dtype = query.dtype();

    // L2 norm if requested
    if (use_qk_l2norm_in_kernel) {
        auto q_norm = torch::rsqrt((query * query).sum(-1, true) + 1e-6f);
        query = query * q_norm;
        auto k_norm = torch::rsqrt((key * key).sum(-1, true) + 1e-6f);
        key = key * k_norm;
    }

    // Transpose to [B, H, T, D] and cast to FP32
    auto q_f32 = query.transpose(1, 2).contiguous().to(torch::kFloat32);
    auto k_f32 = key.transpose(1, 2).contiguous().to(torch::kFloat32);
    auto v_f32 = value.transpose(1, 2).contiguous().to(torch::kFloat32);
    auto g_f32 = g.transpose(1, 2).contiguous().to(torch::kFloat32);
    auto beta_f32 = beta.transpose(1, 2).contiguous().to(torch::kFloat32);

    int B = q_f32.size(0);
    int H = q_f32.size(1);
    int T = q_f32.size(2);
    int K = q_f32.size(3);
    int V = v_f32.size(3);

    float scale = 1.0f / sqrtf((float)K);

    auto output = torch::zeros({B, H, T, V}, q_f32.options());

    torch::Tensor state_in = initial_state.numel() > 0
        ? initial_state.to(torch::kFloat32).contiguous()
        : torch::empty({0}, q_f32.options());

    torch::Tensor state_out;
    if (output_final_state) {
        state_out = torch::zeros({B, H, K, V}, q_f32.options());
    } else {
        state_out = torch::empty({0}, q_f32.options());
    }

    launch_fused_recurrent_gdn(q_f32, k_f32, v_f32, g_f32, beta_f32,
                                state_in, output, state_out, scale);

    // Transpose output back to [B, T, H, V]
    auto output_final = output.transpose(1, 2).contiguous().to(input_dtype);

    torch::Tensor final_state;
    if (output_final_state) {
        final_state = state_out;
    } else {
        final_state = torch::Tensor();  // None equivalent
    }

    return {output_final, final_state};
}
