"""
Test and benchmark fla_volta CUDA kernels against PyTorch fallback.
Run on the V100 server after: cd fla_volta && pip install -e .
"""
import torch
import time
import sys


def test_fused_rms_norm_gated():
    """Validate norm kernel matches PyTorch reference."""
    from fla_volta import fused_rms_norm_gated

    print("=== Test: fused_rms_norm_gated ===")
    torch.manual_seed(42)
    N, D = 1024, 128
    x = torch.randn(N, D, device="cuda", dtype=torch.float16)
    gate = torch.randn(N, D, device="cuda", dtype=torch.float16)
    weight = torch.randn(D, device="cuda", dtype=torch.float16)
    eps = 1e-6

    # Reference (from Qwen3_5RMSNormGated)
    x_f32 = x.float()
    g_f32 = gate.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(variance + eps)
    normed = weight.float() * normed
    ref = (normed * torch.nn.functional.silu(g_f32)).half()

    # CUDA kernel
    out = fused_rms_norm_gated(x, gate, weight, eps)

    diff = (ref.float() - out.float()).abs().max().item()
    print(f"  Max absolute diff: {diff:.6e}")
    assert diff < 1e-2, f"FAIL: diff too large: {diff}"
    print("  PASS")

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        fused_rms_norm_gated(x, gate, weight, eps)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        x_f32 = x.float()
        g_f32 = gate.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + eps)
        normed = weight.float() * normed
        _ = (normed * torch.nn.functional.silu(g_f32)).half()
    torch.cuda.synchronize()
    ref_time = time.time() - start

    print(f"  CUDA kernel: {cuda_time*1000:.1f}ms / 1000 iters")
    print(f"  PyTorch ref: {ref_time*1000:.1f}ms / 1000 iters")
    print(f"  Speedup: {ref_time/cuda_time:.2f}x")
    print()


def test_fused_recurrent_gdn():
    """Validate recurrent GDN kernel matches PyTorch reference."""
    from fla_volta import fused_recurrent_gated_delta_rule

    print("=== Test: fused_recurrent_gated_delta_rule ===")
    torch.manual_seed(42)
    B, T, H, K, V = 1, 4, 4, 128, 128  # small for validation
    query = torch.randn(B, T, H, K, device="cuda", dtype=torch.float16)
    key = torch.randn(B, T, H, K, device="cuda", dtype=torch.float16)
    value = torch.randn(B, T, H, V, device="cuda", dtype=torch.float16)
    g = torch.randn(B, T, H, device="cuda", dtype=torch.float16) * 0.1
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.float16)

    # Reference: torch_recurrent_gated_delta_rule from modeling_qwen3_5.py
    def torch_ref(query, key, value, g, beta):
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]
        B, H, T, K = key.shape
        V = value.shape[-1]
        scale = 1 / (K ** 0.5)
        query = query * scale
        output = torch.zeros(B, H, T, V, device=query.device, dtype=torch.float32)
        state = torch.zeros(B, H, K, V, device=query.device, dtype=torch.float32)
        for i in range(T):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            state = state * g_t
            kv = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        return output.transpose(1, 2).contiguous()

    ref = torch_ref(query, key, value, g, beta).half()

    # CUDA kernel
    out, _ = fused_recurrent_gated_delta_rule(
        query, key, value, g, beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False
    )

    diff = (ref.float() - out.float()).abs().max().item()
    rel_diff = diff / (ref.float().abs().max().item() + 1e-8)
    print(f"  Max absolute diff: {diff:.6e}")
    print(f"  Max relative diff: {rel_diff:.6e}")
    assert rel_diff < 1e-2, f"FAIL: relative diff too large: {rel_diff}"
    print("  PASS")

    # Benchmark: decode scenario (T=1, H=32, K=128, V=128)
    print("\n  Benchmark: decode (T=1, H=32, K=128, V=128)")
    B, T, H, K, V = 1, 1, 32, 128, 128
    query = torch.randn(B, T, H, K, device="cuda", dtype=torch.float16)
    key = torch.randn(B, T, H, K, device="cuda", dtype=torch.float16)
    value = torch.randn(B, T, H, V, device="cuda", dtype=torch.float16)
    g = torch.randn(B, T, H, device="cuda", dtype=torch.float16) * 0.1
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.float16)
    state = torch.randn(B, H, K, V, device="cuda", dtype=torch.float16) * 0.01

    # Warmup
    for _ in range(10):
        fused_recurrent_gated_delta_rule(query, key, value, g, beta,
            initial_state=state, output_final_state=True)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        fused_recurrent_gated_delta_rule(query, key, value, g, beta,
            initial_state=state, output_final_state=True)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    # PyTorch reference benchmark
    for _ in range(10):
        torch_ref(query, key, value, g, beta)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        torch_ref(query, key, value, g, beta)
    torch.cuda.synchronize()
    ref_time = time.time() - start

    print(f"  CUDA kernel: {cuda_time*1000:.1f}ms / 1000 iters")
    print(f"  PyTorch ref: {ref_time*1000:.1f}ms / 1000 iters")
    print(f"  Speedup: {ref_time/cuda_time:.2f}x")
    print()


def test_e2e_model():
    """End-to-end test: load Qwen3.5-2B with monkey patch and generate."""
    print("=== Test: end-to-end Qwen3.5-2B generation ===")

    import fla_volta.monkey_patch  # patches transformers

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3.5-2B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda:0"
    )

    prompt = "Tell me a story about a dragon boy who discovers he can breathe fire for the first time."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    # Warmup
    model.generate(**inputs, max_new_tokens=5)

    # Timed run
    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n{text}\n")
    print(f"--- {n_new} tokens in {elapsed:.2f}s = {n_new/elapsed:.1f} tok/s ---")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--e2e":
        test_e2e_model()
    else:
        test_fused_rms_norm_gated()
        test_fused_recurrent_gdn()
        print("All kernel tests passed. Run with --e2e for full model test.")
