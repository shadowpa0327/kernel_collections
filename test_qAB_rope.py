import transformers
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, Qwen2RotaryEmbedding
from transformers import AutoConfig
from flash_recontruction import qAB_rope_v1, qAB_rope_v2
import triton

def qK(q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, q_len, head_dim = q.shape
    bsz, num_kv_heads, kv_len, head_dim = K.shape

    num_query_heads_per_group = num_heads // num_kv_heads
    q = q.view(bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim).contiguous()
    K = K.view(bsz, num_kv_heads, kv_len, head_dim).contiguous()

    out = torch.matmul(q, K.transpose(-1, -2))
    return out.view(bsz, num_heads, q_len, kv_len)

def qAB_rope_ref(q_pe, Ak, Bk, cos, sin, position_ids):
    bsz, num_heads, q_len, head_dim = q_pe.shape
    bsz_a, kv_len, rank = Ak.shape
    bsz_b, num_kv_heads, rank, head_dim = Bk.shape
    
    num_query_heads_per_group = num_heads // num_kv_heads
    assert bsz == bsz_a == bsz_b, "Batch size mismatch"

    q_pe = q_pe.view(bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim).contiguous()
    
    # multiply Ak and Bk
    # Reconstruct the key matrix by multiplying Ak and Bk
    # Ak: (bsz, kv_len, rank)
    # Bk: (bsz, num_kv_heads, rank, head_dim)
    # First compute Ak @ Bk to get reconstructed keys
    reconstructed_k = torch.matmul(Ak.unsqueeze(1), Bk) # (bsz, num_kv_heads, kv_len, head_dim)
    # applying rotary pos emb
    reconstructed_k, _ = apply_rotary_pos_emb(reconstructed_k, reconstructed_k, cos, sin, position_ids)
    reconstructed_k = reconstructed_k.to(q_pe.dtype)
    # Now compute q @ reconstructed_k.transpose(-1, -2)
    # q: (bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim)
    # reconstructed_k: (bsz, num_kv_heads, kv_len, head_dim)
    out = torch.matmul(q_pe, reconstructed_k.transpose(-1, -2))  # (bsz, num_kv_heads, num_query_heads_per_group*q_len, kv_len)

    return out.view(bsz, num_heads, q_len, kv_len)

def test_correctness(bsz, num_heads, num_kv_heads, rank, head_dim, q_len, kv_len):
    q = torch.randn(bsz, num_heads, q_len, head_dim).to(torch.float16).cuda() / 5
    Ak = torch.randn(bsz, kv_len, rank).to(torch.float16).cuda() / 5
    Bk = torch.randn(bsz, num_kv_heads, rank, head_dim).to(torch.float16).cuda() / 5


    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    rotary_emb = Qwen2RotaryEmbedding(config)
    position_ids = torch.arange(kv_len, device=q.device).unsqueeze(0).expand(bsz, -1)
    cos, sin = rotary_emb(q, position_ids)
    out_ref = qAB_rope_ref(q, Ak, Bk, cos, sin, position_ids)
    out_triton = qAB_rope_v1(q, Ak, Bk, theta=10000000.0)

    print("output match (ref == triton-v1): ", torch.allclose(out_ref, out_triton, atol=1e-2, rtol=1e-2))

    sin_cos = torch.cat([sin[0,...,:head_dim//2], cos[0,...,:head_dim//2]], dim=-1)

    out_triton_v2 = qAB_rope_v2(q, Ak, Bk, sin_cos)
    print(out_triton_v2, out_ref)
    print(torch.max(torch.abs(out_triton_v2-out_ref)))
    print("output match (ref == triton-v2): ", torch.allclose(out_ref, out_triton_v2, atol=1e-2, rtol=1e-2))

@torch.no_grad()
def run_qk_qab_benchmark(
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    rank=192,
    dtype=torch.float16,
    device="cuda",
):
    configs = []
    # Test different sequence lengths
    configs.append(
        triton.testing.Benchmark(
            x_names=['kv_len'],  # What we're varying on x-axis
            x_vals=[2**i for i in range(14, 18)],  # Testing powers of 2 from 4K to 128K
            line_arg='operation',  # Different lines for different operations
            line_vals=['qk', 'qab_v1', 'qab_v2'],
            line_names=['qK (Standard)', 'qAB_v1 (Triton)', 'qAB_v2 (Triton)'],
            styles=[('red', '--'), ('blue', '-'), ('green', ':')],
            ylabel='ms',
            plot_name=f'qk-qab-comparison-kvh{num_kv_heads}-qh{num_heads}-d{head_dim}-r{rank}',
            args={
                'dtype': dtype,
                'num_heads': num_heads,
                'num_kv_heads': num_kv_heads,
                'head_dim': head_dim,
                'rank': rank,
                'device': device,
            },
        )
    )

    @triton.testing.perf_report(configs)
    def bench_qk_qab(kv_len, operation, num_heads, num_kv_heads, head_dim, rank, dtype, device):
        # Fixed parameters
        bsz = 1
        q_len = 1
        
        # Create input tensors
        q = torch.randn(bsz, num_heads, q_len, head_dim).to(dtype).to(device)
        K = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
        
        # For factorized approach
        Ak = torch.randn(bsz, kv_len, rank).to(dtype).to(device)
        Bk = torch.randn(bsz, num_kv_heads, rank, head_dim).to(dtype).to(device)

        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
        rotary_emb = Qwen2RotaryEmbedding(config)
        position_ids = torch.arange(kv_len, device=q.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = rotary_emb(q, position_ids)
        sin_cos = torch.cat([sin[0,...,:head_dim//2], cos[0,...,:head_dim//2]], dim=-1)

        quantiles = [0.5, 0.2, 0.8]
        warmup = 25
        rep = 100

        if operation == 'qk':
            # Standard qK operation
            def fn():
                return qK(q, K)
        elif operation == 'qab_v1':
            # Triton qAB operation
            def fn():
                return qAB_rope_v1(q, Ak, Bk, theta=10000000.0)
        elif operation == 'qab_v2':
            # Compiled qAB operation
            def fn():
                return qAB_rope_v2(q, Ak, Bk, sin_cos)
        else:  # qab_ref
            raise ValueError(f"Unknown operation: {operation}")

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
        return ms, min_ms, max_ms

    # Run benchmarks and save results
    bench_qk_qab.run(print_data=True, show_plots=True, save_path='results/')

if __name__ == "__main__":
    q_len = 1
    bsz = 1
    num_heads = 32
    num_kv_heads = 4
    rank = 384
    head_dim = 128
    kv_len = 8192
    dtype = torch.float16
    device = "cuda"

    #test_correctness(bsz, num_heads, num_kv_heads, rank, head_dim, q_len, kv_len)
    run_qk_qab_benchmark(num_heads, num_kv_heads, head_dim, rank, dtype, device)
    