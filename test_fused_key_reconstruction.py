"""We want triton==3.0.0 for this script
"""

import torch
import triton
from flash_recontruction import qAB_no_pe_v1, qAB_no_pe_v2, qAB_no_pe_ref

def qK(q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, q_len, head_dim = q.shape
    bsz, num_kv_heads, kv_len, head_dim = K.shape

    num_query_heads_per_group = num_heads // num_kv_heads
    q = q.view(bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim).contiguous()
    K = K.view(bsz, num_kv_heads, kv_len, head_dim).contiguous()

    out = torch.matmul(q, K.transpose(-1, -2))
    return out.view(bsz, num_heads, q_len, kv_len)
    

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
            line_vals=['qk', 'qab_no_pe_v1', 'qab_no_pe_v2', 'qab_no_pe_ref'],
            line_names=['qK (Standard)', 'qAB_v1 (Triton)', 'qAB_v2 (Triton)', 'qAB_ref (Eager)'],
            styles=[('red', '--'), ('blue', '-'), ('green', ':'), ('yellow', '-')],
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

        quantiles = [0.5, 0.2, 0.8]
        warmup = 25
        rep = 100

        if operation == 'qk':
            # Standard qK operation
            def fn():
                return qK(q, K)
        elif operation == 'qab_no_pe_v1':
            # Triton qAB operation
            def fn():
                return qAB_no_pe_v1(q, Ak, Bk)
        elif operation == 'qab_no_pe_v2':
            # Compiled qAB operation
            def fn():
                return qAB_no_pe_v2(q, Ak, Bk)
        else:  # qab_ref
            # Reference qAB operation
            def fn():
                return qAB_no_pe_ref(q, Ak, Bk)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
        return ms, min_ms, max_ms

    # Run benchmarks and save results
    bench_qk_qab.run(print_data=True, show_plots=True, save_path='results/')

if __name__ == "__main__":
    q_len = 1
    bsz = 1
    num_heads = 32
    num_kv_heads = 4
    rank = 192
    head_dim = 128
    kv_len = 65536*2

    q = torch.randn(bsz, num_heads, q_len, head_dim).to(torch.float16).cuda()
    Ak = torch.randn(bsz, kv_len, rank).to(torch.float16).cuda()
    Bk = torch.randn(bsz, num_kv_heads, rank, head_dim).to(torch.float16).cuda()
    K = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(torch.float16).cuda()

    # Run the benchmark
    run_qk_qab_benchmark(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, rank=rank, dtype=torch.float16, device="cuda")