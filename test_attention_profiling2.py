import torch
import triton.testing
import matplotlib.pyplot as plt
from test_attention import group_query_attention, group_query_attention_fa, group_query_attention_factorized

@torch.no_grad()
def run_attention_benchmark(
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    rank=192,
    dtype=torch.bfloat16,
    device="cuda",
):
    configs = []
    # Test different sequence lengths
    configs.append(
        triton.testing.Benchmark(
            x_names=['kv_len'],  # What we're varying on x-axis
            x_vals=[2**i for i in range(14, 18)],  # Testing powers of 2 from 16K to 128K
            line_arg='provider',  # Different lines for different implementations
            line_vals=['standard', 'flash_attn', 'factorized'],
            line_names=['Eager', 'FlashAttention', 'Eager + xKV (v-only)'],
            styles=[('red', '--'), ('green', '--'), ('blue', '-')],
            ylabel='ms',
            plot_name=f'attention-comparison-h{num_heads}-d{head_dim}-r{rank}',
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
    def bench_attention(kv_len, provider, num_heads, num_kv_heads, head_dim, rank, dtype, device):
        # Fixed parameters
        bsz = 1
        q_len = 1
        
        # Create input tensors
        q = torch.randn(bsz, num_heads, q_len, head_dim).to(dtype).to(device)
        k = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
        v = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
        v_A = torch.randn(bsz, kv_len, rank).to(dtype).to(device)
        v_B = torch.randn(bsz, num_kv_heads, rank, head_dim).to(dtype).to(device)

        quantiles = [0.5, 0.2, 0.8]
        warmup = 25
        rep = 100

        if provider == 'standard':
            def fn(): return group_query_attention(q, k, v)
        elif provider == 'flash_attn':
            def fn(): return group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        else:  # factorized
            def fn(): return group_query_attention_factorized(q, k, v_A, v_B, use_factorized=True)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
        return ms, min_ms, max_ms

    # Run benchmarks and save results
    bench_attention.run(print_data=True, show_plots=True, save_path='results/')

@torch.no_grad()
def profile_attention_implementations(
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    kv_len=65536,  # 64K sequence length
    rank=192,
    dtype=torch.bfloat16,
    device="cuda",
):
    import torch.profiler
    from torch.profiler import profile, record_function, ProfilerActivity

    # Fixed parameters
    bsz = 1
    q_len = 1
    
    # Create input tensors
    q = torch.randn(bsz, num_heads, q_len, head_dim).to(dtype).to(device)
    k = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    v = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    v_A = torch.randn(bsz, kv_len, rank).to(dtype).to(device)
    v_B = torch.randn(bsz, num_kv_heads, rank, head_dim).to(dtype).to(device)

    # Warmup
    for _ in range(5):
        group_query_attention(q, k, v)
        group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        group_query_attention_factorized(q, k, v_A, v_B, use_factorized=True)

    # Profile each implementation
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Standard attention
        with record_function("Standard Attention"):
            group_query_attention(q, k, v)
        
        torch.cuda.synchronize()

        # Flash attention
        with record_function("Flash Attention"):
            group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        
        torch.cuda.synchronize()
        # Factorized attention
        with record_function("Factorized Attention"):
            group_query_attention_factorized(q, k, v_A, v_B, use_factorized=True)

        torch.cuda.synchronize()

    # Print profiling results
    print("\n=== Profiling Results ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
    ))

    # Save trace to file
    prof.export_chrome_trace("attention_profile_trace.json")

@torch.no_grad()
def run_qk_qab_benchmark(
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    rank=192,
    dtype=torch.bfloat16,
    device="cuda",
):
    configs = []
    # Test different sequence lengths
    configs.append(
        triton.testing.Benchmark(
            x_names=['kv_len'],  # What we're varying on x-axis
            x_vals=[2**i for i in range(14, 18)],  # Testing powers of 2 from 16K to 128K
            line_arg='operation',  # Different lines for different operations
            line_vals=['qk', 'qab'],
            line_names=['q·K (Standard)', 'q·(A·B) (Factorized)'],
            styles=[('red', '--'), ('blue', '-')],
            ylabel='ms',
            plot_name=f'qk-qab-comparison-h{num_heads}-d{head_dim}-r{rank}',
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
        k = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
        
        # For factorized approach
        k_A = torch.randn(bsz, kv_len, rank).to(dtype).to(device)
        k_B = torch.randn(bsz, num_kv_heads, rank, head_dim).to(dtype).to(device)

        quantiles = [0.5, 0.2, 0.8]
        warmup = 25
        rep = 100

        if operation == 'qk':
            # Standard q·K operation
            def fn():
                # Expand k to match q's head dimension
                k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                # Compute attention scores: (bsz, num_heads, q_len, head_dim) @ (bsz, num_heads, kv_len, head_dim)T
                scores = torch.matmul(q, k_expanded.transpose(-1, -2))
                return scores
        else:  # qab
            # Factorized q·(A·B) operation
            def fn():
                # First compute A·B partially: (bsz, kv_len, rank) @ (bsz, num_kv_heads, rank, head_dim)
                # We need to be careful with broadcasting here
                k_factorized = torch.einsum('blr,bnrh->bnlh', k_A, k_B)
                # Expand to match q's head dimension
                k_factorized_expanded = k_factorized.repeat_interleave(num_heads // num_kv_heads, dim=1)
                # Compute attention scores
                scores = torch.matmul(q, k_factorized_expanded.transpose(-1, -2))
                return scores

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
        return ms, min_ms, max_ms

    # Run benchmarks and save results
    bench_qk_qab.run(print_data=True, show_plots=True, save_path='results/')

@torch.no_grad()
def profile_qk_qab_operations(
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    kv_len=65536,  # 64K sequence length
    rank=192,
    dtype=torch.bfloat16,
    device="cuda",
):
    import torch.profiler
    from torch.profiler import profile, record_function, ProfilerActivity

    # Fixed parameters
    bsz = 1
    q_len = 1
    
    # Create input tensors
    q = torch.randn(bsz, num_heads, q_len, head_dim).to(dtype).to(device)
    k = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    
    # For factorized approach
    k_A = torch.randn(bsz, kv_len, rank).to(dtype).to(device)
    k_B = torch.randn(bsz, num_kv_heads, rank, head_dim).to(dtype).to(device)

    # Define operations
    def qk_operation():
        # Expand k to match q's head dimension
        k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        # Compute attention scores
        scores = torch.matmul(q, k_expanded.transpose(-1, -2))
        return scores
    
    def qab_operation():
        # First compute A·B partially
        k_factorized = torch.einsum('blr,bnrh->bnlh', k_A, k_B)
        # Expand to match q's head dimension
        k_factorized_expanded = k_factorized.repeat_interleave(num_heads // num_kv_heads, dim=1)
        # Compute attention scores
        scores = torch.matmul(q, k_factorized_expanded.transpose(-1, -2))
        return scores

    # Warmup
    for _ in range(5):
        qk_operation()
        qab_operation()

    # Profile each operation
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Standard q·K
        with record_function("q·K Operation"):
            qk_operation()
        
        torch.cuda.synchronize()

        # Factorized q·(A·B)
        with record_function("q·(A·B) Operation"):
            qab_operation()

        torch.cuda.synchronize()

    # Print profiling results
    print("\n=== q·K vs q·(A·B) Profiling Results ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
    ))

    # Save trace to file
    prof.export_chrome_trace("qk_qab_profile_trace.json")

if __name__ == "__main__":
    #run_attention_benchmark()
    #profile_attention_implementations()
    run_qk_qab_benchmark()
    #profile_qk_qab_operations()