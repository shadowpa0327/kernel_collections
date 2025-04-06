import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from functools import partial
import os
from termcolor import colored
from attention import group_query_attention, group_query_attention_fa, group_query_attention_factorized_v_only, group_query_attention_factorized
import time
import socket

def trace_handler(prof: torch.profiler.profile, dir_name="torch_profile_output",
                  worker_name = None, use_gzip: bool = False,
                  file_prefix="prefilling", device="cuda:0"):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise RuntimeError("Can't create directory: " + dir_name) from e
    if not worker_name:
        worker_name = f"{socket.gethostname()}_{os.getpid()}"
    # Use nanosecond here to avoid naming clash when exporting the trace
    timestamp = time.time_ns()
    file_name = f"{file_prefix}.{worker_name}.{timestamp}.pt.trace.json"
    if use_gzip:
        file_name = file_name + ".gz"
    prof.export_chrome_trace(os.path.join(dir_name, file_name))
    # Construct the memory timeline file.
    # !!! This does not work for graph cache !!!
    html_name = f"{file_prefix}.{worker_name}.{timestamp}.html"
    prof.export_memory_timeline(os.path.join(dir_name, html_name), device=device)

def torch_profile_attention_implementations(
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    kv_len=65536,  # 64K sequence length
    k_rank=384,
    v_rank=384,
    dtype=torch.bfloat16,
    device="cuda",
    output_dir="torch_profile_output"
):
    """
    Profile different attention implementations using PyTorch profiler.
    
    Args:
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        kv_len: Length of key/value sequence
        k_rank: Rank for factorized key matrices
        v_rank: Rank for factorized value matrices
        dtype: Data type for tensors
        device: Device to run on
        output_dir: Directory to save profiling results
    """
    import torch.profiler
    from torch.profiler import profile, record_function, ProfilerActivity
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Fixed parameters
    bsz = 1
    q_len = 1
    
    # Create input tensors
    q = torch.randn(bsz, num_heads, q_len, head_dim).to(dtype).to(device)
    k = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    v = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    
    # Create factorized matrices
    v_A = torch.randn(bsz, kv_len, v_rank).to(dtype).to(device)
    v_B = torch.randn(bsz, num_kv_heads, v_rank, head_dim).to(dtype).to(device)
    
    k_A = torch.randn(bsz, kv_len, k_rank).to(dtype).to(device)
    k_B = torch.randn(bsz, num_kv_heads, k_rank, head_dim).to(dtype).to(device)

    group_query_attention_factorized_compiled = torch.compile(group_query_attention_factorized)

    # Warmup
    for _ in range(5):
        group_query_attention(q, k, v)
        group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        group_query_attention_factorized_v_only(q, k, v_A, v_B, use_factorized=True)
        group_query_attention_factorized(q, k_A, k_B, v_A, v_B)
        group_query_attention_factorized_compiled(q, k_A, k_B, v_A, v_B)
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
        
        # Factorized V-only attention
        with record_function("Factorized V-only Attention"):
            group_query_attention_factorized_v_only(q, k, v_A, v_B, use_factorized=True)
            
        torch.cuda.synchronize()
        
        # Fully factorized attention
        with record_function("Fully Factorized Attention"):
            group_query_attention_factorized(q, k_A, k_B, v_A, v_B)

        torch.cuda.synchronize()

    # Print profiling results
    print("\n=== Profiling Results ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
    ))

    # Save trace to file
    timestamp = time.time_ns()
    trace_file = os.path.join(output_dir, f"attention_profile_trace_{num_heads}h_{num_kv_heads}kv_{head_dim}d_{kv_len}len_{timestamp}.json")
    prof.export_chrome_trace(trace_file)
    print(f"Trace saved to: {trace_file}")

def benchmark_attention_implementations(
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    kv_len=65536,
    k_rank=384,
    v_rank=384,
):
    """
    Benchmark different attention implementations and compare their latency.
    
    Args:
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        kv_len: Length of key/value sequence
        k_rank: Rank for factorized key matrices
        v_rank: Rank for factorized value matrices
    """
    bsz = 1
    q_len = 1
    device = "cuda"
    dtype = torch.bfloat16
    
    # Create random tensors for Q, K, V
    q = torch.randn(bsz, num_heads, q_len, head_dim).to(dtype).to(device)
    k = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    v = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    
    # Create factorized value matrices
    v_A = torch.randn(bsz, kv_len, v_rank).to(dtype).to(device)
    v_B = torch.randn(bsz, num_kv_heads, v_rank, head_dim).to(dtype).to(device)
    
    # Create factorized key matrices
    k_A = torch.randn(bsz, kv_len, k_rank).to(dtype).to(device)
    k_B = torch.randn(bsz, num_kv_heads, k_rank, head_dim).to(dtype).to(device)
    
    group_query_attention_factorized_compiled = torch.compile(group_query_attention_factorized)

    # Warmup
    for _ in range(5):
        _ = group_query_attention(q, k, v)
        _ = group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        _ = group_query_attention_factorized_v_only(q, k, v_A, v_B, use_factorized=True)
        _ = group_query_attention_factorized(q, k_A, k_B, v_A, v_B)
        _ = group_query_attention_factorized_compiled(q, k_A, k_B, v_A, v_B)
    torch.cuda.synchronize()
    
    # Time the standard implementation
    std_times = []
    for _ in range(10):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = group_query_attention(q, k, v)
        end_time.record()
        torch.cuda.synchronize()
        std_times.append(start_time.elapsed_time(end_time))
    
    # Time the Flash Attention implementation
    fa_times = []
    for _ in range(10):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        end_time.record()
        torch.cuda.synchronize()
        fa_times.append(start_time.elapsed_time(end_time))
    
    # Time the factorized V-only implementation
    fv_times = []
    for _ in range(10):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = group_query_attention_factorized_v_only(q, k, v_A, v_B, use_factorized=True)
        end_time.record()
        torch.cuda.synchronize()
        fv_times.append(start_time.elapsed_time(end_time))
    
    # Time the fully factorized implementation
    ff_times = []
    for _ in range(10):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = group_query_attention_factorized(q, k_A, k_B, v_A, v_B)
        end_time.record()
        torch.cuda.synchronize()
        ff_times.append(start_time.elapsed_time(end_time))
    
    # Time the fully factorized implementation (compiled)
    ff_times_compiled = []
    for _ in range(10):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = group_query_attention_factorized_compiled(q, k_A, k_B, v_A, v_B)
        end_time.record()
        torch.cuda.synchronize()
        ff_times_compiled.append(start_time.elapsed_time(end_time))
    
    # Calculate average times
    avg_std_time = sum(std_times) / len(std_times)
    avg_fa_time = sum(fa_times) / len(fa_times)
    avg_fv_time = sum(fv_times) / len(fv_times)
    avg_ff_time = sum(ff_times) / len(ff_times)
    avg_ff_time_compiled = sum(ff_times_compiled) / len(ff_times_compiled)
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Standard Attention: {avg_std_time:.4f} ms")
    print(f"Flash Attention: {avg_fa_time:.4f} ms (Speedup: {avg_std_time/avg_fa_time:.2f}x)")
    print(f"Factorized V-only: {avg_fv_time:.4f} ms (Speedup: {avg_std_time/avg_fv_time:.2f}x)")
    print(f"Fully Factorized: {avg_ff_time:.4f} ms (Speedup: {avg_std_time/avg_ff_time:.2f}x)")
    print(f"Fully Factorized (compiled): {avg_ff_time_compiled:.4f} ms (Speedup: {avg_std_time/avg_ff_time_compiled:.2f}x)")
    # Print speedups relative to Flash Attention
    print("\n=== Relative to Flash Attention ===")
    print(f"Factorized V-only vs Flash: {avg_fa_time/avg_fv_time:.2f}x")
    print(f"Fully Factorized vs Flash: {avg_fa_time/avg_ff_time:.2f}x")
    print(f"Fully Factorized (compiled) vs Flash: {avg_fa_time/avg_ff_time_compiled:.2f}x")
    # Print rank information
    print("\n=== Rank Information ===")
    print(f"Per-layer K rank: {k_rank} (Compression Rate: {total_rank/k_rank:.2f}x)")
    print(f"Per-layer V rank: {v_rank} (Compression Rate: {total_rank/v_rank:.2f}x)")
    print(f"KV length: {kv_len}")

def test_torch_compile_group_query_attention_factorized():
    group_query_attention_factorized_compiled = torch.compile(group_query_attention_factorized)
    q = torch.randn(1, 32, 1, 128).to("cuda")
    k_A = torch.randn(1, 65536, 384).to("cuda")
    k_B = torch.randn(1, 8, 384, 128).to("cuda")
    v_A = torch.randn(1, 65536, 384).to("cuda")
    v_B = torch.randn(1, 8, 384, 128).to("cuda")

    output = group_query_attention_factorized(q, k_A, k_B, v_A, v_B)
    output_compiled = group_query_attention_factorized_compiled(q, k_A, k_B, v_A, v_B)

    assert torch.allclose(output, output_compiled, atol=1e-4, rtol=1e-4), "Output is not identical"
    print("Compiled success output is identical")

if __name__ == "__main__":
    # Default configuration
    import argparse
    
    # Default configuration
    parser = argparse.ArgumentParser(description='Benchmark different attention implementations')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of query heads')
    parser.add_argument('--num_kv_heads', type=int, default=8, help='Number of key/value heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Dimension of each head')
    parser.add_argument('--kv_len', type=int, default=128*1024, help='Length of key/value sequence')
    parser.add_argument('--k_rank', type=int, default=384, help='Rank for factorized key matrices')
    parser.add_argument('--v_rank', type=int, default=384, help='Rank for factorized value matrices')
    parser.add_argument('--profile', action='store_true', help='Run profiling')
    
    args = parser.parse_args()
    
    # Calculate total rank based on arguments
    total_rank = args.num_kv_heads * args.head_dim
    
    test_torch_compile_group_query_attention_factorized()

    # Run benchmark
    benchmark_attention_implementations(
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        kv_len=args.kv_len,
        k_rank=args.k_rank,
        v_rank=args.v_rank
    )
    
    # Run profiling if requested
    if args.profile:
        torch_profile_attention_implementations(
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            kv_len=args.kv_len,
            k_rank=args.k_rank,
            v_rank=args.v_rank
        )
    
    
    