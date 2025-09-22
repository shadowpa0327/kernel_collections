import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from functools import partial
import os
from termcolor import colored
from attention import (
    group_query_attention, 
    group_query_attention_fa, 
    group_query_attention_factorized_v_only, 
    group_query_attention_factorized,
    group_query_attention_factorized_RoPE,
    group_query_attention_sdpa
)

from flash_recontruction import (
    gqa_xKV_no_pe,
    gqa_xKV_no_pe_v2,
    gqa_xKV_no_pe_k_only
)

import time
import socket
import pandas as pd
# show all columns
pd.set_option('display.max_columns', None)

# (optional) expand column width so values aren’t truncated
pd.set_option('display.max_colwidth', None)
from tqdm import tqdm
from datetime import datetime

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
    batch_size=1,
    dtype=torch.float16,
    device="cuda",
    output_dir="benchmark_results"
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
        batch_size: Batch size for the input tensors
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
    q_len = 1
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, q_len, head_dim).to(dtype).to(device)
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    q_reshaped = q.reshape(batch_size, num_heads*q_len, head_dim)
    # Create factorized matrices
    v_A = torch.randn(batch_size, kv_len, v_rank).to(dtype).to(device)
    v_B = torch.randn(batch_size, num_kv_heads, v_rank, head_dim).to(dtype).to(device)
    
    k_A = torch.randn(batch_size, kv_len, k_rank).to(dtype).to(device)
    k_B = torch.randn(batch_size, num_kv_heads, k_rank, head_dim).to(dtype).to(device)

    group_query_attention_factorized_compiled = torch.compile(group_query_attention_factorized)
    group_query_attention_factorized_v_only_compiled = torch.compile(partial(group_query_attention_factorized_v_only))
    group_query_attention_factorized_RoPE_compiled = torch.compile(group_query_attention_factorized_RoPE)
    group_query_attention_original_compiled = torch.compile(group_query_attention)

    # meta data for fused attention
    o = torch.empty(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    num_kv_splits = torch.tensor([128], dtype=torch.int32, device=device)
    max_kv_splits = 128

    # Warmup
    for _ in range(5):
        group_query_attention(q, k, v)
        group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        group_query_attention_factorized_v_only(q, k, v_A, v_B)
        group_query_attention_factorized(q, k_A, k_B, v_A, v_B)
        group_query_attention_factorized_compiled(q, k_A, k_B, v_A, v_B)
        group_query_attention_factorized_v_only_compiled(q, k, v_A, v_B)
        group_query_attention_factorized_RoPE(q, k_A, k_B, v_A, v_B)
        group_query_attention_factorized_RoPE_compiled(q, k_A, k_B, v_A, v_B)
        group_query_attention_original_compiled(q, k, v)
        _ = gqa_xKV_no_pe(q_reshaped, k_A, k_B, v_A, v_B, o, num_kv_splits, max_kv_splits)
        _ = gqa_xKV_no_pe_v2(q_reshaped, k_A, k_B, v_A, v_B, num_kv_splits, max_kv_splits)
        _ = gqa_xKV_no_pe_k_only(q_reshaped, k_A, k_B, v.transpose(1, 2), o, num_kv_splits, max_kv_splits)
        _ = group_query_attention_sdpa(q, k, v)
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

        # Standard attention (compiled)
        with record_function("Standard Attention (compiled)"):
            group_query_attention_original_compiled(q, k, v)

        torch.cuda.synchronize()

        # Flash attention
        with record_function("Flash Attention"):
            group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        
        torch.cuda.synchronize()
        
        # # Factorized V-only attention
        # with record_function("Factorized V-only Attention"):
        #     group_query_attention_factorized_v_only(q, k, v_A, v_B)
            
        # torch.cuda.synchronize()
        
        # # Fully factorized attention
        # with record_function("Fully Factorized Attention"):
        #     group_query_attention_factorized(q, k_A, k_B, v_A, v_B)
    
        # torch.cuda.synchronize()

        # Fully factorized attention (compiled)
        with record_function("Fully Factorized Attention (compiled)"):
            group_query_attention_factorized_compiled(q, k_A, k_B, v_A, v_B)

        torch.cuda.synchronize()

        # Factorized V-only attention (compiled)
        with record_function("Factorized V-only Attention (compiled)"):
            group_query_attention_factorized_v_only_compiled(q, k, v_A, v_B)

        torch.cuda.synchronize()

        # Fully factorized attention (compiled) RoPE
        with record_function("Fully Factorized Attention (compiled) RoPE"):
            group_query_attention_factorized_RoPE_compiled(q, k_A, k_B, v_A, v_B)

        torch.cuda.synchronize()

        # Fully factorized attention RoPE
        with record_function("Fully Factorized Attention RoPE"):
            group_query_attention_factorized_RoPE(q, k_A, k_B, v_A, v_B)

        torch.cuda.synchronize()

        # # Fused attention
        # with record_function("Fused Attention"):
        #     _ = gqa_xKV_no_pe(q_reshaped, k_A, k_B, v_A, v_B, o, num_kv_splits, max_kv_splits)

        # torch.cuda.synchronize()

        # # Fused attention v2
        # with record_function("Fused Attention v2"):
        #     _ = gqa_xKV_no_pe_v2(q_reshaped, k_A, k_B, v_A, v_B, num_kv_splits, max_kv_splits)
            
        # torch.cuda.synchronize()

        # # Fused attention k only
        # with record_function("Fused Attention k only"):
        #     _ = gqa_xKV_no_pe_k_only(q_reshaped, k_A, k_B, v.transpose(1, 2), o, num_kv_splits, max_kv_splits)

        # torch.cuda.synchronize()

        # # SDPA
        # with record_function("SDPA"):
        #     _ = group_query_attention_sdpa(q, k, v)

        # torch.cuda.synchronize()
        
    # Print profiling results
    print("\n=== Profiling Results ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
    ))

    # Save trace to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = os.path.join(output_dir, f"attention_profile_{batch_size}bs_{num_heads}h_{num_kv_heads}kv_{head_dim}d_{kv_len}len_k{k_rank}_v{v_rank}_{timestamp}.json")
    prof.export_chrome_trace(trace_file)
    print(f"Trace saved to: {trace_file}")

def benchmark_function(func, name, warmup_iterations=5, measure_iterations=10):
    """
    Warm up and measure the execution time of a function.
    
    Args:
        func: Function to benchmark
        name: Name of the function (for display purposes)
        warmup_iterations: Number of warmup iterations
        measure_iterations: Number of measurement iterations
    
    Returns:
        List of execution times in milliseconds
    """
    print(f"Benchmarking {name}...")
    
    # Warmup phase
    print(f"  Warming up ({warmup_iterations} iterations)...")
    for _ in tqdm(range(warmup_iterations), desc="Warmup", leave=False):
        _ = func()
    torch.cuda.synchronize()
    
    # Measurement phase
    print(f"  Measuring ({measure_iterations} iterations)...")
    times = []
    for _ in tqdm(range(measure_iterations), desc="Measure", leave=False):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        _ = func()
        end_time.record()
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time)
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f} ms")
    return times

def benchmark_attention_implementations(
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    kv_len=65536,
    k_rank=384,
    v_rank=384,
    batch_size=1,
    output_dir="benchmark_results"
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
        batch_size: Batch size for the input tensors
        output_dir: Directory to save benchmark results
    
    Returns:
        DataFrame containing benchmark results
    """
    import pandas as pd
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate total rank
    total_rank = num_kv_heads * head_dim
    
    # Setup information
    setup_info = {
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'kv_len': kv_len,
        'k_rank': k_rank,
        'v_rank': v_rank,
        'batch_size': batch_size,
        'k_compression': total_rank / k_rank,
        'v_compression': total_rank / v_rank
    }
    
    q_len = 1
    device = "cuda"
    dtype = torch.float16
    
    # Create random tensors for Q, K, V
    q = torch.randn(batch_size, num_heads, q_len, head_dim).to(dtype).to(device)
    q_reshaped = q.reshape(batch_size, num_heads*q_len, head_dim)
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
    
    # Create factorized value matrices
    v_A = torch.randn(batch_size, kv_len, v_rank).to(dtype).to(device)
    v_B = torch.randn(batch_size, num_kv_heads, v_rank, head_dim).to(dtype).to(device)
    
    # Create factorized key matrices
    k_A = torch.randn(batch_size, kv_len, k_rank).to(dtype).to(device)
    k_B = torch.randn(batch_size, num_kv_heads, k_rank, head_dim).to(dtype).to(device)
    
    # buffer and metadata for fused attention
    o = torch.empty(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    num_kv_splits = torch.tensor([128] * batch_size, dtype=torch.int32, device=device)
    max_kv_splits = 128

    # Compile functions
    group_query_attention_factorized_compiled = torch.compile(group_query_attention_factorized)
    group_query_attention_factorized_v_only_compiled = torch.compile(group_query_attention_factorized_v_only)
    group_query_attention_factorized_RoPE_compiled = torch.compile(group_query_attention_factorized_RoPE)
    group_query_attention_original_compiled = torch.compile(group_query_attention)

    # Create function callables using partial
    functions = {
        "Standard Attention": partial(group_query_attention, q, k, v),
        "Standard Attention (compiled)": partial(group_query_attention_original_compiled, q, k, v),
        "Flash Attention": partial(group_query_attention_fa, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)),
        #"Factorized V-only": partial(group_query_attention_factorized_v_only, q, k, v_A, v_B),
        #"Fully Factorized": partial(group_query_attention_factorized, q, k_A, k_B, v_A, v_B),
        #"Fully Factorized (compiled)": partial(group_query_attention_factorized_compiled, q, k_A, k_B, v_A, v_B),
        #"Factorized V-only (compiled)": partial(group_query_attention_factorized_v_only_compiled, q, k, v_A, v_B),
        "Fully Factorized RoPE": partial(group_query_attention_factorized_RoPE, q, k_A, k_B, v_A, v_B),
        "Fully Factorized RoPE (compiled)": partial(group_query_attention_factorized_RoPE_compiled, q, k_A, k_B, v_A, v_B),
        #"Fused Attention": partial(gqa_xKV_no_pe, q_reshaped, k_A, k_B, v_A, v_B, o, num_kv_splits, max_kv_splits),
        #"Fused Attention v2": partial(gqa_xKV_no_pe_v2, q_reshaped, k_A, k_B, v_A, v_B, num_kv_splits, max_kv_splits),
        #"Fused Attention k only": partial(gqa_xKV_no_pe_k_only, q_reshaped, k_A, k_B, v.transpose(1, 2), o, num_kv_splits, max_kv_splits),
        #"SDPA": partial(group_query_attention_sdpa, q, k, v)
    }
    
    # Benchmark all functions
    results = {}
    for name, func in functions.items():
        #with torch.inference_mode():
        results[name] = benchmark_function(func, name)
    
    # Calculate average times
    avg_times = {name: sum(times) / len(times) for name, times in results.items()}
    
    # Create pandas DataFrame for results
    df_results = pd.DataFrame({
        'Implementation': list(avg_times.keys()),
        'Average Time (ms)': [avg_times[name] for name in avg_times],
        'Min Time (ms)': [min(results[name]) for name in avg_times],
        'Max Time (ms)': [max(results[name]) for name in avg_times],
        'Std Dev (ms)': [pd.Series(results[name]).std() for name in avg_times]
    })
    
    # Add speedup columns
    #std_time = avg_times["Standard Attention"]
    fa_time = avg_times["Flash Attention"]
    #sdpa_time = avg_times["SDPA"]
    
    #df_results['Speedup vs Standard'] = std_time / df_results['Average Time (ms)']
    df_results['Speedup vs Flash'] = fa_time / df_results['Average Time (ms)']
    #df_results['Speedup vs SDPA'] = sdpa_time / df_results['Average Time (ms)']
    
    # Sort by average time
    df_results = df_results.sort_values('Average Time (ms)')
    
    # Print results
    print("\n=== Benchmark Results ===")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)
    print(df_results)
    
    # Add setup information to each row
    for key, value in setup_info.items():
        df_results[key] = value

    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a descriptive filename
    filename = f"attention_benchmark_{batch_size}bs_{num_heads}h_{num_kv_heads}kv_{head_dim}d_{kv_len}len_k{k_rank}_v{v_rank}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save results to CSV
    df_results.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
    
    # Print rank information
    print("\n=== Setup Information ===")
    for key, value in setup_info.items():
        print(f"{key}: {value}")
    
    # Return the DataFrame for further analysis if needed
    return df_results

if __name__ == "__main__":
    # Default configuration
    import argparse
    
    # Default configuration
    parser = argparse.ArgumentParser(description='Benchmark different attention implementations')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of query heads')
    parser.add_argument('--num_kv_heads', type=int, default=8, help='Number of key/value heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Dimension of each head')
    parser.add_argument('--kv_len', type=int, default=64*1024, help='Length of key/value sequence')
    parser.add_argument('--k_rank', type=int, default=384, help='Rank for factorized key matrices')
    parser.add_argument('--v_rank', type=int, default=512, help='Rank for factorized value matrices')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for the input tensors')
    parser.add_argument('--profile', action='store_true', help='Run profiling')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Directory to save benchmark results')
    
    args = parser.parse_args()
    
    # Calculate total rank based on arguments
    total_rank = args.num_kv_heads * args.head_dim

    # Run benchmark
    df_results = benchmark_attention_implementations(
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        kv_len=args.kv_len,
        k_rank=args.k_rank,
        v_rank=args.v_rank,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # Run profiling if requested
    if args.profile:
        torch_profile_attention_implementations(
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            kv_len=args.kv_len,
            k_rank=args.k_rank,
            v_rank=args.v_rank,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    
    
    