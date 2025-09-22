from flash_decoding.deepseek_mla import mla_decode_attention_fwd, mla_decode_attention_fwd_pd_sep, mla_decode_attention_fwd_pd_sep_xKV
import torch
import time
from tqdm import tqdm
import pandas as pd
import os
from datetime import datetime
from functools import partial
import argparse

def decode_attention_reference(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_buffer,           # shape: [batch, kv_len, num_kv_heads, head_dim]
    v_buffer,           # shape: [batch, kv_len, num_kv_heads, head_dim]
    sm_scale,           # float: softmax scaling factor
):
    # Step 1: Store the size 
    batch, num_q_heads, head_dim = q.shape
    batch, kv_len, num_kv_heads, head_dim = k_buffer.shape
    
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
    #assert batch== 1
    #k_buffer = k_buffer.unsqueeze(0) # [batch, kv_len, num_kv_heads, head_dim]
    k_buffer = k_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]
    
    #v_buffer = v_buffer.unsqueeze(0) # [batch, kv_len, num_kv_heads, head_dim]
    v_buffer = v_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]

    q = q.view(batch, num_kv_heads, num_q_heads_per_kv_group, head_dim)

    # Step 2: Compute q@K
    qk = torch.matmul(q, k_buffer.transpose(-1, -2))  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    # Step 3: Compute softmax
    qk = qk / sm_scale
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(qk.dtype)  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    # Step 4: Compute o = softmax(q@K) @ V
    o = torch.matmul(qk, v_buffer) # [batch, num_kv_heads, num_q_heads_per_kv_group, head_dim]
    o = o.view(batch, num_q_heads, head_dim)    
    return o


def benchmark_function(func, name, warmup_iterations=10, measure_iterations=100):
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


def measure_time(func, warmup=10, repeat=100):
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to measure
        args: Arguments to pass to the function
        warmup: Number of warmup iterations
        repeat: Number of measurement iterations
    
    Returns:
        Average execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(repeat):
        func()
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) * 1000 / repeat  # Convert to ms

def benchmark_kernel_implementations(
    num_q_heads=32,
    num_kv_heads=1,
    head_dim=512,
    prefill_kv_len=1024*128,
    decode_kv_len=128,
    rank_latent=256,
    output_dir="benchmark_results"
):
    """
    Benchmark different MLA decoding kernel implementations and compare their latency.
    
    Args:
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        prefill_kv_len: Length of prefill key/value sequence
        decode_kv_len: Length of decode key/value sequence
        rank_latent: Rank for factorized matrices
        output_dir: Directory to save benchmark results
    
    Returns:
        DataFrame containing benchmark results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate total rank
    total_rank = num_kv_heads * head_dim
    
    # Setup information
    setup_info = {
        'num_q_heads': num_q_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'prefill_kv_len': prefill_kv_len,
        'decode_kv_len': decode_kv_len,
        'rank_latent': rank_latent,
        'compression': total_rank / rank_latent
    }
    
    batch = 8
    device = "cuda"
    dtype = torch.float16
    
    # Create random tensors
    q = (torch.rand(batch, num_q_heads, head_dim, dtype=dtype, device=device) - 0.5)
    k_A_buffer = (torch.rand(batch, prefill_kv_len, rank_latent, dtype=dtype, device=device) - 0.5)
    k_B_buffer = (torch.rand(batch, num_kv_heads, rank_latent, head_dim, dtype=dtype, device=device) - 0.5)
    v_A_buffer = (torch.rand(batch, prefill_kv_len, rank_latent, dtype=dtype, device=device) - 0.5)
    v_B_buffer = (torch.rand(batch, num_kv_heads, rank_latent, head_dim, dtype=dtype, device=device) - 0.5)
    k_buffer_decoded = (torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=dtype, device=device) - 0.5)*20
    v_buffer_decoded = (torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=dtype, device=device) - 0.5)*20
    
    # Compute prefill buffers
    k_buffer_prefill = torch.matmul(k_A_buffer.unsqueeze(1), k_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
    v_buffer_prefill = torch.matmul(v_A_buffer.unsqueeze(1), v_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
    
    # Transpose buffers to match expected format
    k_buffer_prefill = k_buffer_prefill.transpose(1, 2)
    v_buffer_prefill = v_buffer_prefill.transpose(1, 2)
    k_buffer_decoded = k_buffer_decoded.transpose(1, 2)
    v_buffer_decoded = v_buffer_decoded.transpose(1, 2)
    
    # Concatenate buffers
    k_buffer = torch.cat([k_buffer_prefill, k_buffer_decoded], dim=1)
    v_buffer = torch.cat([v_buffer_prefill, v_buffer_decoded], dim=1)
    
    sm_scale = 1.0
    
    # Create function callables using partial
    functions = {
        "Regular MLA (Triton)": partial(mla_decode_attention_fwd, q, k_buffer, k_buffer, sm_scale),
        "Separated MLA (Triton)": partial(mla_decode_attention_fwd_pd_sep, q, k_buffer_prefill, k_buffer_prefill, k_buffer_decoded, k_buffer_decoded, sm_scale),
        "xKV MLA (Triton)": partial(mla_decode_attention_fwd_pd_sep_xKV, q, k_A_buffer, k_B_buffer, k_A_buffer, k_B_buffer, k_buffer_decoded, k_buffer_decoded, sm_scale),
        "Eager": partial(decode_attention_reference, q, k_buffer, v_buffer, sm_scale)
    }
    
    # Benchmark all functions
    results = {}
    for name, func in functions.items():
        results[name] = measure_time(func, warmup=50, repeat=500)
    
    # Calculate average times
    #avg_times = {name: sum(times) / len(times) for name, times in results.items()}
    
    # Create pandas DataFrame for results
    df_results = pd.DataFrame({
        'Implementation': list(results.keys()),
        'Average Time (ms)': [results[name] for name in results],
        #'Min Time (ms)': [min(results[name]) for name in avg_times],
        #'Max Time (ms)': [max(results[name]) for name in avg_times],
        #'Std Dev (ms)': [pd.Series(results[name]).std() for name in avg_times]
    })
    
    # Add speedup columns
    ref_time = results["Eager"]
    reg_time = results["Regular MLA (Triton)"]
    
    df_results['Speedup vs Reference'] = ref_time / df_results['Average Time (ms)']
    df_results['Speedup vs Regular'] = reg_time / df_results['Average Time (ms)']
    
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
    filename = f"mla_benchmark_{num_q_heads}h_{num_kv_heads}kv_{head_dim}d_{prefill_kv_len}prefill_{decode_kv_len}decode_{rank_latent}rank_{timestamp}.csv"
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

def test_correctness():
   # Verify correctness
    print("\n=== Correctness Verification ===")
    batch = 1
    num_q_heads = 32
    num_kv_heads = 1
    head_dim = 512
    prefill_kv_len = 1024*128
    decode_kv_len = 128
    rank_latent = 256
    dtype = torch.float16
    
    # Create test tensors
    q = (torch.rand(batch, num_q_heads, head_dim, dtype=dtype, device='cuda') - 0.5)
    k_A_buffer = (torch.rand(batch, prefill_kv_len, rank_latent, dtype=dtype, device='cuda') - 0.5)
    k_B_buffer = (torch.rand(batch, num_kv_heads, rank_latent, head_dim, dtype=dtype, device='cuda') - 0.5)
    v_A_buffer = (torch.rand(batch, prefill_kv_len, rank_latent, dtype=dtype, device='cuda') - 0.5)
    v_B_buffer = (torch.rand(batch, num_kv_heads, rank_latent, head_dim, dtype=dtype, device='cuda') - 0.5)
    k_buffer_decoded = (torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=dtype, device='cuda') - 0.5)*20
    v_buffer_decoded = (torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=dtype, device='cuda') - 0.5)*20
    
    # Compute prefill buffers
    k_buffer_prefill = torch.matmul(k_A_buffer.unsqueeze(1), k_B_buffer)
    v_buffer_prefill = torch.matmul(v_A_buffer.unsqueeze(1), v_B_buffer)
    
    # Transpose buffers
    k_buffer_prefill = k_buffer_prefill.transpose(1, 2)
    v_buffer_prefill = v_buffer_prefill.transpose(1, 2)
    k_buffer_decoded = k_buffer_decoded.transpose(1, 2)
    v_buffer_decoded = v_buffer_decoded.transpose(1, 2)
    
    # Concatenate buffers
    k_buffer = torch.cat([k_buffer_prefill, k_buffer_decoded], dim=1)
    v_buffer = torch.cat([v_buffer_prefill, v_buffer_decoded], dim=1)
    
    sm_scale = 1.0
    
    # Run implementations and compare outputs
    o = mla_decode_attention_fwd(q, k_buffer, v_buffer, sm_scale)
    o_sep = mla_decode_attention_fwd_pd_sep(q, k_buffer_prefill, v_buffer_prefill, k_buffer_decoded, v_buffer_decoded, sm_scale)
    o_xkv = mla_decode_attention_fwd_pd_sep_xKV(q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale)
    o_ref = decode_attention_reference(q, k_buffer, v_buffer, sm_scale)
    
    print("Maximum difference between separated and non-separated implementation:", torch.max(torch.abs(o_sep - o)))
    print("Maximum difference between separated and xKV implementation:", torch.max(torch.abs(o_sep - o_xkv)))
    print("Maximum difference between separated and reference implementation:", torch.max(torch.abs(o_sep - o_ref)))
    print("Maximum difference between triton and reference implementation:", torch.max(torch.abs(o - o_ref)))
    
    
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark MLA decoding implementations')
    parser.add_argument('--num_q_heads', type=int, default=32, help='Number of query heads')
    parser.add_argument('--num_kv_heads', type=int, default=1, help='Number of key/value heads')
    parser.add_argument('--head_dim', type=int, default=512, help='Dimension of each head')
    parser.add_argument('--prefill_kv_len', type=int, default=1024*128, help='Length of prefill key/value sequence')
    parser.add_argument('--decode_kv_len', type=int, default=128, help='Length of decode key/value sequence')
    parser.add_argument('--rank_latent', type=int, default=256, help='Rank for factorized matrices')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Directory to save benchmark results')
    parser.add_argument('--test_correctness', action='store_true', help='Run correctness verification')
    
    args = parser.parse_args()
    
    # Run benchmark with configured parameters
    df_results = benchmark_kernel_implementations(
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        prefill_kv_len=args.prefill_kv_len,
        decode_kv_len=args.decode_kv_len,
        rank_latent=args.rank_latent,
        output_dir=args.output_dir
    )
    
    # Run correctness test if requested
    if args.test_correctness:
        test_correctness()
    
    
    
    