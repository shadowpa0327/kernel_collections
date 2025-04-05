import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from functools import partial
import os
from termcolor import colored
from test_attention import group_query_attention, group_query_attention_fa, group_query_attention_factorized
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

if __name__ == "__main__":
    # Default configuration
    #profile_group_query_attention(kv_len=32768)

    bsz = 1
    num_heads = 32
    num_kv_heads = 8
    q_len = 1
    kv_len = 128*1024*2
    head_dim = 128
    rank = 128 # Rank for factorized value matrices

    # Create random tensors for Q, K, V
    q = torch.randn(bsz, num_heads, q_len, head_dim).to(torch.bfloat16).cuda()
    k = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(torch.bfloat16).cuda()
    v = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(torch.bfloat16).cuda()
    
    
    # Create factorized value matrices
    v_A = torch.randn(bsz, kv_len, rank).to(torch.bfloat16).cuda()
    v_B = torch.randn(bsz, num_kv_heads, rank, head_dim).to(torch.bfloat16).cuda()
    
    # Prepare tensors for Flash Attention
    #k_for_fa = k.transpose(1, 2)
    #v_for_fa = v.transpose(1, 2)

    # Warmup
    for _ in range(5):
        _ = group_query_attention(q, k, v)
        _ = group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        _ = group_query_attention_factorized(q, k, v_A, v_B, use_factorized=True)
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
        output_fa = group_query_attention_fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        end_time.record()
        torch.cuda.synchronize()
        fa_times.append(start_time.elapsed_time(end_time))
    
    # Time the factorized implementation
    factorized_times = []
    for _ in range(10):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output_factorized = group_query_attention_factorized(q, k, v_A, v_B, use_factorized=True)
        end_time.record()
        torch.cuda.synchronize()
        factorized_times.append(start_time.elapsed_time(end_time))
    
    
    # Calculate average times
    avg_std_time = sum(std_times) / len(std_times)
    avg_fa_time = sum(fa_times) / len(fa_times)
    avg_factorized_time = sum(factorized_times) / len(factorized_times)
    
    # Print results
    print(f"Standard implementation time: {avg_std_time:.2f} ms")
    print(f"Flash Attention implementation time: {avg_fa_time:.2f} ms")
    print(f"Factorized implementation time: {avg_factorized_time:.2f} ms")
    
    # Print speedups
    print(f"Flash Attention speedup: {avg_std_time/avg_fa_time:.2f}x")
    print(f"Factorized speedup: {avg_std_time/avg_factorized_time:.2f}x")
    
    
    