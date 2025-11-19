import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
from models.attention import (
    group_query_attention_fa,
    xKV_Attention,
)

import time
import socket
import json
from datetime import datetime

def _export_profiler_trace(prof, output_dir, *, file_prefix="normal_attention", device="cuda:0", use_gzip=False):
    os.makedirs(output_dir, exist_ok=True)
    worker_name = f"{socket.gethostname()}_{os.getpid()}"
    timestamp = time.time_ns()
    trace_path = os.path.join(output_dir, f"{file_prefix}.{worker_name}.{timestamp}.pt.trace.json")
    if use_gzip:
        trace_path = trace_path + ".gz"
    prof.export_chrome_trace(trace_path)
    timeline_path = os.path.join(output_dir, f"{file_prefix}.{worker_name}.{timestamp}.html")
    try:
        prof.export_memory_timeline(timeline_path, device=device)
    except Exception:
        timeline_path = None
    return {"trace_path": trace_path, "memory_timeline_path": timeline_path}

@torch.inference_mode()
def profile_normal_attention(
    *,
    mode: str = "fa",
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    kv_len: int = 65536,
    k_rank: int = 384,
    v_rank: int = 384,
    batch_size: int = 1,
    group_size: int = 4,
    warmup: int = 5,
    iters: int = 5,
    profile_dir: str = "torch_profile_output",
    profile_prefix: str | None = None,
    profiler_activities=None,
    use_gzip: bool = False,
):
    """Profile attention implementations using PyTorch profiler."""
    from torch.profiler import ProfilerActivity, profile, record_function

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    q_len = 1

    torch.cuda.set_device(device)

    # Create input tensors
    q = torch.randn(batch_size, num_heads, q_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)

    # Create factorized matrices
    k_A = torch.randn(batch_size, kv_len, k_rank, device=device, dtype=dtype)
    k_B = torch.randn(batch_size, num_kv_heads, k_rank, head_dim, device=device, dtype=dtype)
    v_A = torch.randn(batch_size, kv_len, v_rank, device=device, dtype=dtype)
    v_B = torch.randn(batch_size, num_kv_heads, v_rank, head_dim, device=device, dtype=dtype)

    # Select attention function based on mode
    if mode == "fa":
        attn_fn = lambda: group_query_attention_fa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
    elif mode == "xkv_no_sparse":
        attn_fn = lambda: xKV_Attention(q, k_A, k_B, v_A, v_B)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    activities = profiler_activities or [ProfilerActivity.CPU]
    if all(act is not ProfilerActivity.CUDA for act in activities):
        activities.append(ProfilerActivity.CUDA)

    # Warmup
    for _ in range(max(warmup, 0)):
        attn_fn()
    torch.cuda.synchronize(device)

    # Profile
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(max(iters, 1)):
            with record_function(f"normal_attention_{mode}"):
                attn_fn()
            torch.cuda.synchronize(device)

    return _export_profiler_trace(
        prof,
        output_dir=profile_dir,
        file_prefix=profile_prefix or f"{mode}_normal_attention",
        device=str(device),
        use_gzip=use_gzip,
    )

@torch.inference_mode()
def benchmark_normal_attention(
    *,
    mode: str = "fa",
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    kv_len: int = 65536,
    k_rank: int = 384,
    v_rank: int = 384,
    batch_size: int = 1,
    group_size: int = 4,
    warmup: int = 10,
    iters: int = 50,
):
    """Benchmark attention implementations.

    Returns a dict of timings and configuration.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    q_len = 1

    # Create input tensors
    q = torch.randn(batch_size, num_heads, q_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)

    # Create factorized matrices
    k_A = torch.randn(batch_size, kv_len, k_rank, device=device, dtype=dtype)
    k_B = torch.randn(batch_size, num_kv_heads, k_rank, head_dim, device=device, dtype=dtype)
    v_A = torch.randn(batch_size, kv_len, v_rank, device=device, dtype=dtype)
    v_B = torch.randn(batch_size, num_kv_heads, v_rank, head_dim, device=device, dtype=dtype)

    # Select attention function based on mode
    if mode == "fa":
        attn_fn = lambda: group_query_attention_fa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
    elif mode == "xkv_no_sparse":
        attn_fn = lambda: xKV_Attention(q, k_A, k_B, v_A, v_B)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Warmup
    for _ in range(max(warmup, 0)):
        _ = attn_fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(max(iters, 1)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        attn_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = float(sum(times) / len(times))

    return {
        "mode": mode,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_len": kv_len,
        "k_rank": k_rank,
        "v_rank": v_rank,
        "batch_size": batch_size,
        "group_size": group_size,
        "avg_ms": avg_ms,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("normal_attention benchmark")
    parser.add_argument("--mode", type=str, default="fa", choices=["fa", "xkv_no_sparse"])
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--kv_len", type=int, default=65536)
    parser.add_argument("--k_rank", type=int, default=384)
    parser.add_argument("--v_rank", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profile", action="store_true", help="Use torch.profiler instead of timing loop")
    parser.add_argument("--profile_dir", type=str, default="torch_profile_output", help="Directory to store profiler traces")

    args = parser.parse_args()

    if args.profile:
        profile_paths = profile_normal_attention(
            mode=args.mode,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            kv_len=args.kv_len,
            k_rank=args.k_rank,
            v_rank=args.v_rank,
            batch_size=args.batch_size,
            group_size=args.group_size,
            warmup=args.warmup,
            iters=args.iters,
            profile_dir=args.profile_dir,
        )
        print(json.dumps({"profile": profile_paths}, ensure_ascii=False))
    else:
        res = benchmark_normal_attention(
            mode=args.mode,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            kv_len=args.kv_len,
            k_rank=args.k_rank,
            v_rank=args.v_rank,
            batch_size=args.batch_size,
            group_size=args.group_size,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(json.dumps(res, ensure_ascii=False))