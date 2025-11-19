import json
import os
import socket
import time
import uuid
import warnings

from flash_attn import flash_attn_with_kvcache
import torch
from transformers import AutoConfig

from models.kv_cache import KV_Cache, ShadowKVCache_xKey_CPU, ShadowKVCache_xKV_CPU
from ops.shadowkv import apply_rotary_pos_emb_single
from models.merge_configs import generate_consecutive_palu_config

try:
    from torch.cuda import nvtx
except ImportError:  # pragma: no cover - NVTX is optional on some builds
    class _NullNvtx:
        @staticmethod
        def range_push(_msg):
            return None

        @staticmethod
        def range_pop():
            return None

    nvtx = _NullNvtx()

def init_xkv_cache(
    *,
    mode: str = "xkey",              # "xkey" or "xkv"
    group_size: int = 4,
    rank_k: int = 64,
    rank_v: int = 96,
    sparse_budget: int = 2048,
    max_length: int = 65536,
    chunk_size: int = 8,
    batch_size: int = 1,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    prefill_len: int = 4096,
    pre_init: bool = True,
):
    """
    Build an xKV cache for benchmarking `xkv_attention` with configurable args.

    Returns (kv_cache, config, merge_config)
    """
    # Fixed device/dtype per request
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    # Build config: try HF, else fall back to a local default config to avoid network.
    if model_name is None or (isinstance(model_name, str) and model_name.lower() == "local"):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=4,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
    else:
        try:
            config = AutoConfig.from_pretrained(model_name)
        except Exception:
            from types import SimpleNamespace
            config = SimpleNamespace(
                hidden_size=4096,
                num_hidden_layers=4,
                num_attention_heads=32,
                num_key_value_heads=8,
            )

    # Derive layer range from config when available
    num_layers = getattr(config, "num_hidden_layers", 4)
    merge_config = generate_consecutive_palu_config(
        start_layer=0,
        end_layer=num_layers - 1,
        group_size=group_size,
        rank_k=rank_k,
        rank_v=rank_v,
    )

    if mode.lower() == "xkey":
        kv_cache = ShadowKVCache_xKey_CPU(
            config,
            merge_config,
            max_length=max_length,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            sparse_budget=sparse_budget,
            chunk_size=chunk_size,
        )
    elif mode.lower() == "xkv":
        kv_cache = ShadowKVCache_xKV_CPU(
            config,
            merge_config,
            max_length=max_length,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            sparse_budget=sparse_budget,
            chunk_size=chunk_size,
        )
    elif mode.lower() == "full":
        # Dense full KV cache baseline
        kv_cache = KV_Cache(
            config,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError("mode must be 'xkey' or 'xkv'")

    # Optional pre-initialization so decode works immediately without prefill
    if pre_init:
        if mode.lower() == "xkey":
            _preinit_xkey_cache(kv_cache, config, prefill_len)
        elif mode.lower() == "xkv":
            _preinit_xkv_cache(kv_cache, config, prefill_len)
        elif mode.lower() == "full":
            _preinit_full_cache(kv_cache, config, prefill_len)

        # Move pre-initialized tensors to CUDA for kernels
        if hasattr(kv_cache, "H2D"):
            kv_cache.H2D()

    return kv_cache, config, merge_config


    
    
def xkv_attention(query_states, layer_idx, kv_cache: KV_Cache, cos_sin_cache=None):
    """Run flash attention with either dense, xKey, or xKV cache."""
    # Dense baseline path is short-circuited.
    if isinstance(kv_cache, KV_Cache):
        seqlen = kv_cache.get_kv_len() or kv_cache.max_length
        key_states = kv_cache.k_cache[layer_idx][:, :, :seqlen]
        value_states = kv_cache.v_cache[layer_idx][:, :, :seqlen]
        return flash_attn_with_kvcache(
            q=query_states.transpose(1, 2),
            k_cache=key_states.transpose(1, 2),
            v_cache=value_states.transpose(1, 2),
            causal=True,
        )

    if cos_sin_cache is None:
        raise ValueError("cos_sin_cache must be provided for ShadowKV caches")

    nvtx.range_push("get_retrieval_position_ids")
    position_ids = kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)
    nvtx.range_pop()

    # Default to synchronous gather; upgrade to async for offloaded caches.
    curr_stream = None
    value_stream = None

    if isinstance(kv_cache, ShadowKVCache_xKV_CPU):
        nvtx.range_push("get_value_cache_xKV_cpu")
        value_states = kv_cache.get_value_cache(layer_idx, position_ids, cos_sin_cache)
        nvtx.range_pop()
    else:
        curr_stream = torch.cuda.current_stream()
        value_stream = kv_cache.copy_stream
        nvtx.range_push("get_value_cache_offload_stream")
        with torch.cuda.stream(value_stream):
            value_stream.wait_stream(curr_stream)
            value_states = kv_cache.get_value_cache(layer_idx, position_ids)
        nvtx.range_pop()

    nvtx.range_push("get_key_cache")
    key_states = kv_cache.get_key_cache(
        layer_idx=layer_idx,
        position_ids=position_ids,
        rope_func=apply_rotary_pos_emb_single,
        cos_sin_cache=cos_sin_cache,
    )
    nvtx.range_pop()

    if value_stream is not None:
        nvtx.range_push("wait_get_value_stream")
        curr_stream.wait_stream(value_stream)
        nvtx.range_pop()

    nvtx.range_push("flash_attn_with_kvcache")
    hidden_states = flash_attn_with_kvcache(
        q=query_states.transpose(1, 2),
        k_cache=key_states.transpose(1, 2),
        v_cache=value_states.transpose(1, 2),
        causal=True,
    )
    nvtx.range_pop()

    return hidden_states


def _maybe_compile_attention(kv_cache, cos_sin_cache, layer_idx, *, compile_mode="reduce-overhead"):
    """Return a callable that executes xkv_attention and is torch.compile'd when possible."""

    def _decode(query_states):
        return xkv_attention(query_states, layer_idx, kv_cache, cos_sin_cache)

    #compile_fn = getattr(torch, "compile", None)
    compile_fn = None
    if compile_fn is None:
        return _decode

    try:
        return compile_fn(_decode)
    except Exception as exc:  # pragma: no cover - backend specific failures
        warnings.warn(f"torch.compile failed for xkv_attention, falling back to eager execution: {exc}")
        return _decode


def _export_profiler_trace(prof, output_dir, *, file_prefix="xkv_attention", device="cuda:0", use_gzip=False):
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
def profile_xkv_attention(
    *,
    mode: str = "xkey",
    group_size: int = 4,
    rank_k: int = 256,
    rank_v: int = 384,
    sparse_budget: int = 2048,
    max_length: int = 65536,
    chunk_size: int = 8,
    batch_size: int = 1,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    layer_idx: int = 0,
    prefill_len: int = 65536,
    warmup: int = 5,
    iters: int = 5,
    profile_dir: str = "torch_profile_output",
    profile_prefix: str | None = None,
    profiler_activities=None,
    use_gzip: bool = False,
):
    from torch.profiler import ProfilerActivity, profile, record_function

    kv_cache, config, _ = init_xkv_cache(
        mode=mode,
        group_size=group_size,
        rank_k=rank_k,
        rank_v=rank_v,
        sparse_budget=sparse_budget,
        max_length=max_length,
        chunk_size=chunk_size,
        batch_size=batch_size,
        model_name=model_name,
        prefill_len=prefill_len,
        pre_init=True,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    head_dim = config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads

    torch.cuda.set_device(device)

    cos_sin_cache = torch.randn((max_length, head_dim), device=device, dtype=dtype)
    q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)

    decode_fn = _maybe_compile_attention(kv_cache, cos_sin_cache, layer_idx)

    activities = profiler_activities or [ProfilerActivity.CPU]
    if all(act is not ProfilerActivity.CUDA for act in activities):
        activities.append(ProfilerActivity.CUDA)

    for _ in range(max(warmup, 0)):
        decode_fn(q)
    torch.cuda.synchronize(device)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(max(iters, 1)):
            with record_function("xkv_attention"):
                decode_fn(q)
            torch.cuda.synchronize(device)

    return _export_profiler_trace(
        prof,
        output_dir=profile_dir,
        file_prefix=profile_prefix or f"{mode}_{uuid.uuid4().hex[:8]}",
        device=str(device),
        use_gzip=use_gzip,
    )


@torch.inference_mode()
def _preinit_full_cache(kv_cache: KV_Cache, config, prefill_len: int):
    """Pre-initialize full KV cache with random content and set kv_offset."""
    # Clamp prefill_len to allocated max_length
    prefill_len = min(prefill_len, kv_cache.max_length)
    # k_cache/v_cache were already allocated with random values on init
    kv_cache.kv_offset = prefill_len

@torch.inference_mode()
def _preinit_xkey_cache(kv_cache: ShadowKVCache_xKey_CPU, config, prefill_len: int):
    """Pre-initialize xKey cache state so decode can run without explicit prefill."""
    device = torch.device("cuda:0")
    dtype = kv_cache.dtype
    bsz = kv_cache.batch_size
    num_kv = config.num_key_value_heads
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads

    max_ctx_chunks = max(prefill_len // kv_cache.chunk_size, 1)
    kv_cache.max_ctx_chunks_len = max_ctx_chunks * kv_cache.chunk_size

    chunks = max_ctx_chunks - kv_cache.local_chunk
    chunks -= chunks % 8
    chunks = max(chunks, 8)  # ensure positive and aligned
    kv_cache.chunks = chunks
    kv_cache.prefill_local = prefill_len - chunks * kv_cache.chunk_size
    kv_cache.prefill = prefill_len

    # Set sparse regions and kernel parameters
    outlier_chunk = kv_cache.outlier_chunk
    if outlier_chunk >= chunks:
        outlier_chunk = max(chunks // 2, 1)
    kv_cache.sparse_start = kv_cache.prefill_local + outlier_chunk * kv_cache.chunk_size
    kv_cache.sparse_end = kv_cache.sparse_start + kv_cache.sparse_budget
    kv_cache.kernel_offset = kv_cache.sparse_start * head_dim
    kv_cache.kernel_stride = kv_cache.v_cache_buffer.shape[-2] * head_dim

    # Randomly initialize U, SV for keys (post-merging shapes)
    kv_cache.U = torch.randn(kv_cache.num_layers // kv_cache.group_size, bsz, prefill_len, kv_cache.rank_k, device="cuda:0", dtype=dtype)
    kv_cache.SV = torch.randn(kv_cache.num_layers, bsz, num_kv, head_dim, kv_cache.rank_k, device="cuda:0", dtype=dtype)

    # Landmarks and indices
    num_landmarks = max(chunks - outlier_chunk, 1)
    kv_cache.k_landmark = torch.randn(kv_cache.num_layers, bsz, num_kv, num_landmarks, head_dim, device="cuda:0", dtype=dtype)
    kv_cache.k_landmark_idx = torch.randint(low=0, high=chunks, size=(kv_cache.num_layers, bsz, num_kv, num_landmarks), device="cuda:0", dtype=torch.long)

    # Fused gemm buffers for retrieval
    num_kv_groups = num_heads // num_kv
    tiles = (num_landmarks + 256 - 1) // 256
    kv_cache.gemm_o = torch.randn(bsz, num_kv, num_kv_groups, num_landmarks, device="cuda:0", dtype=torch.bfloat16).contiguous()
    kv_cache.softmax_o = torch.randn(bsz, num_kv, num_kv_groups, num_landmarks, device="cuda:0", dtype=torch.bfloat16).contiguous()
    kv_cache.norm = torch.randn(bsz*num_kv, num_kv_groups, tiles, device="cuda:0", dtype=torch.float).contiguous()
    kv_cache.sum = torch.randn(bsz*num_kv, num_kv_groups, tiles, device="cuda:0", dtype=torch.float).contiguous()

    # Position ids buffer already allocated; ensure -1 init
    kv_cache.position_ids.fill_(-1)

    # Random fill caches for safe reads
    kv_cache.v_cache_cpu.normal_()
    kv_cache.k_cache_buffer.normal_()
    kv_cache.v_cache_buffer.normal_()


@torch.inference_mode()
def _preinit_xkv_cache(kv_cache: ShadowKVCache_xKV_CPU, config, prefill_len: int):
    """Pre-initialize xKV cache state so decode can run without explicit prefill."""
    device = torch.device("cuda:0")
    dtype = kv_cache.dtype
    bsz = kv_cache.batch_size
    num_kv = config.num_key_value_heads
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads

    max_ctx_chunks = max(prefill_len // kv_cache.chunk_size, 1)
    kv_cache.max_ctx_chunks_len = max_ctx_chunks * kv_cache.chunk_size

    chunks = max_ctx_chunks - kv_cache.local_chunk
    chunks -= chunks % 8
    chunks = max(chunks, 8)
    kv_cache.chunks = chunks
    kv_cache.prefill_local = prefill_len - chunks * kv_cache.chunk_size
    kv_cache.prefill = prefill_len

    outlier_chunk = kv_cache.outlier_chunk
    if outlier_chunk >= chunks:
        outlier_chunk = max(chunks // 2, 1)
    kv_cache.sparse_start = kv_cache.prefill_local + outlier_chunk * kv_cache.chunk_size
    kv_cache.sparse_end = kv_cache.sparse_start + kv_cache.sparse_budget
    kv_cache.kernel_offset = kv_cache.sparse_start * head_dim
    kv_cache.kernel_stride = kv_cache.v_cache_buffer.shape[-2] * head_dim

    # Randomly initialize decomposed caches for K and V
    kv_cache.U_k = torch.randn(kv_cache.num_layers // kv_cache.group_size, bsz, prefill_len, kv_cache.rank_k, device="cuda:0", dtype=dtype)
    kv_cache.SV_k = torch.randn(kv_cache.num_layers, bsz, num_kv, head_dim, kv_cache.rank_k, device="cuda:0", dtype=dtype)
    kv_cache.U_v = torch.randn(
        kv_cache.num_layers // kv_cache.group_size, bsz, prefill_len, kv_cache.rank_v, device="cuda:0", dtype=dtype
    )
    # For values, kernel expects b shape [bsz, heads, head_dim, rank_v]
    kv_cache.SV_v = torch.randn(
        kv_cache.num_layers, bsz, num_kv, head_dim, kv_cache.rank_v, device="cuda:0", dtype=dtype
    )

    # Landmarks and indices
    num_landmarks = max(chunks - outlier_chunk, 1)
    kv_cache.k_landmark = torch.randn(kv_cache.num_layers, bsz, num_kv, num_landmarks, head_dim, device="cuda:0", dtype=dtype)
    kv_cache.k_landmark_idx = torch.randint(low=0, high=chunks, size=(kv_cache.num_layers, bsz, num_kv, num_landmarks), device="cuda:0", dtype=torch.long)

    # Fused gemm buffers for retrieval
    num_kv_groups = num_heads // num_kv
    tiles = (num_landmarks + 256 - 1) // 256
    kv_cache.gemm_o = torch.randn(bsz, num_kv, num_kv_groups, num_landmarks, device="cuda:0", dtype=torch.bfloat16).contiguous()
    kv_cache.softmax_o = torch.randn(bsz, num_kv, num_kv_groups, num_landmarks, device="cuda:0", dtype=torch.bfloat16).contiguous()
    kv_cache.norm = torch.randn(bsz*num_kv, num_kv_groups, tiles, device="cuda:0", dtype=torch.float).contiguous()
    kv_cache.sum = torch.randn(bsz*num_kv, num_kv_groups, tiles, device="cuda:0", dtype=torch.float).contiguous()

    # Sanity checks to catch layout issues early
    assert kv_cache.SV_k.shape[-2:] == (head_dim, kv_cache.rank_k), f"SV_k last dims should be (head_dim, rank_k), got {kv_cache.SV_k.shape[-2:]}"
    assert kv_cache.SV_v.shape[-2:] == (head_dim, kv_cache.rank_v), f"SV_v last dims should be (head_dim, rank_v), got {kv_cache.SV_v.shape[-2:]}"

    # Position ids buffer already allocated; ensure -1 init
    kv_cache.position_ids.fill_(-1)

    # Random fill buffers
    kv_cache.k_cache_buffer.normal_()
    kv_cache.v_cache_buffer.normal_()


@torch.inference_mode()
def benchmark_xkv_attention(
    *,
    mode: str = "xkey",        # "xkey" or "xkv"
    group_size: int = 4,
    rank_k: int = 256,
    rank_v: int = 384,
    sparse_budget: int = 2048,
    max_length: int = 65536,
    chunk_size: int = 8,
    batch_size: int = 1,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    layer_idx: int = 0,
    prefill_len: int = 65536,
    warmup: int = 10,
    iters: int = 50,
):
    """Simple throughput benchmark for current xkv_attention path.

    Returns a dict of timings and shapes.
    """
    kv_cache, config, merge_config = init_xkv_cache(
        mode=mode,
        group_size=group_size,
        rank_k=rank_k,
        rank_v=rank_v,
        sparse_budget=sparse_budget,
        max_length=max_length,
        chunk_size=chunk_size,
        batch_size=batch_size,
        model_name=model_name,
        prefill_len=prefill_len,
        pre_init=True,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    head_dim = config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads

    cos_sin_cache = torch.randn((max_length, head_dim), device=device, dtype=dtype)

    # Cache is pre-initialized; no prefill needed

    # Prepare a decode query (q_len=1)
    q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)

    decode_fn = _maybe_compile_attention(kv_cache, cos_sin_cache, layer_idx)

    # xkv_attention = torch.compile(xkv_attention)

    # Warmup
    for _ in range(max(warmup, 0)):
        _ = decode_fn(q)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(max(iters, 1)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record() 
        decode_fn(q)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # # Print profiling results
    # print("\n=== Profiling Results ===")
    # print(prof.key_averages().table(
    #     sort_by="cuda_time_total",
    #     row_limit=10,
    # ))

    import statistics as stats
    avg_ms = float(sum(times) / len(times))

    return {
        "mode": mode,
        "group_size": group_size,
        "rank_k": rank_k,
        "rank_v": rank_v,
        "sparse_budget": sparse_budget,
        "max_length": max_length,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "prefill_len": prefill_len,
        "avg_ms": avg_ms,
    }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser("xkv_attention benchmark")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "xkey", "xkv"]) 
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--rank_k", type=int, default=64)
    parser.add_argument("--rank_v", type=int, default=96)
    parser.add_argument("--sparse_budget", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=131072)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="local")
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--prefill_len", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="Use torch.profiler instead of timing loop")
    parser.add_argument("--profile_dir", type=str, default="torch_profile_output", help="Directory to store profiler traces")
    args = parser.parse_args()

    if args.profile:
        profile_paths = profile_xkv_attention(
            mode=args.mode,
            group_size=args.group_size,
            rank_k=args.rank_k,
            rank_v=args.rank_v,
            sparse_budget=args.sparse_budget,
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            model_name=args.model_name,
            layer_idx=args.layer_idx,
            prefill_len=args.prefill_len,
            warmup=args.warmup,
            iters=args.iters,
            profile_dir=args.profile_dir,
        )
        print(json.dumps({"profile": profile_paths}, ensure_ascii=False))
    else:
        res = benchmark_xkv_attention(
            mode=args.mode,
            group_size=args.group_size,
            rank_k=args.rank_k,
            rank_v=args.rank_v,
            sparse_budget=args.sparse_budget,
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            model_name=args.model_name,
            layer_idx=args.layer_idx,
            prefill_len=args.prefill_len,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(json.dumps(res, ensure_ascii=False))
