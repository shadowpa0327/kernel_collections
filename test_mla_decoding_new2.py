import torch
import argparse
from flash_decoding.mla2 import mla_decode_attention_fwd, mla_decode_attention_fwd_ref, mla_decode_attention_xKV
from tqdm import tqdm
from functools import partial

def test_mla_decode_attention_fwd(
    batch_size=4,
    num_q_heads=16,
    num_kv_heads=1,
    lora_rank=512,
    xKV_rank=256,
    rope_dim=64,
    kv_len=64*1024,
    sm_scale=1.0,
    dtype=torch.float16,
    device=None,
):
    """
    Test function to compare the results of the optimized flash decoding 
    implementation against the reference implementation.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Create test tensors
    q_nope = torch.randn(batch_size, num_q_heads, lora_rank, dtype=dtype, device=device)/3
    q_rope = torch.randn(batch_size, num_q_heads, rope_dim, dtype=dtype, device=device)/3
    #kv_nope = torch.randn(batch_size, kv_len, num_kv_heads, lora_rank, dtype=dtype, device=device)/3 
    kv_A_nope = torch.randn(batch_size, kv_len, 1, xKV_rank, dtype=dtype, device=device)/3
    kv_B_nope = torch.randn(batch_size, 1, xKV_rank, lora_rank, dtype=dtype, device=device)/3

    kv_nope = torch.matmul(kv_A_nope.transpose(1, 2), kv_B_nope) # [batch, num_kv_heads, kv_len, lora_rank]
    kv_nope = kv_nope.transpose(1, 2) # [batch, kv_len, num_kv_heads, lora_rank]
    
    k_rope = torch.randn(batch_size, kv_len, num_kv_heads, rope_dim, dtype=dtype, device=device)/3
    
    # Run both implementations
    out_flash = mla_decode_attention_fwd(
        q_nope.clone(), 
        q_rope.clone(), 
        kv_nope.clone(), 
        k_rope.clone(), 
        sm_scale=sm_scale
    )

    out_ref = mla_decode_attention_fwd_ref(
        q_nope.clone(), 
        q_rope.clone(), 
        kv_nope.clone(), 
        k_rope.clone(), 
        sm_scale=sm_scale
    )

    out_flash_xKV = mla_decode_attention_xKV(
        q_nope.clone(), 
        q_rope.clone(), 
        kv_A_nope.clone(), 
        kv_B_nope.clone(), 
        k_rope.clone(), 
        sm_scale=sm_scale
    )
    
    # Compare outputs
    # Compare standard implementation outputs
    max_diff = torch.max(torch.abs(out_flash - out_ref))
    mean_diff = torch.mean(torch.abs(out_flash - out_ref))
    print(f"Standard implementation - Maximum absolute difference: {max_diff.item():.6e}")
    print(f"Standard implementation - Mean absolute difference: {mean_diff.item():.6e}")
    print(f"Cosine similarity: {torch.nn.functional.cosine_similarity(out_flash, out_ref, dim=-1).mean().item()}")
    
    # Check if the standard outputs are close enough
    rtol = 1e-2  # Relative tolerance
    atol = 1e-2  # Absolute tolerance
    is_close = torch.allclose(out_flash, out_ref, rtol=rtol, atol=atol)
    
    if is_close:
        print("Standard Test PASSED: Flash implementation matches reference implementation")
    else:
        print("Standard Test FAILED: Implementations produce different results")
    
    # Compare xKV implementation with reference
    # For reference, we need to reconstruct the output using the same approach
    out_ref_xKV = mla_decode_attention_fwd_ref(
        q_nope.clone(), 
        q_rope.clone(), 
        kv_nope.clone(), 
        k_rope.clone(), 
        sm_scale=sm_scale
    )
    
    max_diff_xKV = torch.max(torch.abs(out_flash_xKV - out_ref_xKV))
    mean_diff_xKV = torch.mean(torch.abs(out_flash_xKV - out_ref_xKV))
    print(f"xKV implementation - Maximum absolute difference: {max_diff_xKV.item():.6e}")
    print(f"xKV implementation - Mean absolute difference: {mean_diff_xKV.item():.6e}")
    print(f"Cosine similarity: {torch.nn.functional.cosine_similarity(out_flash_xKV, out_ref_xKV, dim=-1).mean().item()}")
    
    is_close_xKV = torch.allclose(out_flash_xKV, out_ref_xKV, rtol=rtol, atol=atol)
    
    if is_close_xKV:
        print("xKV Test PASSED: Flash xKV implementation matches reference implementation")
    else:
        print("xKV Test FAILED: xKV implementation produces different results")

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
    return times, avg_time

def test_mla_decode_attention_latency(
    batch_size=8,
    num_q_heads=16,
    num_kv_heads=1,
    lora_rank=512,
    xKV_rank=256,
    rope_dim=64,
    kv_len=64*1024,
    dtype=torch.float16,
    device="cuda",
    num_warmup=50,
    num_repeats=1000,
    test_xkv=True,
):
    """
    Test the latency of mla_decode_attention_fwd and mla_decode_attention_xKV implementations.
    
    Args:
        batch_size: Batch size
        num_q_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        lora_rank: Rank for low-rank adaptation
        xKV_rank: Rank for xKV implementation
        rope_dim: Dimension for rotary positional embedding
        kv_len: Length of key-value sequence
        dtype: Data type for tensors
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_repeats: Number of timing iterations
        test_xkv: Whether to test xKV implementation
    
    Returns:
        Dictionary with mean latencies in milliseconds
    """
    # Create test tensors for standard implementation
    q_nope = torch.randn(batch_size, num_q_heads, lora_rank, dtype=dtype, device=device)/3
    q_rope = torch.randn(batch_size, num_q_heads, rope_dim, dtype=dtype, device=device)/3
    kv_nope = torch.randn(batch_size, kv_len, num_kv_heads, lora_rank, dtype=dtype, device=device)/3 
    k_rope = torch.randn(batch_size, kv_len, num_kv_heads, rope_dim, dtype=dtype, device=device)/3
    
    # Create additional tensors for xKV implementation
    kv_A_nope = torch.randn(batch_size, kv_len, 1, xKV_rank, dtype=dtype, device=device)/3
    kv_B_nope = torch.randn(batch_size, 1, xKV_rank, lora_rank, dtype=dtype, device=device)/3
    
    sm_scale = 1.0 / (lora_rank ** 0.5)
    
    funcs = {
        "mla_decode_attention_fwd": partial(mla_decode_attention_fwd, 
                                            q_nope, 
                                            q_rope, 
                                            kv_nope, 
                                            k_rope, 
                                            sm_scale=sm_scale),
        "mla_decode_attention_xKV": partial(mla_decode_attention_xKV,
                                            q_nope,
                                            q_rope,
                                            kv_A_nope,
                                            kv_B_nope,
                                            k_rope,
                                            sm_scale=sm_scale),
    }
    results = {}
    for name, func in funcs.items():
        times, avg_time = benchmark_function(func, name, num_warmup, num_repeats)
        results[name] = avg_time
    
    # calculate the ratio of xKV to standard
    speedup = results['mla_decode_attention_fwd'] / results['mla_decode_attention_xKV']
    print(f"Speedup (standard/xKV): {speedup:.2f}x")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test MLA Decoding implementations")
    
    # Common parameters
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-q-heads", type=int, default=16, help="Number of query heads")
    parser.add_argument("--num-kv-heads", type=int, default=1, help="Number of key-value heads")
    parser.add_argument("--lora-rank", type=int, default=512, help="Rank for low-rank adaptation")
    parser.add_argument("--xkv-rank", type=int, default=256, help="Rank for xKV implementation")
    parser.add_argument("--rope-dim", type=int, default=64, help="Dimension for rotary positional embedding")
    parser.add_argument("--kv-len", type=int, default=64*1024, help="Length of key-value sequence")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--precision", type=str, default="float16", 
                        choices=["float16", "float32", "bfloat16"], 
                        help="Precision to use for tensors")
    
    # Test selection
    parser.add_argument("--run-correctness", action="store_true", help="Run correctness test")
    parser.add_argument("--run-latency", action="store_true", help="Run latency test")
    
    # Latency test specific parameters
    parser.add_argument("--num-warmup", type=int, default=50, help="Number of warmup iterations for latency test")
    parser.add_argument("--num-repeats", type=int, default=1000, help="Number of timing iterations for latency test")
    parser.add_argument("--test-xkv", action="store_true", help="Test xKV implementation in latency test")
    
    args = parser.parse_args()
    
    # If no test is specified, run both
    if not args.run_correctness and not args.run_latency:
        args.run_correctness = True
        args.run_latency = True
    
    return args


def print_setup_info(args, test_type):
    """Print the current test setup information."""
    print(f"\n===== {test_type} Test Setup =====")
    print(f"Batch size: {args.batch_size}")
    print(f"Query heads: {args.num_q_heads}")
    print(f"KV heads: {args.num_kv_heads}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"xKV rank: {args.xkv_rank}")
    print(f"RoPE dimension: {args.rope_dim}")
    print(f"KV sequence length: {args.kv_len}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision}")
    
    if test_type == "Latency":
        print(f"Warmup iterations: {args.num_warmup}")
        print(f"Test iterations: {args.num_repeats}")
        print(f"Testing xKV: {args.test_xkv}")
    print("="*30)


if __name__ == "__main__":
    args = parse_args()
    
    # Set up precision
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.precision]
    
    # Run tests based on arguments
    if args.run_correctness:
        print_setup_info(args, "Correctness")
        test_mla_decode_attention_fwd(
            batch_size=args.batch_size,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            lora_rank=args.lora_rank,
            xKV_rank=args.xkv_rank,
            rope_dim=args.rope_dim,
            kv_len=args.kv_len,
            dtype=dtype,
            device=args.device
        )
    
    if args.run_latency:
        print_setup_info(args, "Latency")
        test_mla_decode_attention_latency(
            batch_size=args.batch_size,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            lora_rank=args.lora_rank,
            xKV_rank=args.xkv_rank,
            rope_dim=args.rope_dim,
            kv_len=args.kv_len,
            dtype=dtype,
            device=args.device,
            num_warmup=args.num_warmup,
            num_repeats=args.num_repeats,
            test_xkv=args.test_xkv
        )