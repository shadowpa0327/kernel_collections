"""
Benchmark: CUDA vs Triton batch_gather_gemm Performance

Compares the performance of CUTLASS-based CUDA implementation against
auto-tuned Triton implementation across various problem sizes.

Usage:
    python benchmark_batch_gather_gemm.py [--rank-sweep] [--budget-sweep] [--pipeline-sweep]
"""

import torch
import triton
import triton.testing
import argparse
import os

# Import implementations
from triton_batch_gather_gemm_clean import (
    batch_gather_gemm_triton,
    batch_gather_gemm_rotary_pos_emb_triton
)
from kernels import shadowkv


def create_inputs(batch_size, heads, seq_len, rank, head_dim, chunk_size, num_chunks, max_seq_len, device='cuda', dtype=torch.bfloat16):
    """
    Create realistic inputs for batch_gather_gemm operation.

    Returns:
        dict with keys: a, b, position_ids, cos_sin, cnts, output, cache
    """
    # Input matrix A
    a = torch.rand(batch_size, seq_len, rank, dtype=dtype, device=device) * 2 - 1

    # Weight matrix B (with batch dimension for CUDA kernel)
    b = torch.rand(batch_size, heads, head_dim, rank, dtype=dtype, device=device) * 2 - 1

    # Position IDs (sorted chunk indices)
    max_chunk_id = seq_len // chunk_size - 1
    position_ids = torch.randint(0, max_chunk_id + 1, (batch_size, heads, num_chunks), dtype=torch.int32, device=device)
    position_ids, _ = torch.sort(position_ids, dim=-1)

    # RoPE cache (realistic trigonometric values)
    half_dim = head_dim // 2
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    freqs = torch.outer(positions, inv_freq)
    cos_cache = torch.cos(freqs).to(dtype)
    sin_cache = torch.sin(freqs).to(dtype)
    cos_sin = torch.cat([cos_cache, sin_cache], dim=-1)

    # Cnts (process all chunks)
    cnts = torch.zeros(batch_size * heads, dtype=torch.int32, device=device)

    # Output buffers
    sparse_budget = num_chunks * chunk_size
    output = torch.zeros(batch_size, heads, sparse_budget, head_dim, dtype=dtype, device=device)
    cache = torch.zeros(batch_size, heads, max_seq_len, head_dim, dtype=dtype, device=device)

    return {
        'a': a,
        'b': b,
        'position_ids': position_ids,
        'cos_sin': cos_sin,
        'cnts': cnts,
        'output': output,
        'cache': cache,
        'chunk_size': chunk_size,
        'sparse_budget': sparse_budget,
        'max_seq_len': max_seq_len,
    }


def run_rank_sweep():
    """Benchmark: GEMM performance vs rank (K dimension)"""

    configs = [
        triton.testing.Benchmark(
            x_names=["rank"],
            x_vals=[32, 64, 128, 256],
            line_arg="provider",
            line_vals=["cuda", "triton"],
            line_names=["CUDA (CUTLASS)", "Triton (Auto-tuned)"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (μs)",
            plot_name="batch_gather_gemm_by_rank",
            args={
                "batch_size": 1,
                "heads": 8,
                "seq_len": 4096,
                "head_dim": 128,
                "sparse_budget": 512,
                "dtype": torch.bfloat16,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_gemm_by_rank(batch_size, heads, seq_len, rank, head_dim, sparse_budget, provider, dtype=torch.bfloat16, device="cuda"):
        chunk_size = 8
        num_chunks = sparse_budget // chunk_size
        max_seq_len = max(seq_len, sparse_budget)

        # Create inputs
        inputs = create_inputs(batch_size, heads, seq_len, rank, head_dim, chunk_size, num_chunks, max_seq_len, device, dtype)

        # Benchmark settings
        warmup = 25
        rep = 100
        quantiles = [0.5, 0.2, 0.8]

        if provider == "cuda":
            # CUDA GEMM-only (no RoPE)
            def fn():
                shadowkv.batch_gather_gemm(
                    inputs['a'], inputs['b'],
                    inputs['cos_sin'], inputs['cos_sin'],
                    inputs['position_ids'], inputs['output'],
                    batch_size, heads, seq_len, head_dim, rank,
                    sparse_budget, max_seq_len, chunk_size,
                    inputs['cnts']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        elif provider == "triton":
            # Triton GEMM-only (no RoPE)
            def fn():
                batch_gather_gemm_triton(
                    inputs['a'], inputs['b'], inputs['position_ids'],
                    chunk_size, inputs['cnts'], out=inputs['output']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        return ms * 1000, min_ms * 1000, max_ms * 1000  # Convert to microseconds

    os.makedirs('results', exist_ok=True)
    bench_gemm_by_rank.run(print_data=True, show_plots=True, save_path='results/')


def run_sparse_budget_sweep():
    """Benchmark: GEMM performance vs sparse budget (M dimension)"""

    configs = [
        triton.testing.Benchmark(
            x_names=["sparse_budget"],
            x_vals=[64, 128, 256, 512, 1024],
            line_arg="provider",
            line_vals=["cuda", "triton"],
            line_names=["CUDA (CUTLASS)", "Triton (Auto-tuned)"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (μs)",
            plot_name="batch_gather_gemm_by_sparse_budget",
            args={
                "batch_size": 1,
                "heads": 8,
                "seq_len": 8192,
                "rank": 128,
                "head_dim": 128,
                "dtype": torch.bfloat16,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_gemm_by_sparse_budget(batch_size, heads, seq_len, rank, head_dim, sparse_budget, provider, dtype=torch.bfloat16, device="cuda"):
        chunk_size = 8
        num_chunks = sparse_budget // chunk_size
        max_seq_len = max(seq_len, sparse_budget)

        # Create inputs
        inputs = create_inputs(batch_size, heads, seq_len, rank, head_dim, chunk_size, num_chunks, max_seq_len, device, dtype)

        # Benchmark settings
        warmup = 25
        rep = 100
        quantiles = [0.5, 0.2, 0.8]

        if provider == "cuda":
            # CUDA GEMM-only (no RoPE)
            def fn():
                shadowkv.batch_gather_gemm(
                    inputs['a'], inputs['b'],
                    inputs['cos_sin'], inputs['cos_sin'],
                    inputs['position_ids'], inputs['output'],
                    batch_size, heads, seq_len, head_dim, rank,
                    sparse_budget, max_seq_len, chunk_size,
                    inputs['cnts']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        elif provider == "triton":
            # Triton GEMM-only (no RoPE)
            def fn():
                batch_gather_gemm_triton(
                    inputs['a'], inputs['b'], inputs['position_ids'],
                    chunk_size, inputs['cnts'], out=inputs['output']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        return ms * 1000, min_ms * 1000, max_ms * 1000  # Convert to microseconds

    os.makedirs('results', exist_ok=True)
    bench_gemm_by_sparse_budget.run(print_data=True, show_plots=True, save_path='results/')


def run_seq_len_sweep():
    """Benchmark: GEMM performance vs sequence length"""

    configs = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[8192, 16384, 32768, 65536, 131072],
            line_arg="provider",
            line_vals=["cuda", "triton"],
            line_names=["CUDA (CUTLASS)", "Triton (Auto-tuned)"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (μs)",
            plot_name="batch_gather_gemm_by_seq_len",
            args={
                "batch_size": 1,
                "heads": 8,
                "rank": 128,
                "head_dim": 128,
                "sparse_budget": 512,
                "dtype": torch.bfloat16,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_gemm_by_seq_len(batch_size, heads, seq_len, rank, head_dim, sparse_budget, provider, dtype=torch.bfloat16, device="cuda"):
        chunk_size = 8
        num_chunks = sparse_budget // chunk_size
        max_seq_len = max(seq_len, sparse_budget)

        # Create inputs
        inputs = create_inputs(batch_size, heads, seq_len, rank, head_dim, chunk_size, num_chunks, max_seq_len, device, dtype)

        # Benchmark settings
        warmup = 25
        rep = 100
        quantiles = [0.5, 0.2, 0.8]

        if provider == "cuda":
            # CUDA GEMM-only (no RoPE)
            def fn():
                shadowkv.batch_gather_gemm(
                    inputs['a'], inputs['b'],
                    inputs['cos_sin'], inputs['cos_sin'],
                    inputs['position_ids'], inputs['output'],
                    batch_size, heads, seq_len, head_dim, rank,
                    sparse_budget, max_seq_len, chunk_size,
                    inputs['cnts']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        elif provider == "triton":
            # Triton GEMM-only (no RoPE)
            def fn():
                batch_gather_gemm_triton(
                    inputs['a'], inputs['b'], inputs['position_ids'],
                    chunk_size, inputs['cnts'], out=inputs['output']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        return ms * 1000, min_ms * 1000, max_ms * 1000  # Convert to microseconds

    os.makedirs('results', exist_ok=True)
    bench_gemm_by_seq_len.run(print_data=True, show_plots=True, save_path='results/')


def run_skip_ratio_sweep():
    """Benchmark: GEMM performance vs skip ratio (cnts filtering)"""

    configs = [
        triton.testing.Benchmark(
            x_names=["skip_ratio"],
            x_vals=[0.0, 0.25, 0.5, 0.75],
            line_arg="provider",
            line_vals=["cuda", "triton"],
            line_names=["CUDA (CUTLASS)", "Triton (Auto-tuned)"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (μs)",
            plot_name="batch_gather_gemm_by_skip_ratio",
            args={
                "batch_size": 1,
                "heads": 8,
                "seq_len": 8192,
                "rank": 128,
                "head_dim": 128,
                "sparse_budget": 512,
                "dtype": torch.bfloat16,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_gemm_by_skip_ratio(batch_size, heads, seq_len, rank, head_dim, sparse_budget, skip_ratio, provider, dtype=torch.bfloat16, device="cuda"):
        chunk_size = 8
        num_chunks = sparse_budget // chunk_size
        max_seq_len = max(seq_len, sparse_budget)

        # Create inputs
        inputs = create_inputs(batch_size, heads, seq_len, rank, head_dim, chunk_size, num_chunks, max_seq_len, device, dtype)

        # Compute skip counts based on skip_ratio
        # skip_ratio = 0.0 means process all chunks (cnts = 0)
        # skip_ratio = 0.5 means skip first 50% of chunks (cnts = num_chunks * 0.5)
        # skip_ratio = 0.75 means skip first 75% of chunks (cnts = num_chunks * 0.75)
        skip_chunks = int(num_chunks * skip_ratio)
        inputs['cnts'] = torch.full((batch_size * heads,), skip_chunks, dtype=torch.int32, device=device)

        # Benchmark settings
        warmup = 25
        rep = 100
        quantiles = [0.5, 0.2, 0.8]

        if provider == "cuda":
            # CUDA GEMM-only (no RoPE)
            def fn():
                shadowkv.batch_gather_gemm(
                    inputs['a'], inputs['b'],
                    inputs['cos_sin'], inputs['cos_sin'],
                    inputs['position_ids'], inputs['output'],
                    batch_size, heads, seq_len, head_dim, rank,
                    sparse_budget, max_seq_len, chunk_size,
                    inputs['cnts']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        elif provider == "triton":
            # Triton GEMM-only (no RoPE)
            def fn():
                batch_gather_gemm_triton(
                    inputs['a'], inputs['b'], inputs['position_ids'],
                    chunk_size, inputs['cnts'], out=inputs['output']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        return ms * 1000, min_ms * 1000, max_ms * 1000  # Convert to microseconds

    os.makedirs('results', exist_ok=True)
    bench_gemm_by_skip_ratio.run(print_data=True, show_plots=True, save_path='results/')


def run_pipeline_sweep():
    """Benchmark: Full pipeline performance vs number of heads"""

    configs = [
        triton.testing.Benchmark(
            x_names=["heads"],
            x_vals=[2, 4, 8, 16, 32],
            line_arg="provider",
            line_vals=["cuda_gemm", "triton_gemm", "triton_full"],
            line_names=["CUDA GEMM", "Triton GEMM", "Triton GEMM+RoPE"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="Latency (μs)",
            plot_name="batch_gather_gemm_full_pipeline",
            args={
                "batch_size": 1,
                "seq_len": 4096,
                "rank": 128,
                "head_dim": 128,
                "sparse_budget": 512,
                "dtype": torch.bfloat16,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_full_pipeline(batch_size, heads, seq_len, rank, head_dim, sparse_budget, provider, dtype=torch.bfloat16, device="cuda"):
        chunk_size = 8
        num_chunks = sparse_budget // chunk_size
        max_seq_len = max(seq_len, sparse_budget)

        # Create inputs
        inputs = create_inputs(batch_size, heads, seq_len, rank, head_dim, chunk_size, num_chunks, max_seq_len, device, dtype)

        # Benchmark settings
        warmup = 25
        rep = 100
        quantiles = [0.5, 0.2, 0.8]

        if provider == "cuda_gemm":
            # CUDA GEMM-only (no RoPE)
            def fn():
                shadowkv.batch_gather_gemm(
                    inputs['a'], inputs['b'],
                    inputs['cos_sin'], inputs['cos_sin'],
                    inputs['position_ids'], inputs['output'],
                    batch_size, heads, seq_len, head_dim, rank,
                    sparse_budget, max_seq_len, chunk_size,
                    inputs['cnts']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        elif provider == "triton_gemm":
            # Triton GEMM-only (no RoPE)
            def fn():
                batch_gather_gemm_triton(
                    inputs['a'], inputs['b'], inputs['position_ids'],
                    chunk_size, inputs['cnts'], out=inputs['output']
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        elif provider == "triton_full":
            # Triton full pipeline (GEMM + RoPE)
            def fn():
                batch_gather_gemm_rotary_pos_emb_triton(
                    inputs['a'], inputs['b'], inputs['cos_sin'], inputs['position_ids'],
                    inputs['output'], chunk_size, inputs['cache'],
                    0, max_seq_len, inputs['cnts'], no_rope=False
                )
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)

        return ms * 1000, min_ms * 1000, max_ms * 1000  # Convert to microseconds

    os.makedirs('results', exist_ok=True)
    bench_full_pipeline.run(print_data=True, show_plots=True, save_path='results/')


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA vs Triton batch_gather_gemm")
    parser.add_argument("--rank-sweep", action="store_true", help="Run rank dimension sweep")
    parser.add_argument("--budget-sweep", action="store_true", help="Run sparse budget sweep")
    parser.add_argument("--seq-len-sweep", action="store_true", help="Run sequence length sweep")
    parser.add_argument("--skip-ratio-sweep", action="store_true", help="Run skip ratio (cnts filtering) sweep")
    parser.add_argument("--pipeline-sweep", action="store_true", help="Run full pipeline sweep")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")

    args = parser.parse_args()

    # If no specific benchmark selected, run all
    if not (args.rank_sweep or args.budget_sweep or args.seq_len_sweep or args.skip_ratio_sweep or args.pipeline_sweep or args.all):
        args.all = True

    print("="*80)
    print("Benchmark: CUDA vs Triton batch_gather_gemm")
    print("="*80)

    if args.rank_sweep or args.all:
        print("\n[1/5] Running rank dimension sweep...")
        run_rank_sweep()

    if args.budget_sweep or args.all:
        print("\n[2/5] Running sparse budget sweep...")
        run_sparse_budget_sweep()

    if args.seq_len_sweep or args.all:
        print("\n[3/5] Running sequence length sweep...")
        run_seq_len_sweep()

    if args.skip_ratio_sweep or args.all:
        print("\n[4/5] Running skip ratio sweep...")
        run_skip_ratio_sweep()

    if args.pipeline_sweep or args.all:
        print("\n[5/5] Running full pipeline sweep...")
        run_pipeline_sweep()

    print("\n" + "="*80)
    print("Benchmarks complete! Results saved to results/ directory")
    print("="*80)


if __name__ == "__main__":
    main()
