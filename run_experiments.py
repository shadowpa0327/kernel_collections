#!/usr/bin/env python3
"""
Run comprehensive benchmarking experiments across different configurations.
Logs all results to a timestamped JSON file.
"""

import subprocess
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import argparse


def run_benchmark(script: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a benchmark script with given arguments and return parsed JSON output."""
    cmd = ["python", script]

    for key, value in args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 minute timeout
        )

        # Parse JSON output
        output = json.loads(result.stdout)
        output["success"] = True
        output["command"] = " ".join(cmd)
        return output

    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stderr: {e.stderr}")
        return {
            "success": False,
            "error": str(e),
            "stderr": e.stderr,
            "command": " ".join(cmd)
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        print(f"Stdout: {result.stdout}")
        return {
            "success": False,
            "error": f"JSON decode error: {str(e)}",
            "stdout": result.stdout,
            "command": " ".join(cmd)
        }
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(cmd)}")
        return {
            "success": False,
            "error": "Timeout",
            "command": " ".join(cmd)
        }


def run_experiments(
    output_dir: str = "experiment_results",
    warmup: int = 10,
    iters: int = 50,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    group_size: int = 4,
):
    """Run all experiment configurations."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {
        "timestamp": timestamp,
        "metadata": {
            "warmup": warmup,
            "iters": iters,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "group_size": group_size,
        },
        "experiments": []
    }

    # Define experiment configurations
    configs = [
        # seqlen=60k configurations
        {"seqlen": 60 * 1024, "batch_sizes": [8, 16, 32]},
        # seqlen=122k configurations
        {"seqlen": 122 * 1024, "batch_sizes": [4, 8, 16]},
    ]

    # ShadowKV configurations
    shadow_configs = [
        {"mode": "xkey", "rank_k": 384, "rank_v": 0, "name": "xKey"},
        {"mode": "xkv", "rank_k": 384, "rank_v": 512, "name": "xKV"},
        {"mode": "full", "rank_k": 0, "rank_v": 0, "name": "FullKV"},
    ]

    # Normal attention configurations
    normal_configs = [
        {"mode": "xkv_no_sparse", "rank_k": 384, "rank_v": 512, "name": "xKV_no_sparse"},
        {"mode": "fa", "rank_k": 0, "rank_v": 0, "name": "FlashAttention"},
    ]

    total_experiments = 0
    for config in configs:
        total_experiments += len(config["batch_sizes"]) * (len(shadow_configs) + len(normal_configs))

    print(f"\n{'='*80}")
    print(f"Starting {total_experiments} experiments")
    print(f"Results will be saved to: {output_dir}")
    print(f"{'='*80}\n")

    experiment_count = 0

    # Run experiments for each sequence length
    for config in configs:
        seqlen = config["seqlen"]
        batch_sizes = config["batch_sizes"]

        print(f"\n{'='*80}")
        print(f"Running experiments for seqlen={seqlen} ({seqlen//1024}k)")
        print(f"{'='*80}\n")

        for batch_size in batch_sizes:
            print(f"\n{'-'*80}")
            print(f"Batch size: {batch_size}")
            print(f"{'-'*80}\n")

            # Run ShadowKV benchmarks
            for shadow_cfg in shadow_configs:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] ShadowKV {shadow_cfg['name']}")

                args = {
                    "mode": shadow_cfg["mode"],
                    "batch_size": batch_size,
                    "prefill_len": seqlen,
                    "max_length": seqlen,
                    "warmup": warmup,
                    "iters": iters,
                    "group_size": group_size,
                }

                # Add rank arguments only if non-zero
                if shadow_cfg["rank_k"] > 0:
                    args["rank_k"] = shadow_cfg["rank_k"]
                if shadow_cfg["rank_v"] > 0:
                    args["rank_v"] = shadow_cfg["rank_v"]

                result = run_benchmark("bench_shadowKV_xkv_attn.py", args)
                result["experiment_type"] = "shadowkv"
                result["config_name"] = shadow_cfg["name"]
                all_results["experiments"].append(result)

                # Save intermediate results
                intermediate_file = os.path.join(
                    output_dir,
                    f"experiments_intermediate_{timestamp}.json"
                )
                with open(intermediate_file, "w") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

                print(f"  Status: {'✓ Success' if result.get('success') else '✗ Failed'}")
                if result.get("success"):
                    print(f"  Avg time: {result.get('avg_ms', 'N/A'):.2f} ms")

            # Run normal attention benchmarks
            for normal_cfg in normal_configs:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] Normal {normal_cfg['name']}")

                args = {
                    "mode": normal_cfg["mode"],
                    "batch_size": batch_size,
                    "kv_len": seqlen,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "warmup": warmup,
                    "iters": iters,
                    "group_size": group_size,
                }

                # Add rank arguments only if non-zero
                if normal_cfg["rank_k"] > 0:
                    args["k_rank"] = normal_cfg["rank_k"]
                if normal_cfg["rank_v"] > 0:
                    args["v_rank"] = normal_cfg["rank_v"]

                result = run_benchmark("bench_normal_attention.py", args)
                result["experiment_type"] = "normal"
                result["config_name"] = normal_cfg["name"]
                all_results["experiments"].append(result)

                # Save intermediate results
                with open(intermediate_file, "w") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

                print(f"  Status: {'✓ Success' if result.get('success') else '✗ Failed'}")
                if result.get("success"):
                    print(f"  Avg time: {result.get('avg_ms', 'N/A'):.2f} ms")

    # Save final results
    final_file = os.path.join(output_dir, f"experiments_{timestamp}.json")
    with open(final_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Experiments Complete!")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {sum(1 for e in all_results['experiments'] if e.get('success'))}")
    print(f"Failed: {sum(1 for e in all_results['experiments'] if not e.get('success'))}")
    print(f"\nResults saved to: {final_file}")

    # Generate summary table
    print(f"\n{'='*80}")
    print(f"Summary Table")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'Type':<15} {'Batch':<8} {'SeqLen':<10} {'Time (ms)':<12} {'Status'}")
    print(f"{'-'*80}")

    for exp in all_results["experiments"]:
        config_name = exp.get("config_name", "Unknown")
        exp_type = exp.get("experiment_type", "Unknown")
        batch_size = exp.get("batch_size", "N/A")

        # Get sequence length from different possible fields
        seqlen = exp.get("prefill_len") or exp.get("kv_len") or exp.get("max_length", "N/A")
        if isinstance(seqlen, int):
            seqlen_str = f"{seqlen//1024}k"
        else:
            seqlen_str = str(seqlen)

        avg_ms = exp.get("avg_ms", "N/A")
        if isinstance(avg_ms, (int, float)):
            time_str = f"{avg_ms:.2f}"
        else:
            time_str = str(avg_ms)

        status = "✓" if exp.get("success") else "✗"

        print(f"{config_name:<20} {exp_type:<15} {batch_size:<8} {seqlen_str:<10} {time_str:<12} {status}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive attention benchmarking experiments")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save results")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of measurement iterations")
    parser.add_argument("--num_heads", type=int, default=32,
                        help="Number of query heads")
    parser.add_argument("--num_kv_heads", type=int, default=8,
                        help="Number of key/value heads")
    parser.add_argument("--head_dim", type=int, default=128,
                        help="Head dimension")
    parser.add_argument("--group_size", type=int, default=4,
                        help="Group size for factorization")

    args = parser.parse_args()

    run_experiments(
        output_dir=args.output_dir,
        warmup=args.warmup,
        iters=args.iters,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        group_size=args.group_size,
    )
