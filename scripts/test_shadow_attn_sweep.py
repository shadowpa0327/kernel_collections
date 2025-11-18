#!/usr/bin/env python3
"""
Script to sweep over different batch sizes and prefill lengths for shadow attention testing.
This script will run the shadow_attn.py benchmark with all three configurations (xkv, xkey, full) and dump all JSON results.
"""

import os
import sys
import json
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import from xKV
sys.path.append(str(Path(__file__).parent.parent / "xKV"))

def run_shadow_attn_benchmark(mode, model_name, rank_k, rank_v, batch_size, prefill_len, 
                             max_length=None, warmup=10, iters=50):
    """
    Run the shadow_attn.py benchmark with specified parameters.
    
    Returns:
        dict: Benchmark results or None if failed
    """
    # Set max_length to be at least prefill_len
    if max_length is None:
        max_length = max(prefill_len, 131072)
    
    cmd = [
        "python", "xKV/shadow_attn.py",
        "--mode", mode,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--prefill_len", str(prefill_len),
        "--max_length", str(max_length),
        "--warmup", str(warmup),
        "--iters", str(iters)
    ]
    
    # Add rank parameters only for modes that use them
    if mode in ["xkv", "xkey"]:
        cmd.extend(["--rank_k", str(rank_k)])
        if mode == "xkv":
            cmd.extend(["--rank_v", str(rank_v)])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse JSON output
            output = json.loads(result.stdout.strip())
            return output
        else:
            print(f"Error running benchmark: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Benchmark timed out for {mode} batch_size={batch_size}, prefill_len={prefill_len}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON output: {e}")
        print(f"Raw output: {result.stdout}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    # Test all three configurations for comparison
    test_configs = [
        {"mode": "xkv", "rank_k": 384, "rank_v": 576, "name": "xkv_k384_v576"},
        {"mode": "xkey", "rank_k": 256, "rank_v": None, "name": "xkey_k256"},
        {"mode": "full", "rank_k": None, "rank_v": None, "name": "full"}
    ]
    
    model_name = "local"
    
    # Parameter sweeps
    batch_sizes = [4]
    prefill_lens = [32*1024, 64*1024, 128*1024]
    
    # Benchmark settings
    warmup = 10
    iters = 25
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"shadow_attn_sweep_comparison_{timestamp}.csv"
    json_output_file = f"shadow_attn_sweep_comparison_{timestamp}.json"
    
    print(f"Starting shadow attention comparison sweep...")
    print(f"Model: {model_name}")
    print(f"Test configurations:")
    for config in test_configs:
        if config["mode"] == "xkv":
            print(f"  - {config['mode']}: rank_k={config['rank_k']}, rank_v={config['rank_v']}")
        elif config["mode"] == "xkey":
            print(f"  - {config['mode']}: rank_k={config['rank_k']}")
        else:
            print(f"  - {config['mode']}: full attention baseline")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prefill lengths: {prefill_lens}")
    print(f"Output files: {output_file}, {json_output_file}")
    print("-" * 80)
    
    # Prepare CSV output
    fieldnames = [
        'config_name', 'mode', 'model_name', 'rank_k', 'rank_v', 'batch_size', 'prefill_len', 
        'max_length', 'avg_ms', 'group_size', 'sparse_budget', 'chunk_size',
        'head_dim', 'num_heads', 'timestamp'
    ]
    
    results = []
    all_json_results = []
    total_runs = len(test_configs) * len(batch_sizes) * len(prefill_lens)
    current_run = 0
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for config in test_configs:
            print(f"\n{'='*20} Testing {config['name']} {'='*20}")
            
            for batch_size in batch_sizes:
                for prefill_len in prefill_lens:
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}] {config['name']}: batch_size={batch_size}, prefill_len={prefill_len}")
                    
                    # Run benchmark
                    result = run_shadow_attn_benchmark(
                        mode=config["mode"],
                        model_name=model_name,
                        rank_k=config["rank_k"],
                        rank_v=config["rank_v"],
                        batch_size=batch_size,
                        prefill_len=prefill_len,
                        warmup=warmup,
                        iters=iters
                    )
                    
                    if result:
                        # Add metadata
                        result['config_name'] = config['name']
                        result['model_name'] = model_name
                        result['timestamp'] = datetime.now().isoformat()
                        
                        # Write to CSV
                        writer.writerow(result)
                        csvfile.flush()  # Ensure data is written immediately
                        
                        results.append(result)
                        all_json_results.append(result)
                        print(f"  ✓ Completed: {result['avg_ms']:.2f} ms")
                        
                        # Dump individual JSON result
                        print(f"  JSON Result: {json.dumps(result, indent=2)}")
                    else:
                        print(f"  ✗ Failed")
                    
                    # Brief pause between runs
                    time.sleep(1)
    
    # Save all JSON results to file
    with open(json_output_file, 'w') as jsonfile:
        json.dump(all_json_results, jsonfile, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Sweep completed!")
    print(f"Results saved to: {output_file}")
    print(f"JSON results saved to: {json_output_file}")
    print(f"Successful runs: {len(results)}/{total_runs}")
    
    if results:
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Fastest run: {min(r['avg_ms'] for r in results):.2f} ms")
        print(f"  Slowest run: {max(r['avg_ms'] for r in results):.2f} ms")
        print(f"  Average time: {sum(r['avg_ms'] for r in results) / len(results):.2f} ms")
        
        # Group by configuration
        config_stats = {}
        for result in results:
            config = result['config_name']
            if config not in config_stats:
                config_stats[config] = []
            config_stats[config].append(result['avg_ms'])
        
        print(f"\nBy Configuration:")
        for config in sorted(config_stats.keys()):
            config_times = config_stats[config]
            avg_time = sum(config_times) / len(config_times)
            min_time = min(config_times)
            max_time = max(config_times)
            print(f"  {config}: {avg_time:.2f} ms avg, {min_time:.2f}-{max_time:.2f} ms range ({len(config_times)} runs)")
        
        # Group by batch size
        batch_stats = {}
        for result in results:
            bs = result['batch_size']
            if bs not in batch_stats:
                batch_stats[bs] = []
            batch_stats[bs].append(result['avg_ms'])
        
        print(f"\nBy Batch Size:")
        for bs in sorted(batch_stats.keys()):
            batch_times = batch_stats[bs]
            avg_time = sum(batch_times) / len(batch_times)
            print(f"  Batch {bs}: {avg_time:.2f} ms avg ({len(batch_times)} runs)")
        
        # Group by prefill length
        prefill_stats = {}
        for result in results:
            pl = result['prefill_len']
            if pl not in prefill_stats:
                prefill_stats[pl] = []
            prefill_stats[pl].append(result['avg_ms'])
        
        print(f"\nBy Prefill Length:")
        for pl in sorted(prefill_stats.keys()):
            prefill_times = prefill_stats[pl]
            avg_time = sum(prefill_times) / len(prefill_times)
            print(f"  Prefill {pl}: {avg_time:.2f} ms avg ({len(prefill_times)} runs)")
        
        # Performance comparison (relative to full attention)
        if any(r['config_name'] == 'full' for r in results):
            print(f"\nPerformance vs Full Attention:")
            full_results = {(r['batch_size'], r['prefill_len']): r['avg_ms'] 
                           for r in results if r['config_name'] == 'full'}
            
            for config_name in ['xkv_k384_v576', 'xkey_k256']:
                config_results = [(r['batch_size'], r['prefill_len'], r['avg_ms']) 
                                for r in results if r['config_name'] == config_name]
                
                if config_results:
                    speedups = []
                    for bs, pl, avg_ms_val in config_results:
                        if (bs, pl) in full_results:
                            speedup = full_results[(bs, pl)] / avg_ms_val
                            speedups.append(speedup)
                    
                    if speedups:
                        avg_speedup = sum(speedups) / len(speedups)
                        min_speedup = min(speedups)
                        max_speedup = max(speedups)
                        print(f"  {config_name}: {avg_speedup:.2f}x speedup avg, {min_speedup:.2f}-{max_speedup:.2f}x range")
    
    # Dump all JSON results at the end
    print(f"\n" + "="*80)
    print("ALL JSON RESULTS")
    print("="*80)
    print(json.dumps(all_json_results, indent=2))

if __name__ == "__main__":
    main() 