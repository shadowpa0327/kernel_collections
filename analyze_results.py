#!/usr/bin/env python3
"""
Analyze and visualize experiment results.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys


def load_results(results_file: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def load_jsonl_results(results_file: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def print_summary_table(results: Dict[str, Any]):
    """Print a formatted summary table of results."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)

    if "metadata" in results:
        print("\nMetadata:")
        for key, value in results["metadata"].items():
            print(f"  {key}: {value}")

    experiments = results.get("experiments", [])
    if not experiments:
        print("\nNo experiments found in results file.")
        return

    # Print table header
    print("\n" + "-"*100)
    print(f"{'Config':<20} {'Type':<12} {'Batch':<6} {'SeqLen':<10} {'Time (ms)':<12} {'Status':<8}")
    print("-"*100)

    # Group results by seqlen and batch_size
    grouped = {}
    for exp in experiments:
        if not exp.get("success"):
            continue

        # Extract key info
        config_name = exp.get("config_name", "Unknown")
        exp_type = exp.get("experiment_type", "Unknown")
        batch_size = exp.get("batch_size", "N/A")

        # Get sequence length from different possible fields
        seqlen = exp.get("prefill_len") or exp.get("kv_len") or exp.get("max_length", "N/A")
        if isinstance(seqlen, int):
            seqlen_key = seqlen
            seqlen_str = f"{seqlen//1024}k"
        else:
            seqlen_key = 0
            seqlen_str = str(seqlen)

        avg_ms = exp.get("avg_ms", "N/A")
        if isinstance(avg_ms, (int, float)):
            time_str = f"{avg_ms:.2f}"
        else:
            time_str = str(avg_ms)

        status = "✓" if exp.get("success") else "✗"

        # Store in grouped dict
        key = (seqlen_key, batch_size)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append({
            "config_name": config_name,
            "exp_type": exp_type,
            "seqlen_str": seqlen_str,
            "avg_ms": avg_ms,
            "status": status
        })

    # Print sorted by seqlen and batch_size
    for (seqlen, batch_size), exps in sorted(grouped.items()):
        for exp in exps:
            print(f"{exp['config_name']:<20} {exp['exp_type']:<12} {batch_size:<6} "
                  f"{exp['seqlen_str']:<10} {exp['avg_ms']:<12.2f} {exp['status']:<8}")

    # Print statistics
    print("\n" + "-"*100)
    total = len(experiments)
    successful = sum(1 for e in experiments if e.get("success"))
    failed = total - successful
    print(f"Total experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Print comparison for each configuration
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON")
    print("="*100)

    for (seqlen, batch_size), exps in sorted(grouped.items()):
        seqlen_str = f"{seqlen//1024}k" if seqlen > 0 else "N/A"
        print(f"\nSeqLen: {seqlen_str}, Batch Size: {batch_size}")
        print("-"*80)

        # Find baseline (FlashAttention)
        baseline_ms = None
        for exp in exps:
            if "FlashAttention" in exp["config_name"] or exp["config_name"] == "fa":
                baseline_ms = exp["avg_ms"]
                break

        # Print each config with speedup
        for exp in exps:
            avg_ms = exp["avg_ms"]
            speedup_str = ""
            if baseline_ms and isinstance(avg_ms, (int, float)) and avg_ms > 0:
                speedup = baseline_ms / avg_ms
                speedup_str = f"(Speedup: {speedup:.2f}x)"

            print(f"  {exp['config_name']:<25} {avg_ms:>10.2f} ms  {speedup_str}")


def export_csv(results: Dict[str, Any], output_file: str):
    """Export results to CSV format."""
    import csv

    experiments = results.get("experiments", [])
    if not experiments:
        print("No experiments to export.")
        return

    # Collect all unique keys from all experiments
    all_keys = set()
    for exp in experiments:
        all_keys.update(exp.keys())

    # Sort keys for consistent column order
    fieldnames = sorted(all_keys)

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            writer.writerow(exp)

    print(f"\nResults exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--export-csv", help="Export results to CSV file")
    parser.add_argument("--jsonl", action="store_true", help="Input is JSONL format")

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)

    # Load results
    if args.jsonl:
        experiments = load_jsonl_results(args.results_file)
        results = {"experiments": experiments}
    else:
        results = load_results(args.results_file)

    # Print summary
    print_summary_table(results)

    # Export to CSV if requested
    if args.export_csv:
        export_csv(results, args.export_csv)


if __name__ == "__main__":
    main()
