"""
Phase 3: Cost-per-1M-tokens calculator.

Combines:
  - throughput (tokens/sec) measured by locust/vLLM benchmark runs
  - GPU hourly cost (on-demand / spot / reserved, or your own hardware amortized)

to produce a $/1M output tokens figure per quantization config.

Usage:
    python cost_calculator.py \
        --throughput-csv results/fp16_run_stats.csv,results/int8_run_stats.csv,results/fp8_run_stats.csv \
        --configs fp16,int8,fp8 \
        --gpu-hourly-cost 2.50 \
        --out results/cost_report.json

If you don't have locust CSVs yet, you can also pass --tokens-per-sec directly:
    python cost_calculator.py --configs fp16,int8,fp8 \
        --tokens-per-sec 850,1400,2100 --gpu-hourly-cost 2.50
"""
import argparse
import csv
import json
from pathlib import Path


def tokens_per_sec_from_locust_csv(csv_path: str, avg_output_tokens: float) -> float:
    """
    Parses locust's _stats.csv. Expects an aggregated row; we approximate
    tokens/sec as (avg output tokens per request) * (requests/sec).
    `avg_output_tokens` MUST match the actual max_tokens used in locustfile.py
    for this run (previously this was silently hardcoded to 256 and would
    silently produce wrong numbers if that value was ever changed).
    NOTE: for a precise figure, prefer vLLM's own benchmark_serving.py output
    or instrument gateway_tokens_generated_total from Prometheus directly,
    since real completions rarely hit max_tokens exactly (early EOS shortens
    them) -- treat this as an upper-bound approximation, not ground truth.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    agg = next((r for r in rows if r.get("Name") == "Aggregated"), rows[-1] if rows else None)
    if agg is None:
        raise ValueError(f"Could not parse aggregate row from {csv_path}")
    rps = float(agg["Requests/s"])
    return rps * avg_output_tokens


def cost_per_million_tokens(tokens_per_sec: float, gpu_hourly_cost: float) -> float:
    if tokens_per_sec <= 0:
        return float("inf")
    tokens_per_hour = tokens_per_sec * 3600
    cost_per_token = gpu_hourly_cost / tokens_per_hour
    return cost_per_token * 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--throughput-csv", default=None, help="comma-separated locust CSV paths, same order as --configs")
    parser.add_argument("--tokens-per-sec", default=None, help="comma-separated manual tokens/sec values, same order as --configs")
    parser.add_argument("--gpu-hourly-cost", type=float, required=True, help="$/hour for the GPU instance/hardware")
    parser.add_argument("--avg-output-tokens", type=float, default=256,
                         help="avg output tokens per request used in the locust run (MUST match locustfile.py's max_tokens, or the actual observed average if you have it from gateway_tokens_generated_total)")
    parser.add_argument("--out", default="results/cost_report.json")
    args = parser.parse_args()

    configs = args.configs.split(",")

    if args.tokens_per_sec:
        tps_values = [float(x) for x in args.tokens_per_sec.split(",")]
    elif args.throughput_csv:
        csv_paths = args.throughput_csv.split(",")
        tps_values = [tokens_per_sec_from_locust_csv(p, args.avg_output_tokens) for p in csv_paths]
    else:
        raise ValueError("Must supply either --throughput-csv or --tokens-per-sec")

    assert len(configs) == len(tps_values), "configs and throughput values must align"

    report = {}
    for cfg, tps in zip(configs, tps_values):
        report[cfg] = {
            "tokens_per_sec": tps,
            "gpu_hourly_cost_usd": args.gpu_hourly_cost,
            "cost_per_1m_tokens_usd": round(cost_per_million_tokens(tps, args.gpu_hourly_cost), 4),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Cost report written to {args.out}\n")
    for cfg, r in report.items():
        print(f"  {cfg}: {r['tokens_per_sec']:.1f} tok/s -> ${r['cost_per_1m_tokens_usd']}/1M tokens")


if __name__ == "__main__":
    main()
