"""
Phase 2: Quality Evaluation Harness

Runs the same fixed prompt/eval set against each quantization config's live
gateway endpoint, scores responses, and stores a per-config quality report.

Usage:
    python quality_harness.py --configs fp16,int8,fp8 \
        --gateway-urls http://localhost:8001,http://localhost:8002,http://localhost:8003 \
        --dataset datasets/eval_set.jsonl \
        --out results/quality_report.json

Design notes:
- Uses a reference config (default: fp16) as ground truth for "quality drop" deltas.
- Scoring: exact-match for closed-form QA, embedding cosine similarity for
  open-ended generation (via sentence-transformers), reported separately.
- Run with enough samples (>=200) for a statistically meaningful delta —
  this harness will warn if the dataset is too small.
"""
import argparse
import json
import os
import time
import statistics
from pathlib import Path

import requests

MIN_RECOMMENDED_SAMPLES = 200
API_KEY = os.environ.get("GATEWAY_API_KEY")


def load_dataset(path: str):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def query_gateway(url: str, prompt: str, max_tokens: int = 256) -> dict:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # deterministic for eval reproducibility
    }
    t0 = time.perf_counter()
    headers = {"X-API-Key": API_KEY} if API_KEY else None
    resp = requests.post(
        f"{url}/v1/chat/completions", json=payload, headers=headers, timeout=120
    )
    resp.raise_for_status()
    latency = time.perf_counter() - t0
    data = resp.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "latency": latency,
        "tokens": data.get("usage", {}).get("completion_tokens", 0),
    }


def exact_match_score(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def embedding_similarity(pred: str, gold: str, model=None) -> float:
    # Lazy import so this file works even without sentence-transformers installed
    # for exact-match-only datasets.
    from sentence_transformers import SentenceTransformer, util
    global _EMB_MODEL
    if model is None:
        if "_EMB_MODEL" not in globals():
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        model = _EMB_MODEL
    emb = model.encode([pred, gold], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]))


def run_eval(config_name: str, gateway_url: str, dataset: list, max_retries: int = 2) -> dict:
    scores = []
    latencies = []
    failed_items = []

    for idx, item in enumerate(dataset):
        prompt = item["prompt"]
        gold = item["reference"]
        qtype = item.get("type", "open_ended")

        # Per-item error isolation: a single flaky/failed request must not
        # abort the whole eval run and discard every score gathered so far.
        result = None
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = query_gateway(gateway_url, prompt)
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                continue

        if result is None:
            print(f"[WARN] item {idx} failed after {max_retries + 1} attempts: {last_error}")
            failed_items.append({"index": idx, "prompt": prompt, "error": str(last_error)})
            continue

        latencies.append(result["latency"])

        try:
            if qtype == "closed_form":
                score = exact_match_score(result["text"], gold)
            else:
                score = embedding_similarity(result["text"], gold)
        except Exception as e:
            print(f"[WARN] scoring failed for item {idx}: {e}")
            failed_items.append({"index": idx, "prompt": prompt, "error": f"scoring error: {e}"})
            continue

        scores.append(score)

    if failed_items:
        print(f"[WARN] {len(failed_items)}/{len(dataset)} items failed for config={config_name}. "
              f"Quality numbers below only reflect the {len(scores)} that succeeded.")

    return {
        "config": config_name,
        "n_samples": len(dataset),
        "n_succeeded": len(scores),
        "n_failed": len(failed_items),
        "failed_items": failed_items,
        "mean_score": statistics.mean(scores) if scores else None,
        "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "raw_scores": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="comma-separated config names, e.g. fp16,int8,fp8")
    parser.add_argument("--gateway-urls", required=True, help="comma-separated URLs matching --configs order")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference-config", default=None, help="config to use as quality baseline (default: first)")
    parser.add_argument("--out", default="results/quality_report.json")
    args = parser.parse_args()

    configs = args.configs.split(",")
    urls = args.gateway_urls.split(",")
    assert len(configs) == len(urls), "configs and gateway-urls must be the same length"

    dataset = load_dataset(args.dataset)
    if len(dataset) < MIN_RECOMMENDED_SAMPLES:
        print(f"[WARN] Dataset has {len(dataset)} samples; "
              f"recommend >= {MIN_RECOMMENDED_SAMPLES} for a statistically meaningful quality delta.")

    reference = args.reference_config or configs[0]
    reports = {}
    for cfg, url in zip(configs, urls):
        print(f"Running eval for config={cfg} against {url} ...")
        reports[cfg] = run_eval(cfg, url, dataset)

    ref_score = reports[reference]["mean_score"]
    for cfg, report in reports.items():
        if ref_score is None or report["mean_score"] is None:
            report["quality_delta_vs_reference"] = None
        else:
            report["quality_delta_vs_reference"] = report["mean_score"] - ref_score
        report["reference_config"] = reference

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(reports, f, indent=2)

    print(f"\nQuality report written to {args.out}")
    for cfg, report in reports.items():
        score_str = f"{report['mean_score']:.4f}" if report["mean_score"] is not None else "N/A (all items failed)"
        delta_str = (f"{report['quality_delta_vs_reference']:+.4f}"
                     if report["quality_delta_vs_reference"] is not None else "N/A")
        fail_note = f" ({report['n_failed']} failed)" if report.get("n_failed") else ""
        print(f"  {cfg}: mean_score={score_str} delta_vs_{reference}={delta_str}{fail_note}")


if __name__ == "__main__":
    main()
