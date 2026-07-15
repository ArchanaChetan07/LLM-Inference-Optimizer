#!/usr/bin/env bash
# Runs all 5 phases end-to-end on a SINGLE GPU box (sequential, since you
# likely can't fit fp16+int8+fp8 engines in VRAM simultaneously).
# For a multi-GPU or cloud setup, deploy all three via k8s/helm instead and
# skip straight to eval/benchmark against the live URLs.
set -euo pipefail

cd "$(dirname "$0")/.."

# This is a local-only pipeline. Respect an explicit auth setting, otherwise
# opt into the gateway's required local-development no-auth mode.
export GATEWAY_ALLOW_NO_AUTH="${GATEWAY_ALLOW_NO_AUTH:-1}"

echo "=== Phase 1: Start gateway for each config, one at a time, and smoke test ==="
for cfg in fp16 int8 fp8; do
  echo "--- starting config: $cfg ---"
  MODEL_CONFIG=$cfg uvicorn gateway.app:app --host 0.0.0.0 --port 8000 &
  PID=$!
  sleep 45   # allow model load
  curl -sf http://localhost:8000/healthz || { echo "FAILED health check for $cfg"; kill $PID; exit 1; }
  echo "$cfg healthy."
  kill $PID
  wait $PID 2>/dev/null || true
done

echo "=== Phase 2: Quality eval (run gateway per config on a fixed port, then eval) ==="
echo "NOTE: run gateway instances on distinct ports (8001/8002/8003) in separate"
echo "terminals/tmux panes before running this, e.g.:"
echo "  MODEL_CONFIG=fp16 uvicorn gateway.app:app --port 8001"
echo "  MODEL_CONFIG=int8 uvicorn gateway.app:app --port 8002"
echo "  MODEL_CONFIG=fp8  uvicorn gateway.app:app --port 8003"
python eval/quality_harness.py \
  --configs fp16,int8,fp8 \
  --gateway-urls http://localhost:8001,http://localhost:8002,http://localhost:8003 \
  --dataset eval/datasets/eval_set_sample.jsonl \
  --out eval/results/quality_report.json || echo "Skipped — start gateways first."

echo "=== Phase 3: Load test + cost calculation (run against same 3 ports) ==="
for cfg_port in fp16:8001 int8:8002 fp8:8003; do
  cfg="${cfg_port%%:*}"
  port="${cfg_port##*:}"
  locust -f benchmark/locustfile.py --host "http://localhost:${port}" \
    --users 50 --spawn-rate 5 --run-time 2m --headless \
    --csv "benchmark/results/${cfg}_run" || echo "Skipped $cfg — gateway not running."
done

python benchmark/cost_calculator.py \
  --configs fp16,int8,fp8 \
  --throughput-csv benchmark/results/fp16_run_stats.csv,benchmark/results/int8_run_stats.csv,benchmark/results/fp8_run_stats.csv \
  --gpu-hourly-cost 2.50 \
  --out benchmark/results/cost_report.json || echo "Skipped — run load tests first."

echo "=== Phase 4: Enable speculative decoding in gateway/config/models.yaml (speculative.enabled: true) and re-run Phase 3 to compare ==="

echo "=== Phase 5: Deploy to k8s with autoscaling ==="
echo "helm upgrade --install llm-optimizer ./k8s/helm --namespace inference --create-namespace"
echo "helm install prom-adapter prometheus-community/prometheus-adapter -f k8s/prometheus/prometheus-adapter-values.yaml"
echo "Import k8s/prometheus/grafana-dashboard.json into Grafana."

echo "All phases scripted. Review eval/results/ and benchmark/results/ for the tradeoff report."
