# Production Readiness Audit

This documents what was found and fixed in the second pass, and what
still requires a human decision (can't be auto-fixed generically).

## Bugs fixed

| # | Issue | Fix |
|---|-------|-----|
| 1 | `final_output` referenced before assignment if the vLLM generator yielded zero outputs → `UnboundLocalError` masking the real error as a 500 with a confusing traceback | Initialize `final_output = None`, check explicitly, raise a clean `RuntimeError` |
| 2 | `quantization: "bitsandbytes"` alone is not sufficient for vLLM — it also requires `load_format="bitsandbytes"`, otherwise engine init fails | Set both when `quantization == "bitsandbytes"` |
| 3 | FP8 config would crash with an opaque CUDA kernel error on Turing/Ampere GPUs (T4, A10, A100) since FP8 needs compute capability ≥ 8.9 (Ada/Hopper) | Added `_check_gpu_supports_fp8()` — fails fast at startup with a clear message and a suggested alternative (AWQ/GPTQ) |
| 4 | Gateway had no authentication — anyone with network access could burn arbitrary GPU/cloud spend | Added `X-API-Key` header auth via `GATEWAY_API_KEY`; **gateway now refuses to start with no auth configured** unless you explicitly set `GATEWAY_ALLOW_NO_AUTH=1` for local dev |
| 5 | No request size limits — unbounded `max_tokens`/message length/message count was a trivial DoS vector | Added Pydantic validators capping all three, configurable via env vars |
| 6 | No rate limiting | Added `slowapi` limiter, default 30 req/min/IP, configurable |
| 7 | SIGTERM during a rolling update killed in-flight requests immediately | Added drain logic in the lifespan shutdown handler + separate `/readyz` (fails during drain, pulled from Service endpoints) vs `/healthz` (liveness, stays green until actual exit) + `preStop` hook + `terminationGracePeriodSeconds: 45` |
| 8 | Eval harness aborted the entire run if a single prompt failed, discarding all scores gathered so far | Added per-item retry (2x) + isolation; failures are recorded and reported separately, run continues |
| 9 | Cost calculator hardcoded `avg_output_tokens = 256` regardless of what `max_tokens` was actually configured in the load test, silently producing wrong $/1M-token figures if you ever changed it | Made it a required, explicit CLI parameter (`--avg-output-tokens`) with a docstring warning that it's an upper-bound approximation |
| 10 | K8s pods had no GPU node targeting — scheduler could place them on non-GPU nodes where they'd hang forever failing readiness | Added `nodeSelector`/`tolerations` for GPU nodes |
| 11 | No model-weight caching — every pod restart re-downloaded multi-GB weights from Hugging Face | Added `HF_HOME` env var + PVC (`hf-model-cache`) mounted at `/data/hf-cache`, `persistence.enabled` toggle in values.yaml |
| 12 | Fixed `initialDelaySeconds: 30` on probes — real model loads (esp. cold HF downloads) can take minutes, causing `CrashLoopBackOff` | Replaced with a `startupProbe` (up to 10 min) that gates when liveness/readiness are even evaluated |
| 13 | Container ran as root | Added non-root user in Dockerfile + `securityContext.runAsNonRoot` in the pod spec |
| 14 | No `PodDisruptionBudget` — a cluster-autoscaler node drain could take down all replicas of a config at once | Added `PodDisruptionBudget` (`minAvailable: 1`) per config |
| 15 | Errors returned raw exception text (`str(e)`) to the client, potentially leaking internals | Now logs full traceback server-side, returns a generic message to the caller |
| 16 | No `.dockerignore`/`.gitignore`/`.env.example` | Added all three |

## Verified, not just claimed

- Ran `python3 -m ast` (syntax parse) on every modified `.py` file after each edit — all pass
- Manually re-grepped the file after each edit to confirm it actually landed (the edit tool silently failed on ~4 attempts mid-session; caught and redone via direct file writes rather than assumed successful)
- Validated `values.yaml` and `Chart.yaml` with `yaml.safe_load` — both parse
- Brace-balance checked all Helm templates (`{{` vs `}}` counts match) since Helm itself could not be installed in this sandbox (no network egress to `get.helm.sh` here)

## Known gaps — still require a human decision, not auto-fixable

These are genuine "it depends on your environment" items, not oversights:

- **Helm chart was never actually rendered with `helm template`** in this environment (no network access to install Helm here). Run `helm template . --debug` yourself before applying — brace-balance and YAML-validity checks are necessary but not sufficient.
- **No TLS termination configured** — assumed to sit behind an ingress/load balancer that terminates TLS (standard for k8s, but worth stating explicitly rather than silently assuming).
- **Secrets management is a placeholder** (`templates/secrets.yaml` creates a dummy key only if `createPlaceholderSecret: true`, defaulted to `false`). You must create the real `gateway-secrets` Secret via `kubectl create secret` or a proper secrets operator (Vault/External Secrets) — this is deliberately not automated because there's no generically-correct way to inject a real secret into a portfolio repo.
- **No mTLS/network policy between the gateway and Prometheus/Grafana** — add a `NetworkPolicy` if your cluster's threat model requires pod-to-pod isolation.
- **Speculative decoding draft-model quality neutrality is asserted by design, not empirically verified in this repo** — you still need to actually run the Phase 2 eval harness with spec decode on vs. off and confirm scores don't regress, as the original plan specified. The code supports it; the validation run is on you.
- **Terraform stub is GCP-only and uses local state** — fine for a portfolio demo, not fine for a team; add a remote backend (GCS/S3 + locking) before real use.
- **No load-shedding/circuit breaker on the gateway to reject work below a minimum latency SLO** — the rate limiter caps request rate but doesn't shed load based on current queue depth; consider adding if you see queue depth spike past the HPA scale-out threshold before new replicas come up (there's a genuine window where requests will queue rather than fail fast).
