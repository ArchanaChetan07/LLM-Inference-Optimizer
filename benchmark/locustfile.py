"""
Phase 3: Load test against a gateway config.

Run:
    locust -f locustfile.py --host http://localhost:8001 \
        --users 50 --spawn-rate 5 --run-time 5m --headless \
        --csv=results/fp16_run

Repeat per config (int8, fp8) pointing --host at each gateway instance.
Locust's CSV output (results/<config>_run_stats.csv) feeds directly into
cost_calculator.py.
"""
import random
import os
from locust import HttpUser, task, between, events

PROMPTS = [
    "Write a short paragraph about renewable energy.",
    "Explain the difference between TCP and UDP.",
    "Summarize the plot of a mystery novel in two sentences.",
    "List three benefits of regular exercise.",
    "What are the main causes of inflation?",
]


class GatewayUser(HttpUser):
    wait_time = between(0.1, 1.0)

    def on_start(self):
        api_key = os.environ.get("GATEWAY_API_KEY")
        if api_key:
            self.client.headers.update({"X-API-Key": api_key})

    @task
    def chat_completion(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
        }
        with self.client.post("/v1/chat/completions", json=payload, catch_response=True, timeout=120) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
            else:
                resp.success()


@events.quitting.add_listener
def _summary(environment, **kwargs):
    stats = environment.stats.total
    print(f"\nRun summary: requests={stats.num_requests} "
          f"failures={stats.num_failures} "
          f"p50={stats.get_response_time_percentile(0.5)}ms "
          f"p95={stats.get_response_time_percentile(0.95)}ms "
          f"p99={stats.get_response_time_percentile(0.99)}ms "
          f"rps={stats.total_rps:.2f}")
