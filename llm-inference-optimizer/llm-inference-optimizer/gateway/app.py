"""
Production gateway wrapping vLLM's AsyncLLMEngine.
- Config-driven quantization (fp16/int8/fp8) via MODEL_CONFIG env var
- Dynamic/continuous batching (native to vLLM's engine)
- Optional speculative decoding
- OpenAI-compatible /v1/chat/completions
- Prometheus metrics at /metrics (queue depth, in-flight requests, tokens/sec)
- OpenTelemetry tracing
"""
import os
import time
import uuid
import yaml
import signal
import logging
import json as json_lib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
except ImportError:
    # Keep request shaping and validation importable for CPU-only unit tests.
    # The lifespan check below still fails fast if the gateway is actually run.
    AsyncEngineArgs = AsyncLLMEngine = SamplingParams = None


class JsonFormatter(logging.Formatter):
    """Structured JSON logs — required for any real log aggregation (Loki/ELK/CloudWatch)."""
    def format(self, record):
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json_lib.dumps(payload)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger("gateway")

# ---- Auth ----
# Set GATEWAY_API_KEY in the environment (K8s Secret in prod). If unset, the
# gateway refuses to start in "production" mode to avoid accidentally
# deploying an open endpoint. Set GATEWAY_ALLOW_NO_AUTH=1 explicitly for local dev.
API_KEY = os.environ.get("GATEWAY_API_KEY")
ALLOW_NO_AUTH = os.environ.get("GATEWAY_ALLOW_NO_AUTH", "0") == "1"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

if not API_KEY and not ALLOW_NO_AUTH:
    raise RuntimeError(
        "GATEWAY_API_KEY is not set. Refusing to start an unauthenticated gateway. "
        "Set GATEWAY_API_KEY (recommended, e.g. via a K8s Secret) or set "
        "GATEWAY_ALLOW_NO_AUTH=1 explicitly for local development only."
    )


async def verify_api_key(key: str = Depends(api_key_header)):
    if ALLOW_NO_AUTH:
        return True
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


limiter = Limiter(key_func=get_remote_address)

# ---- Request limits (DoS protection) ----
MAX_TOKENS_CAP = int(os.environ.get("MAX_TOKENS_CAP", "2048"))
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "16000"))
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", "50"))

CONFIG_NAME = os.environ.get("MODEL_CONFIG", "fp16")
CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config/models.yaml"))

with open(CONFIG_PATH) as f:
    ALL_CONFIGS = yaml.safe_load(f)

if CONFIG_NAME not in ALL_CONFIGS["configs"]:
    raise ValueError(f"Unknown MODEL_CONFIG={CONFIG_NAME}. Options: {list(ALL_CONFIGS['configs'].keys())}")

CFG = ALL_CONFIGS["configs"][CONFIG_NAME]
SPEC_CFG = ALL_CONFIGS.get("speculative", {"enabled": False})

# ---- Prometheus metrics (Phase 5 depends on these) ----
REQUEST_COUNT = Counter("gateway_requests_total", "Total requests", ["config"])
IN_FLIGHT = Gauge("gateway_inflight_requests", "Requests currently being processed", ["config"])
QUEUE_DEPTH = Gauge("gateway_queue_depth", "Pending requests waiting on the engine", ["config"])
LATENCY_TTFT = Histogram("gateway_ttft_seconds", "Time to first token", ["config"])
LATENCY_E2E = Histogram("gateway_e2e_seconds", "End-to-end request latency", ["config"])
TOKENS_GENERATED = Counter("gateway_tokens_generated_total", "Total output tokens generated", ["config"])
SPEC_ACCEPT_RATE = Gauge("gateway_spec_decode_acceptance_rate", "Speculative decoding acceptance rate", ["config"])

engine: Any = None
_pending = 0


def _check_gpu_supports_fp8():
    """
    FP8 (both W8A8 fp8 and fp8 marlin kernels) requires compute capability
    >= 8.9 (Ada/Hopper: L4, RTX 40xx, H100). It will NOT run on Turing (T4)
    or Ampere (A10/A100) — those fail at engine load with an opaque CUDA
    kernel error, not a helpful message. Check proactively and fail fast
    with a clear message instead.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        major, minor = torch.cuda.get_device_capability(0)
        cc = major + minor / 10
        if cc < 8.9:
            gpu_name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                f"FP8 quantization requires compute capability >= 8.9 (Ada/Hopper: "
                f"L4, RTX 40xx, H100). Detected GPU '{gpu_name}' has compute "
                f"capability {cc}. Use MODEL_CONFIG=fp16 or MODEL_CONFIG=int8 instead, "
                f"or switch quantization to 'awq'/'gptq' with a pre-quantized model "
                f"checkpoint, which work on older GPUs (Turing/Ampere)."
            )
    except ImportError:
        log.warning("torch not importable at config-check time; skipping FP8 GPU capability check.")


def build_engine_args() -> AsyncEngineArgs:
    if AsyncEngineArgs is None:
        raise RuntimeError(
            "vLLM is not installed. Install gateway/requirements.txt on a "
            "supported Linux/CUDA host before starting the gateway."
        )
    kwargs = dict(
        model=CFG["model"],
        dtype=CFG.get("dtype", "auto"),
        max_num_seqs=CFG.get("max_num_seqs", 64),
        max_num_batched_tokens=CFG.get("max_num_batched_tokens", 8192),
        gpu_memory_utilization=CFG.get("gpu_memory_utilization", 0.85),
        enforce_eager=False,
        disable_log_stats=False,
    )
    quant = CFG.get("quantization")
    if quant:
        kwargs["quantization"] = quant
        if quant == "bitsandbytes":
            # vLLM requires BOTH quantization and load_format set to
            # "bitsandbytes" for on-the-fly int8/int4 loading — setting only
            # `quantization` fails at startup with a load_format mismatch error.
            kwargs["load_format"] = "bitsandbytes"
        if quant == "fp8":
            _check_gpu_supports_fp8()
    if SPEC_CFG.get("enabled"):
        kwargs["speculative_model"] = SPEC_CFG["draft_model"]
        kwargs["num_speculative_tokens"] = SPEC_CFG.get("num_speculative_tokens", 5)
    return AsyncEngineArgs(**kwargs)


SHUTTING_DOWN = False
DRAIN_TIMEOUT_S = int(os.environ.get("DRAIN_TIMEOUT_S", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    log.info(f"Loading engine with config '{CONFIG_NAME}' model={CFG['model']}")
    try:
        engine_args = build_engine_args()
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    except Exception:
        log.exception("Engine failed to load - refusing to start")
        raise
    log.info("Engine loaded successfully, gateway is ready")
    yield
    global SHUTTING_DOWN
    SHUTTING_DOWN = True
    log.info(f"Shutdown signal received; draining up to {DRAIN_TIMEOUT_S}s for {_pending} in-flight request(s)")
    waited = 0
    while _pending > 0 and waited < DRAIN_TIMEOUT_S:
        await __import__("asyncio").sleep(1)
        waited += 1
    if _pending > 0:
        log.warning(f"Drain timeout hit with {_pending} request(s) still in-flight; shutting down anyway")
    log.info("Shutting down engine")


app = FastAPI(title="LLM Inference Cost/Latency Optimizer", lifespan=lifespan)
app.state.limiter = limiter

_cors_origins = os.environ.get("GATEWAY_CORS_ORIGINS", "")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return Response(content=json_lib.dumps({"error": "rate limit exceeded"}),
                     status_code=429, media_type="application/json")


class ChatMessage(BaseModel):
    role: str
    content: str

    @field_validator("content")
    @classmethod
    def content_length(cls, v):
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message content exceeds {MAX_MESSAGE_CHARS} characters")
        return v

    @field_validator("role")
    @classmethod
    def role_valid(cls, v):
        if v not in ("system", "user", "assistant"):
            raise ValueError("role must be one of: system, user, assistant")
        return v


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def messages_bounds(cls, v):
        if len(v) == 0:
            raise ValueError("messages must not be empty")
        if len(v) > MAX_MESSAGES:
            raise ValueError(f"too many messages (max {MAX_MESSAGES})")
        return v

    @field_validator("max_tokens")
    @classmethod
    def cap_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return min(v, MAX_TOKENS_CAP)

    @field_validator("temperature")
    @classmethod
    def temp_bounds(cls, v):
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    # Minimal chat template; swap for tokenizer.apply_chat_template in real deployment
    parts = []
    for m in messages:
        parts.append(f"<|{m.role}|>\n{m.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


@app.post("/v1/chat/completions")
@limiter.limit(os.environ.get("GATEWAY_RATE_LIMIT", "30/minute"))
async def chat_completions(request: Request, req: ChatRequest, _auth: bool = Depends(verify_api_key)):
    global _pending

    if SHUTTING_DOWN:
        # Reject new work during drain so the load balancer / k8s can route
        # elsewhere instead of piling onto a pod that's about to die.
        raise HTTPException(status_code=503, detail="Gateway is shutting down, retry against another pod")

    request_id = str(uuid.uuid4())
    prompt = messages_to_prompt(req.messages)
    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    REQUEST_COUNT.labels(config=CONFIG_NAME).inc()
    _pending += 1
    QUEUE_DEPTH.labels(config=CONFIG_NAME).set(_pending)
    IN_FLIGHT.labels(config=CONFIG_NAME).inc()

    start = time.perf_counter()
    first_token_time = None
    output_text = ""
    token_count = 0
    final_output = None  # BUG FIX: was previously undefined if the generator
                          # yielded zero results (e.g. immediate engine error),
                          # causing an UnboundLocalError masking the real error.

    try:
        results_generator = engine.generate(prompt, sampling_params, request_id)
        async for request_output in results_generator:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                LATENCY_TTFT.labels(config=CONFIG_NAME).observe(first_token_time - start)
            final_output = request_output

        if final_output is None:
            raise RuntimeError("engine produced no output for this request")

        output_text = final_output.outputs[0].text
        token_count = len(final_output.outputs[0].token_ids)
        TOKENS_GENERATED.labels(config=CONFIG_NAME).inc(token_count)

        # Speculative decoding acceptance rate, if enabled and exposed by vLLM metrics
        if SPEC_CFG.get("enabled") and hasattr(final_output, "metrics") and final_output.metrics:
            accept_rate = getattr(final_output.metrics, "spec_token_acceptance_rate", None)
            if accept_rate is not None:
                SPEC_ACCEPT_RATE.labels(config=CONFIG_NAME).set(accept_rate)

    except Exception as e:
        log.exception(f"generation failed request_id={request_id}")
        raise HTTPException(status_code=500, detail="internal generation error")
    finally:
        _pending -= 1
        QUEUE_DEPTH.labels(config=CONFIG_NAME).set(_pending)
        IN_FLIGHT.labels(config=CONFIG_NAME).dec()
        LATENCY_E2E.labels(config=CONFIG_NAME).observe(time.perf_counter() - start)

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": CFG["model"],
        "config": CONFIG_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "completion_tokens": token_count,
        }
    }


@app.get("/healthz")
async def healthz():
    """Liveness: process is up and the engine object exists. Kept cheap on purpose."""
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return {"status": "ok", "config": CONFIG_NAME, "model": CFG["model"]}


@app.get("/readyz")
async def readyz():
    """
    Readiness, distinct from liveness. Fails during shutdown drain so k8s
    pulls the pod out of Service endpoints immediately (readinessProbe),
    while the liveness probe stays green until the process actually exits,
    letting in-flight requests finish without new traffic arriving.
    """
    if engine is None or SHUTTING_DOWN:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready", "config": CONFIG_NAME, "in_flight": _pending}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
