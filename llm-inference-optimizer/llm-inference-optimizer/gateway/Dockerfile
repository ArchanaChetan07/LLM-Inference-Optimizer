FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user: running inference servers as root is an avoidable
# privilege-escalation surface in a multi-tenant cluster.
RUN groupadd -r gateway && useradd -r -g gateway -m -d /home/gateway gateway

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .
COPY config/ config/

# HF model cache: without this, every pod restart re-downloads multi-GB
# weights from Hugging Face. Mount a PVC or emptyDir at this path in k8s
# (see k8s/helm/templates/deployment.yaml volumeMounts).
ENV HF_HOME=/data/hf-cache
RUN mkdir -p /data/hf-cache && chown -R gateway:gateway /data /app

ENV MODEL_CONFIG=fp16
EXPOSE 8000

USER gateway

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
