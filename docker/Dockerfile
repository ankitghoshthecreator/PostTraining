# Multi-stage Dockerfile
# Target 1: trainer (CUDA base)
# Target 2: evaluator (CPU slim)

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS trainer-base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3", "trainer/train_sft.py", "--config", "configs/curriculum.yaml"]

# Slim CPU evaluator target
FROM python:3.11-slim AS evaluator
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir pydantic>=2.0 groq httpx rouge-score bert-score scikit-learn pandas pyyaml pytest tenacity

COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "eval/run_stage_eval.py", "--stage", "5"]
