FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY model ./model
COPY training ./training
RUN python3.11 -m pip install --upgrade pip && python3.11 -m pip install -e .[all]

COPY . .

# Security hardening: Create non-root user and assign permissions
RUN groupadd -g 10001 vajra && \
    useradd -u 10001 -g vajra -m -s /bin/bash vajra && \
    chown -R vajra:vajra /app

USER vajra

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3.11 -c "import model, training" || exit 1

CMD ["python3.11", "-m", "scripts.smoke_test"]
