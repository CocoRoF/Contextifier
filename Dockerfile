# ── Build stage ────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies for binary format handlers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY contextifier/ contextifier/

RUN pip install --no-cache-dir build \
    && python -m build --wheel

# ── Runtime stage ─────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="CocoRoF <gkfua00@gmail.com>"
LABEL description="Contextifier — AI-ready document processing"

# Install Tesseract OCR + Poppler (for pdf2image) + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-kor \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the built wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl

# Non-root user for security
RUN useradd --create-home --shell /bin/bash contextifier
USER contextifier

# Default working directory for documents
WORKDIR /data

ENTRYPOINT ["python", "-m", "contextifier"]
