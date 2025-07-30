# ──────────────────────────────────────────────────────────────────────────────
# Minimal Dockerfile for a RunPod Serverless worker
# Base image: PyTorch 2.1, Python 3.10, CUDA 11.8, Ubuntu 22.04
# ──────────────────────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Use a non‑root workspace directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your worker code into the image
COPY handler.py .

CMD ["python", "handler.py"]
