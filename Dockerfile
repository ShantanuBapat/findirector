# Dockerfile — FinDirector serving image
#
# WHAT THIS BUILDS:
#   A container image that runs the FinDirector FastAPI app (api/main.py) via
#   uvicorn. Contains only the SERVING dependencies (requirements-serve.txt),
#   CPU-only torch, the app + scripts code, and the BGE-M3 embedding model baked
#   in (so pods start fast without downloading it at runtime).
#
# LAYER ORDER (important for build caching):
#   base image -> system deps -> python deps -> model download -> app code.
#   Code changes (the most frequent) land in the LAST layer, so a code edit
#   doesn't force re-installing dependencies or re-downloading the model.

# --- Base: slim Python, matching your local 3.13 ---
FROM python:3.13-slim

# Don't buffer stdout/stderr (so logs appear immediately in kubectl logs);
# don't write .pyc files (smaller, cleaner image).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# --- System deps: minimal build tools some wheels need, then clean up ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
# Copy ONLY the requirements first so this layer caches unless deps change.
COPY requirements-serve.txt .

# Install CPU-only torch explicitly from PyTorch's CPU index (avoids the huge
# default GPU build), then the rest of the serving deps.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-serve.txt

# --- Bake in the BGE-M3 model (so pods don't download it at startup) ---
# Pre-download the embedding model into the image. Uses the same model id the
# LocalEmbedder uses; cached under the default HF cache path.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# --- App code (LAST — changes most often) ---
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/

# --- Runtime ---
# The app listens on 8000; document it. EKS/Service will map to this.
EXPOSE 8000

# Start the FastAPI app with uvicorn. Host 0.0.0.0 so it's reachable from
# outside the container (not just localhost inside it).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]