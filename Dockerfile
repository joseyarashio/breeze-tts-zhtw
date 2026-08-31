# Breeze TTS 2 for Traditional Chinese — standalone GPU container.
# Weights are NOT baked in; mount them:
#   /models/breeze-tts-2   (BreezeBlue/Breeze-TTS-2)
#   /models/breeze-asr     (Breeze-ASR-25, auto-transcribes reference clips)
#
# Needs the upstream submodule checked out (git clone --recurse-submodules).
# Build:  docker compose build
# Run:    docker compose up -d      → http://localhost:7772
#
# torch cu128 wheels bundle their own CUDA runtime, so a plain python base image
# + the NVIDIA container runtime (host driver) is all that's needed (RTX 50-series OK).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    BREEZE_TTS2_HOST=0.0.0.0 \
    BREEZE_TTS2_PORT=7772 \
    BREEZE_TTS2_MODEL_PATH=/models/breeze-tts-2 \
    BREEZE_ASR_MODEL_PATH=/models/breeze-asr \
    BREEZE_ASR_SERVICE_URL= \
    BREEZE_TTS2_FAST=0

# libsndfile (wav/flac/ogg/mp3 decode for soundfile), ffmpeg (torchaudio/qwen-tts helpers)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heavy, rarely-changing layers first so code edits don't re-download torch.
RUN pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
COPY upstream/breeze-tts/requirements.txt ./upstream-requirements.txt
COPY requirements.txt .
RUN pip install -r upstream-requirements.txt -r requirements.txt

# Optional: flash-attn enables the upstream fast path (BREEZE_TTS2_FAST=1).
# Prebuilt wheel only — never compile in the image; skip silently if unavailable.
ARG INSTALL_FLASH_ATTN=1
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      pip install flash-attn --no-build-isolation \
        --find-links https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3 \
        2>/dev/null || echo ">> flash-attn wheel not available for this torch/python; fast path disabled"; \
    fi

COPY upstream/breeze-tts ./upstream/breeze-tts
COPY app ./app
RUN mkdir -p outputs uploads

EXPOSE 7772
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://127.0.0.1:7772/health || exit 1

CMD ["python", "app/server.py"]
