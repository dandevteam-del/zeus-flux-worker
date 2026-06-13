# Zeus FLUX image worker — RunPod serverless GPU image (built by RunPod from this repo).
# FLUX.1-schnell (Apache-2.0). Model is NOT baked at build (it's ~24GB) — it
# downloads on first cold start, cached via HF_HOME. Point HF_HOME at a RunPod
# network volume (mounted /runpod-volume) so the download persists across workers.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    SDXL_MODEL=stabilityai/stable-diffusion-xl-base-1.0 \
    HF_HUB_DISABLE_XET=1 \
    HF_XET_DISABLE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install diffusers transformers accelerate sentencepiece protobuf \
        runpod huggingface_hub

COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
