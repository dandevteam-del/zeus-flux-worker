# zeus-flux-worker

RunPod serverless worker for **FLUX.1-schnell** image generation — Zeus's own
image generator (commercial-safe, Apache-2.0). RunPod builds this image from the
Dockerfile and runs `handler.py` as a serverless endpoint.

- **Input** (`event["input"]`): `prompt` (required), `negative`, `width`/`height` (default 768×1024), `steps` (default 4), `seed`
- **Output**: `image_b64` — PNG, base64

The model (~24GB) downloads on first cold start and caches via `HF_HOME`.

## Deploy notes (important)

- **Attach a network volume** (≥40 GB) to the endpoint so the FLUX weights
  persist at `/runpod-volume/hf` across workers (otherwise every cold start
  re-downloads 24 GB). Set the endpoint's **Container Disk ≥ 20 GB** too.
- GPU: a 24 GB card (e.g. RTX 4090 / L4) runs FLUX.1-schnell comfortably.
- Active workers 0 (scale to zero), max 1 — keep cost near zero.

Client: Zeus calls this endpoint with a prompt and saves the returned PNG as the
avatar reference image.
