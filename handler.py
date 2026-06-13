"""RunPod serverless handler — FLUX.1-schnell image generation (Zeus image gen).

Generates images on a rented CUDA GPU. FLUX.1-schnell is Apache-2.0 (free for
commercial use). Model downloads on first cold start (cached on the worker /
network volume via HF_HOME) so the build stays small and reliable.

Input  (event["input"]):
    prompt        text prompt (required)
    negative      negative prompt (optional)
    width/height  default 768x1024 (portrait)
    steps         default 4 (schnell is distilled for few steps)
    seed          default 0
Output:
    image_b64     PNG, base64
"""
import base64
import io
import os
import time

import runpod

MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
_pipe = None


def _load(hf_token: str | None = None):
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import FluxPipeline
    # FLUX.1-schnell is gated; the token (passed per-request, never baked) is used
    # only for the first download, then the model is cached on the network volume.
    token = hf_token or os.environ.get("HF_TOKEN")
    last = None
    for attempt in range(3):
        try:
            p = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, token=token)
            # FLUX (transformer + T5-XXL + VAE) exceeds 24GB if fully on GPU.
            # CPU offload streams components to the GPU as needed → fits a 24GB card.
            p.enable_model_cpu_offload()
            p.vae.enable_slicing()
            p.vae.enable_tiling()
            _pipe = p
            return _pipe
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"FLUX load failed after retries: {last}")


def handler(event):
    inp = event.get("input") or {}
    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}
    import torch
    try:
        pipe = _load(inp.get("hf_token"))
    except Exception as e:
        return {"error": f"model unavailable: {e}"}
    g = torch.Generator("cuda").manual_seed(int(inp.get("seed", 0)))
    img = pipe(
        prompt,
        guidance_scale=float(inp.get("guidance", 0.0)),
        num_inference_steps=int(inp.get("steps", 4)),
        width=int(inp.get("width", 768)),
        height=int(inp.get("height", 1024)),
        generator=g,
    ).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "model": MODEL, "steps": int(inp.get("steps", 4))}


runpod.serverless.start({"handler": handler})
