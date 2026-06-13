"""RunPod serverless handler — SDXL image generation (Zeus image gen).

SDXL base 1.0 (stabilityai/stable-diffusion-xl-base-1.0): ungated, OpenRAIL-M
(commercial use OK), ~7GB. Downloads on first cold start, cached on the network
volume via HF_HOME. Reliable where FLUX (gated + 33GB) was not.

Input (event["input"]): prompt (req), negative, width/height (default 832x1216),
                        steps (default 30), guidance (default 7.0), seed
Output: image_b64 (PNG, base64)
"""
import base64
import io
import os
import time

import runpod

MODEL = os.environ.get("SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
_pipe = None


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import StableDiffusionXLPipeline
    last = None
    for attempt in range(3):
        try:
            p = StableDiffusionXLPipeline.from_pretrained(
                MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
            p.to("cuda")
            _pipe = p
            return _pipe
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"SDXL load failed after retries: {last}")


def handler(event):
    inp = event.get("input") or {}
    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}
    import torch
    try:
        pipe = _load()
    except Exception as e:
        return {"error": f"model unavailable: {e}"}
    g = torch.Generator("cuda").manual_seed(int(inp.get("seed", 0)))
    img = pipe(
        prompt,
        negative_prompt=inp.get("negative", "cartoon, illustration, deformed, extra fingers, blurry, low quality"),
        guidance_scale=float(inp.get("guidance", 7.0)),
        num_inference_steps=int(inp.get("steps", 30)),
        width=int(inp.get("width", 832)),
        height=int(inp.get("height", 1216)),
        generator=g,
    ).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "model": MODEL}


runpod.serverless.start({"handler": handler})
