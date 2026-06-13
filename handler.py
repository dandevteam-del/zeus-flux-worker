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


def _df(path):
    """Free GB on the filesystem holding `path` (0 if it doesn't exist)."""
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except Exception:
        return 0.0


def _free_volume():
    """Earlier failed FLUX attempts left ~33GB of dead downloads PLUS a large
    Xet partial-download cache on the 50GB volume. `_free_volume` removes
    everything under the HF cache except a fully-present SDXL, so the ~7GB
    SDXL download always fits. Also clears the Xet cache (the real space hog)."""
    import shutil
    root = "/runpod-volume/hf"
    print(f"[free] before: {_df('/runpod-volume'):.1f}GB free on /runpod-volume", flush=True)
    # 1. Xet cache — partial/incomplete downloads from the FLUX Xet attempts.
    for xet in (os.path.join(root, "xet"), os.path.join(root, "hub", "xet")):
        if os.path.isdir(xet):
            shutil.rmtree(xet, ignore_errors=True)
    # 2. Any model cache that isn't SDXL (dead FLUX, etc.).
    hub = os.path.join(root, "hub")
    if os.path.isdir(hub):
        for d in os.listdir(hub):
            full = os.path.join(hub, d)
            if d.startswith("models--") and "stable-diffusion-xl" not in d:
                shutil.rmtree(full, ignore_errors=True)
            elif d in (".locks", "tmp") or d.startswith("tmp"):
                shutil.rmtree(full, ignore_errors=True)
    print(f"[free] after:  {_df('/runpod-volume'):.1f}GB free on /runpod-volume", flush=True)


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import StableDiffusionXLPipeline
    _free_volume()
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
