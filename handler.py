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
    """The 50GB volume kept ending up full of dead FLUX downloads + Xet partials
    that surgical deletes missed. Decisive fix: if free space is low, NUKE the
    entire HF cache and re-download SDXL clean (~7GB, ~1 min). The volume holds
    nothing precious — it's a model-download scratch cache."""
    import shutil
    root = "/runpod-volume/hf"
    free = _df("/runpod-volume")
    print(f"[free] before: {free:.1f}GB free on /runpod-volume", flush=True)
    # List what's eating the volume (debug — visible in worker logs).
    for base in ("/runpod-volume", root, os.path.join(root, "hub")):
        if os.path.isdir(base):
            try:
                print(f"[free] {base}: {os.listdir(base)}", flush=True)
            except Exception:
                pass
    sdxl_ok = False
    hub = os.path.join(root, "hub")
    if os.path.isdir(hub):
        for d in os.listdir(hub):
            if "stable-diffusion-xl" in d:
                # crude completeness check: snapshot dir has the big unet weight
                snap = os.path.join(hub, d, "snapshots")
                sdxl_ok = os.path.isdir(snap) and any(
                    os.path.exists(os.path.join(snap, s, "unet", "diffusion_pytorch_model.fp16.safetensors"))
                    for s in (os.listdir(snap) if os.path.isdir(snap) else []))
    # Wipe everything unless a complete SDXL is already cached AND we have room.
    if not (sdxl_ok and free > 8):
        print("[free] wiping entire HF cache for a clean SDXL download", flush=True)
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(hub, exist_ok=True)
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
