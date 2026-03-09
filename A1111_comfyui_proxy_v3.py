#!/usr/bin/env python3
"""
ComfyUI A1111 API Proxy - v3
=============================
pip install fastapi uvicorn pillow requests pydantic
uvicorn comfy_a1111_proxy:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests, base64, os, time, io, uuid, random
from PIL import Image

app = FastAPI(title="ComfyUI A1111 Proxy", version="3.0")

# ── Config ──────────────────────────────────────────────────────────
COMFY_URL = "http://localhost:8188"
TEMP_DIR  = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ── Conversion A1111 → ComfyUI ──────────────────────────────────────
SAMPLER_MAP = {
    "DPM++ 2M Karras":     "dpmpp_2m",
    "DPM++ 2M":            "dpmpp_2m",
    "DPM++ SDE Karras":    "dpmpp_sde",
    "DPM++ SDE":           "dpmpp_sde",
    "DPM++ 2M SDE Karras": "dpmpp_2m_sde",
    "DPM++ 2M SDE":        "dpmpp_2m_sde",
    "DPM++ 3M SDE Karras": "dpmpp_3m_sde",
    "DPM++ 3M SDE":        "dpmpp_3m_sde",
    "Euler":               "euler",
    "Euler a":             "euler_ancestral",
    "DDIM":                "ddim",
    "UniPC":               "uni_pc",
    "LMS":                 "lms",
    "Heun":                "heun",
    "DPM2":                "dpm_2",
    "DPM2 a":              "dpm_2_ancestral",
    "DPM fast":            "dpm_fast",
    "DPM adaptive":        "dpm_adaptive",
    "PLMS":                "euler",
}

SCHEDULER_MAP = {
    "DPM++ 2M Karras":     "karras",
    "DPM++ SDE Karras":    "karras",
    "DPM++ 2M SDE Karras": "karras",
    "DPM++ 3M SDE Karras": "karras",
}

def convert_sampler(name: str) -> str:
    return SAMPLER_MAP.get(name, name.lower().replace(" ", "_").replace("+", "p"))

def convert_scheduler(name: str) -> str:
    return SCHEDULER_MAP.get(name, "normal")

def fix_seed(seed: int) -> int:
    return random.randint(0, 2**32 - 1) if seed < 0 else seed

# ── Pydantic Models ─────────────────────────────────────────────────
class ControlNetUnit(BaseModel):
    image: str = ""
    model: str = ""
    module: str = "openpose_full"
    weight: float = 0.95
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    control_mode: str = "Balanced"
    pixel_perfect: bool = False
    processor_res: int = 512
    resize_mode: str = "Crop and Resize"

class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    init_images: List[str]
    width: int = 1024
    height: int = 1024
    denoising_strength: float = 0.75
    seed: int = -1
    steps: int = 25
    cfg_scale: float = 7.0
    sampler_name: str = "Euler"
    batch_size: int = 1
    n_iter: int = 1
    controlnet_units: Optional[List[ControlNetUnit]] = None

class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = -1
    steps: int = 25
    cfg_scale: float = 7.0
    sampler_name: str = "Euler"
    batch_size: int = 1
    n_iter: int = 1

# ── Helpers ─────────────────────────────────────────────────────────
def upload_to_comfy(image_b64: str) -> str:
    img_data = base64.b64decode(image_b64)
    img_name = f"temp_{uuid.uuid4().hex[:8]}.png"
    img_path = os.path.join(TEMP_DIR, img_name)
    Image.open(io.BytesIO(img_data)).save(img_path)
    with open(img_path, "rb") as f:
        r = requests.post(
            f"{COMFY_URL}/upload/image",
            files={"image": (img_name, f, "image/png")},
            timeout=30
        )
    os.unlink(img_path)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Upload error: {r.text}")
    return r.json()["name"]

def submit_workflow(workflow: Dict) -> List[str]:
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow},
        timeout=300
    )
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"ComfyUI error: {r.text}")
    prompt_id = r.json()["prompt_id"]
    while True:
        history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
        if prompt_id in history:
            images_b64 = []
            for node_output in history[prompt_id]["outputs"].values():
                for img_info in node_output.get("images", []):
                    img_resp = requests.get(
                        f"{COMFY_URL}/view",
                        params={
                            "filename": img_info["filename"],
                            "subfolder": img_info.get("subfolder", ""),
                            "type": img_info.get("type", "output")
                        }
                    )
                    images_b64.append(base64.b64encode(img_resp.content).decode())
            return images_b64
        time.sleep(0.5)

# ── Workflows ────────────────────────────────────────────────────────
def build_txt2img_workflow(prompt, negative_prompt, width, height,
                           seed, steps, cfg_scale, sampler_name) -> Dict:
    sampler  = convert_sampler(sampler_name)
    sched    = convert_scheduler(sampler_name)
    seed     = fix_seed(seed)
    steps_b  = int(steps * 0.8)

    return {
        "1":  {"inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},    "class_type": "CheckpointLoaderSimple"},
        "2":  {"inputs": {"ckpt_name": "sd_xl_refiner_1.0.safetensors"}, "class_type": "CheckpointLoaderSimple"},
        "3":  {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "4":  {"inputs": {"clip": ["1", 1], "text": prompt},          "class_type": "CLIPTextEncode"},
        "5":  {"inputs": {"clip": ["1", 1], "text": negative_prompt}, "class_type": "CLIPTextEncode"},
        "6":  {"inputs": {
                   "add_noise": "enable", "noise_seed": seed, "steps": steps,
                   "cfg": cfg_scale, "sampler_name": sampler, "scheduler": sched,
                   "start_at_step": 0, "end_at_step": steps_b,
                   "return_with_leftover_noise": "enable",
                   "model": ["1", 0], "positive": ["4", 0],
                   "negative": ["5", 0], "latent_image": ["3", 0]},
               "class_type": "KSamplerAdvanced"},
        "7":  {"inputs": {"clip": ["2", 1], "text": prompt},          "class_type": "CLIPTextEncode"},
        "8":  {"inputs": {"clip": ["2", 1], "text": negative_prompt}, "class_type": "CLIPTextEncode"},
        "9":  {"inputs": {
                   "add_noise": "disable", "noise_seed": seed, "steps": steps,
                   "cfg": cfg_scale, "sampler_name": sampler, "scheduler": sched,
                   "start_at_step": steps_b, "end_at_step": 10000,
                   "return_with_leftover_noise": "disable",
                   "model": ["2", 0], "positive": ["7", 0],
                   "negative": ["8", 0], "latent_image": ["6", 0]},
               "class_type": "KSamplerAdvanced"},
        "10": {"inputs": {"samples": ["9", 0], "vae": ["2", 2]},      "class_type": "VAEDecode"},
        "11": {"inputs": {"filename_prefix": "ComfyOutput", "images": ["10", 0]}, "class_type": "SaveImage"}
    }

def build_img2img_workflow(img_name, prompt, negative_prompt,
                           width, height, denoising_strength,
                           seed, steps, cfg_scale, sampler_name,
                           controlnet_units=None) -> Dict:
    sampler = convert_sampler(sampler_name)
    sched   = convert_scheduler(sampler_name)
    seed    = fix_seed(seed)

    wf = {
        "1": {"inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},    "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"ckpt_name": "sd_xl_refiner_1.0.safetensors"}, "class_type": "CheckpointLoaderSimple"},
        "3": {"inputs": {"image": img_name},                            "class_type": "LoadImage"},
        "4": {"inputs": {"pixels": ["3", 0], "vae": ["1", 2]},          "class_type": "VAEEncode"},
        "5": {"inputs": {"clip": ["1", 1], "text": prompt},             "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"clip": ["1", 1], "text": negative_prompt},    "class_type": "CLIPTextEncode"},
        "7": {"inputs": {
                  "seed": seed, "steps": steps, "cfg": cfg_scale,
                  "sampler_name": sampler, "scheduler": sched,
                  "denoise": denoising_strength,
                  "model": ["1", 0], "positive": ["5", 0],
                  "negative": ["6", 0], "latent_image": ["4", 0]},
              "class_type": "KSampler"},
        "8": {"inputs": {"samples": ["7", 0], "vae": ["1", 2]},         "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": "ComfyOutput", "images": ["8", 0]}, "class_type": "SaveImage"}
    }

    # ControlNet (sans OpenposePreprocessor — pas d'extension requise)
    if controlnet_units and len(controlnet_units) > 0:
        cn = controlnet_units[0]
        wf["10"] = {
            "inputs": {"control_net_name": cn.model},
            "class_type": "ControlNetLoader"
        }
        wf["11"] = {
            "inputs": {
                "conditioning": ["5", 0],
                "control_net": ["10", 0],
                "image": ["3", 0],
                "strength": cn.weight
            },
            "class_type": "ControlNetApply"
        }
        wf["12"] = {
            "inputs": {
                "conditioning": ["6", 0],
                "control_net": ["10", 0],
                "image": ["3", 0],
                "strength": cn.weight
            },
            "class_type": "ControlNetApply"
        }
        wf["7"]["inputs"]["positive"] = ["11", 0]
        wf["7"]["inputs"]["negative"] = ["12", 0]

    return wf

# ── Endpoints ────────────────────────────────────────────────────────
@app.post("/sdapi/v1/txt2img")
async def txt2img(req: Txt2ImgRequest):
    images = []
    for _ in range(req.n_iter):
        wf = build_txt2img_workflow(
            req.prompt, req.negative_prompt,
            req.width, req.height, req.seed,
            req.steps, req.cfg_scale, req.sampler_name
        )
        images.extend(submit_workflow(wf))
    return {"images": images, "info": "{}"}

@app.post("/sdapi/v1/img2img")
async def img2img(req: Img2ImgRequest):
    img_name = upload_to_comfy(req.init_images[0])
    images = []
    for _ in range(req.n_iter):
        wf = build_img2img_workflow(
            img_name, req.prompt, req.negative_prompt,
            req.width, req.height, req.denoising_strength,
            req.seed, req.steps, req.cfg_scale, req.sampler_name,
            req.controlnet_units
        )
        images.extend(submit_workflow(wf))
    return {"images": images, "info": "{}"}

@app.get("/controlnet/model_list")
async def controlnet_model_list():
    r = requests.get(f"{COMFY_URL}/object_info/ControlNetLoader")
    data = r.json()
    models = data.get("ControlNetLoader", {}).get("input", {}).get("required", {}).get("control_net_name", [[]])[0]
    return {"model_list": models}

@app.get("/sdapi/v1/samplers")
async def samplers():
    return [{"name": k, "aliases": [v]} for k, v in SAMPLER_MAP.items()]

@app.get("/sdapi/v1/sd-models")
async def sd_models():
    r = requests.get(f"{COMFY_URL}/object_info/CheckpointLoaderSimple")
    models = r.json().get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
    return [{"model_name": m, "title": m} for m in models]

@app.get("/health")
async def health():
    try:
        requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        return {"status": "ok", "comfyui": "online", "version": "3.0"}
    except:
        return {"status": "error", "comfyui": "offline"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

