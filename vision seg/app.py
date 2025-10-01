# backend/app.py
"""
FastAPI backend for segmentation inference.
Endpoints:
  - POST /predict  : accept multipart/form-data file=image, threshold (float)
  - GET  /health   : simple health check
Returns JSON with base64 PNG data:
  { overlay: "data:image/png;base64,...", mask: "...", prob: "...", cutout: "..." }
"""
import io
import os
import base64
from pathlib import Path
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from PIL import Image, ImageOps, ImageChops, ImageFilter
import numpy as np
import torch
import torchvision.transforms as T

# ---------- Config ----------
MODEL_PATH = os.environ.get("SEG_MODEL_PATH", str(Path(__file__).parent / "model_scripted.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="VisionSeg API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utility ----------
def to_base64_png_bytes(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    return tuple(int(hex_color[i:i+lv//3], 16) for i in range(0, lv, lv//3))

# ---------- Robust loader (jit -> torch.load -> state_dict candidates) ----------
def _strip_prefix_from_state(state_dict, prefix="module."):
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith(prefix):
            new_k = new_k[len(prefix):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        if new_k.startswith("network."):
            new_k = new_k[len("network."):]
        new_state[new_k] = v
    return new_state

def load_model(model_path: str):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    # 1) Try TorchScript
    try:
        jm = torch.jit.load(str(p), map_location=DEVICE)
        jm.eval()
        jm.to(DEVICE)
        print("Loaded TorchScript model via torch.jit.load()")
        return jm
    except Exception:
        pass

    # 2) Try torch.load (full model object)
    try:
        obj = torch.load(str(p), map_location=DEVICE)
    except Exception as e:
        raise RuntimeError(f"Failed to torch.load('{p}'): {e}")

    if not isinstance(obj, dict):
        # assume full model object
        model = obj
        model.eval()
        model.to(DEVICE)
        print("Loaded full model object via torch.load()")
        return model

    # 3) treat as state_dict
    state = obj
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    state = _strip_prefix_from_state(state, prefix="module.")
    state = _strip_prefix_from_state(state, prefix="model.")
    sample_keys = list(state.keys())[:16]

    def try_candidate(candidate_model):
        try:
            candidate_model.to(DEVICE)
            candidate_model.eval()
            res = candidate_model.load_state_dict(state, strict=False)
            missing = getattr(res, "missing_keys", None)
            unexpected = getattr(res, "unexpected_keys", None)
            return True, candidate_model, missing, unexpected
        except Exception as e:
            return False, str(e), None, None

    tried = []
    # Candidate: fcn_resnet50
    try:
        from torchvision.models.segmentation import fcn_resnet50
        cand = fcn_resnet50(pretrained=False, num_classes=1)
        ok, model_or_err, missing, unexpected = try_candidate(cand)
        tried.append(("fcn_resnet50", ok))
        if ok:
            print("Loaded into fcn_resnet50 (state_dict with strict=False)")
            return model_or_err.to(DEVICE).eval()
    except Exception:
        tried.append(("fcn_resnet50", False))

    # Candidate: deeplabv3_resnet50
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
        cand = deeplabv3_resnet50(pretrained=False, num_classes=1)
        ok, model_or_err, missing, unexpected = try_candidate(cand)
        tried.append(("deeplabv3_resnet50", ok))
        if ok:
            print("Loaded into deeplabv3_resnet50 (state_dict with strict=False)")
            return model_or_err.to(DEVICE).eval()
    except Exception:
        tried.append(("deeplabv3_resnet50", False))

    # Candidate: simple UNet fallback
    try:
        import torch.nn as nn
        class DoubleConv(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            def forward(self, x): return self.net(x)

        class SimpleUNet(nn.Module):
            def __init__(self, in_ch=3, out_ch=1, features=[64,128,256]):
                super().__init__()
                self.encs = nn.ModuleList()
                self.pools = nn.ModuleList()
                prev = in_ch
                for f in features:
                    self.encs.append(DoubleConv(prev, f))
                    self.pools.append(nn.MaxPool2d(2))
                    prev = f
                self.bottleneck = DoubleConv(prev, prev*2)
                self.upconvs = nn.ModuleList()
                self.decs = nn.ModuleList()
                for f in reversed(features):
                    self.upconvs.append(nn.ConvTranspose2d(prev*2, f, kernel_size=2, stride=2))
                    self.decs.append(DoubleConv(f*2, f))
                    prev = f
                self.final = nn.Conv2d(prev, out_ch, kernel_size=1)
            def forward(self, x):
                skips = []
                for i, enc in enumerate(self.encs):
                    x = enc(x)
                    skips.append(x)
                    x = self.pools[i](x)
                x = self.bottleneck(x)
                for i in range(len(self.upconvs)):
                    x = self.upconvs[i](x)
                    skip = skips[-(i+1)]
                    if x.shape[2:] != skip.shape[2:]:
                        _,_,h,w = x.shape
                        skip = skip[:, :, :h, :w]
                    x = torch.cat([skip, x], dim=1)
                    x = self.decs[i](x)
                x = self.final(x)
                return x

        cand = SimpleUNet(in_ch=3, out_ch=1, features=[32,64,128])
        ok, model_or_err, missing, unexpected = try_candidate(cand)
        tried.append(("SimpleUNet", ok))
        if ok:
            print("Loaded into SimpleUNet (state_dict with strict=False)")
            return model_or_err.to(DEVICE).eval()
    except Exception:
        tried.append(("SimpleUNet", False))

    raise RuntimeError(
        "Could not load state_dict into known architectures. Sample keys: "
        f"{sample_keys}. Tried candidates: {tried}. "
        "If you used a custom model, provide the class or use TorchScript (.pt)."
    )

# Load once at startup
MODEL = None
try:
    print("Loading model from:", MODEL_PATH)
    MODEL = load_model(MODEL_PATH)
    print("Model loaded successfully. Device:", DEVICE)
except Exception as e:
    print("Model load at startup failed:", e)
    MODEL = None

# ---------- Inference helpers ----------
def preprocess_image(pil_img: Image.Image, target_size=(512,512)):
    img = pil_img
    if target_size:
        img = ImageOps.contain(img, target_size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def postprocess_mask(mask_tensor: torch.Tensor, orig_size: tuple, threshold: float):
    with torch.no_grad():
        if mask_tensor.max() > 1.0 or mask_tensor.min() < 0.0:
            probs = torch.sigmoid(mask_tensor)
        else:
            probs = mask_tensor
        probs = probs.squeeze().cpu().numpy()
    probs_img = Image.fromarray((probs * 255).astype("uint8"))
    probs_img = probs_img.resize(orig_size, resample=Image.BILINEAR)
    probs_np = np.array(probs_img).astype("float32")/255.0
    bin_mask = (probs_np >= threshold).astype("uint8")*255
    return Image.fromarray(bin_mask.astype("uint8"), mode="L"), probs_img

def overlay_mask_on_image(orig_img: Image.Image, mask_img: Image.Image, color=(255,0,0), alpha=0.5):
    base = orig_img.convert("RGBA")
    mask = mask_img.convert("L")
    overlay_color = Image.new("RGBA", base.size, color + (int(alpha*255),))
    colored = Image.composite(overlay_color, Image.new("RGBA", base.size, (0,0,0,0)), mask)
    out = Image.alpha_composite(base, colored)
    return out.convert("RGB")

def apply_cutout(orig_img: Image.Image, bin_mask: Image.Image, background=(0,0,0)):
    base = orig_img.convert("RGBA")
    mask = bin_mask.convert("L")
    bg = Image.new("RGBA", base.size, background + (255,))
    cutout = Image.composite(base, bg, mask)
    return cutout.convert("RGB")

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": MODEL is not None, "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = Form(0.3), resize: int = Form(512), overlay_color: str = Form("#ff3b30")):
    """
    Accepts multipart form:
      - file: image file
      - threshold: 0.0..1.0
      - resize: integer (model input size, e.g., 512) or 0 for no resize
      - overlay_color: hex like "#ff3b30"
    Returns JSON with base64 images: overlay, mask, prob_map, cutout
    """
    global MODEL
    if MODEL is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded on server. Check logs."})

    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_size = pil_img.size
    resize_for_model = (resize, resize) if resize and resize > 0 else None

    # preprocess & run
    input_tensor = preprocess_image(pil_img, target_size=resize_for_model).to(DEVICE)
    with torch.no_grad():
        out = MODEL(input_tensor)
        if isinstance(out, dict):
            mask = out.get("out", next(iter(out.values())))
        else:
            mask = out
        # unify shape
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = mask[:,0,:,:]
            else:
                probs = torch.softmax(mask, dim=1)
                cls = min(1, mask.shape[1]-1)
                mask = probs[:, cls, :, :]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

    bin_mask_img, prob_img = postprocess_mask(mask, orig_size, threshold)
    color_rgb = hex_to_rgb(overlay_color)
    overlay_img = overlay_mask_on_image(pil_img, bin_mask_img, color=color_rgb, alpha=0.5)
    cutout_img = apply_cutout(pil_img, bin_mask_img)

    return {
        "overlay": to_base64_png_bytes(overlay_img),
        "mask": to_base64_png_bytes(bin_mask_img.convert("L")),
        "prob_map": to_base64_png_bytes(prob_img.convert("L")),
        "cutout": to_base64_png_bytes(cutout_img)
    }
