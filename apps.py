import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation_models
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from io import BytesIO
from scipy import ndimage as ndi
import os
from streamlit_image_comparison import image_comparison
import base64

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "img_size": (512, 512),
    "model_name": "deeplabv3_resnet50",
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "best_seg_model.pth",
    "fg_thresh": 0.3,
}

DEMO_IMAGE_PATHS = {
    "Demo Image 1": "demo/Orginal.jpg",
    "Demo Image 2": "demo/Extracted.jpg"
}

BG_IMAGE_PATHS = {
    "Background 1": "backgrounds/bg1.jpg",
    "Background 2": "backgrounds/bg2.jpg",
    "Background 3": "backgrounds/bg3.jpg",
    "Background 4": "backgrounds/bg4.jpg",
}

CROP_PRESETS = {
    "Freeform": None,
    "Square (1:1)": (1, 1),
    "Instagram Post (4:5)": (4, 5),
    "Instagram Story (9:16)": (9, 16),
    "Landscape (16:9)": (16, 9),
    "Portrait (9:16)": (9, 16),
    "Facebook Cover (16:9)": (16, 9),
    "Twitter Post (16:9)": (16, 9),
}

FILTERS = {
    "None": lambda img: img,
    "Grayscale": lambda img: ImageEnhance.Color(img).enhance(0),
    "Sepia": lambda img: apply_sepia(img),
    "Vintage": lambda img: apply_vintage(img),
    "Cool": lambda img: apply_cool_tone(img),
    "Warm": lambda img: apply_warm_tone(img),
    "High Contrast": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
    "Soft": lambda img: img.filter(ImageFilter.SMOOTH),
    "Sharpen": lambda img: img.filter(ImageFilter.SHARPEN),
    "Blur": lambda img: img.filter(ImageFilter.GaussianBlur(5)),
}

# ============================================================================
# MODERN PROFESSIONAL CSS - PREMIUM GOLD & DARK BLUE THEME
# ============================================================================
CUSTOM_CSS = """
<style>
:root {
    --primary-gold: #d4af37;
    --primary-dark: #0a0e27;
    --secondary-dark: #151932;
    --tertiary-dark: #1a1f3a;
    --accent-blue: #4a90e2;
    --accent-purple: #9b59b6;
    --text-light: #e8e8e8;
    --text-muted: #a0a0a0;
    --border-gold: rgba(212, 175, 55, 0.3);
    --glass-bg: rgba(21, 25, 50, 0.85);
}

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #151932 50%, #1a1f3a 100%);
    background-attachment: fixed;
}

/* Sophisticated header with glassmorphism */
.modern-header {
    background: rgba(21, 25, 50, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 2rem 2.5rem;
    border-radius: 24px;
    text-align: center;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(212, 175, 55, 0.1);
    margin-bottom: 2rem;
    border: 1px solid rgba(212, 175, 55, 0.3);
    position: relative;
    overflow: hidden;
}

.modern-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(212, 175, 55, 0.08) 0%, rgba(155, 89, 182, 0.04) 70%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.modern-header h1 {
    margin: 0;
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, #d4af37 0%, #f4d03f 50%, #c9a227 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    position: relative;
    z-index: 1;
    text-transform: uppercase;
}

.modern-header p {
    margin: 1rem 0 0 0;
    color: #cbd5e1;
    font-size: 1.1rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151932 0%, #1a1f3a 100%);
    border-right: 1px solid rgba(212, 175, 55, 0.2);
}

section[data-testid="stSidebar"] * {
    color: #e8e8e8 !important;
}

section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    margin-bottom: 1rem !important;
}

/* Left sidebar navigation */
.sidebar-nav {
    background: rgba(21, 25, 50, 0.6);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(212, 175, 55, 0.2);
}

.nav-item {
    background: rgba(26, 35, 58, 0.5);
    padding: 0.8rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 1px solid transparent;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.nav-item:hover {
    background: rgba(212, 175, 55, 0.1);
    border-color: rgba(212, 175, 55, 0.3);
    transform: translateX(5px);
}

.nav-item.active {
    background: rgba(212, 175, 55, 0.2);
    border-color: rgba(212, 175, 55, 0.5);
}

/* Main content tabs - horizontal workflow */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(21, 25, 50, 0.6);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 20px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    height: auto;
    min-height: 50px;
    background: rgba(26, 35, 58, 0.5);
    color: #cbd5e1;
    border-radius: 14px;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 0.7rem 1.8rem;
    border: 1px solid rgba(212, 175, 55, 0.2);
    transition: all 0.3s ease;
    white-space: nowrap;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(212, 175, 55, 0.1);
    border-color: rgba(212, 175, 55, 0.3);
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.3) 0%, rgba(155, 89, 182, 0.2) 100%);
    color: #d4af37;
    border-color: rgba(212, 175, 55, 0.5);
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2);
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #d4af37 0%, #c9a227 100%);
    color: #0a0e27;
    border: 1px solid rgba(212, 175, 55, 0.5);
    padding: 0.7rem 1.8rem;
    font-weight: 600;
    border-radius: 12px;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #f4d03f 0%, #d4af37 100%);
    box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
    transform: translateY(-2px);
}

/* File uploader */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(21, 25, 50, 0.4) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px dashed rgba(212, 175, 55, 0.3) !important;
    border-radius: 20px !important;
    padding: 3rem !important;
    transition: all 0.3s ease !important;
}

section[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(212, 175, 55, 0.6) !important;
    background: rgba(21, 25, 50, 0.6) !important;
}

/* Process flow indicator */
.process-flow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(21, 25, 50, 0.6);
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 20px;
    margin: 2rem 0;
    border: 1px solid rgba(212, 175, 55, 0.2);
}

.flow-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
}

.flow-icon {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: rgba(26, 35, 58, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 0.8rem;
    border: 2px solid rgba(212, 175, 55, 0.2);
    transition: all 0.4s ease;
    color: #d4af37;
    font-weight: 600;
}

.flow-step.active .flow-icon {
    background: linear-gradient(135deg, #d4af37 0%, #c9a227 100%);
    border-color: rgba(212, 175, 55, 0.5);
    box-shadow: 0 0 30px rgba(212, 175, 55, 0.5);
    transform: scale(1.1);
    color: #0a0e27;
}

.flow-label {
    color: #a0a0a0;
    font-weight: 500;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.flow-step.active .flow-label {
    color: #d4af37;
    font-weight: 600;
}

.flow-connector {
    height: 3px;
    background: linear-gradient(90deg, rgba(212, 175, 55, 0.2) 0%, rgba(155, 89, 182, 0.2) 100%);
    flex: 1;
    margin: 0 1rem;
    align-self: flex-start;
    margin-top: 30px;
}

/* Settings panel */
.settings-panel {
    background: rgba(21, 25, 50, 0.6);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin-bottom: 1.5rem;
}

.settings-section {
    margin-bottom: 1.5rem;
}

.settings-title {
    color: #d4af37;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Input styling */
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] textarea {
    background: rgba(26, 35, 58, 0.5) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 10px !important;
    color: #e8e8e8 !important;
    padding: 0.6rem !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    background: rgba(21, 25, 50, 0.4);
    padding: 0.8rem;
    border-radius: 12px;
}

/* Image preview cards */
.preview-card {
    background: rgba(21, 25, 50, 0.6);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin-bottom: 1.5rem;
}

.preview-header {
    color: #d4af37;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Footer */
.modern-footer {
    text-align: center;
    padding: 2.5rem;
    background: rgba(21, 25, 50, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin-top: 3rem;
}

.modern-footer h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #d4af37;
    margin: 0;
    text-transform: uppercase;
}

.modern-footer p {
    color: #a0a0a0;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Comparison view styling */
.comparison-controls {
    background: rgba(21, 25, 50, 0.6);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin-bottom: 1.5rem;
}

/* Info boxes */
.info-box {
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid rgba(212, 175, 55, 0.3);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    color: #d4af37;
    margin: 1rem 0;
}

.success-box {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #86efac;
}

.warning-box {
    background: rgba(251, 146, 60, 0.1);
    border: 1px solid rgba(251, 146, 60, 0.3);
    color: #fdba74;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: #0a0e27;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #d4af37 0%, #c9a227 100%);
    border-radius: 10px;
    border: 2px solid #0a0e27;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #f4d03f 0%, #d4af37 100%);
}

/* General text colors */
p, span, div, label {
    color: #e8e8e8;
}

h4, h5, h6 {
    color: #d4af37 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stCaption {
    color: #a0a0a0 !important;
    font-style: italic;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Radio buttons horizontal */
.stRadio > div {
    flex-direction: row;
    gap: 1rem;
}

.stRadio [role="radiogroup"] {
    gap: 1rem;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(26, 35, 58, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(212, 175, 55, 0.2);
}

/* Project cards */
.project-card {
    background: rgba(26, 35, 58, 0.5);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.project-card:hover {
    border-color: rgba(212, 175, 55, 0.3);
    background: rgba(26, 35, 58, 0.7);
}
</style>
"""

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    defaults = {
        'uploaded_images': [],
        'current_image_idx': 0,
        'original_image': None,
        'current_image': None,
        'mask': None,
        'prob_map': None,
        'history': [],
        'history_idx': -1,
        'fg_thresh': CONFIG["fg_thresh"],
        'min_area': 300,
        'extraction_mode': "Black",
        'selected_bg': None,
        'current_step': 1,
        'crop_preset': "Freeform",
        'filter_type': "None",
        'brightness': 1.0,
        'contrast': 1.0,
        'saturation': 1.0,
        'batch_mode': False,
        'custom_color': '#d4af37',
        'resize_percent': 100,
        'show_bg_presets': False,
        'zoom_percentage': 100,
        'blend_slider': 0.5,
        'saved_projects': [],
        'show_profile_section': False,
        'current_project_name': "",
        'active_nav': "upload",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def apply_sepia(img):
    arr = np.array(img).astype(np.float32)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                              [0.349, 0.686, 0.168],
                              [0.272, 0.534, 0.131]])
    h, w, c = arr.shape
    sepia_arr = arr.reshape(-1, 3) @ sepia_filter.T
    sepia_arr = sepia_arr.reshape(h, w, 3)
    sepia_arr = np.clip(sepia_arr, 0, 255)
    return Image.fromarray(sepia_arr.astype(np.uint8))

def apply_vintage(img):
    img = apply_sepia(img)
    img = ImageEnhance.Contrast(img).enhance(0.8)
    img = ImageEnhance.Brightness(img).enhance(0.9)
    return img

def apply_cool_tone(img):
    arr = np.array(img).astype(float)
    arr[:, :, 0] *= 0.9
    arr[:, :, 2] *= 1.1
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def apply_warm_tone(img):
    arr = np.array(img).astype(float)
    arr[:, :, 0] *= 1.1
    arr[:, :, 1] *= 1.05
    arr[:, :, 2] *= 0.9
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

@st.cache_resource
def get_model(path=CONFIG["model_path"]):
    if not Path(path).exists():
        st.error(f"Model file not found: {path}")
        st.info("Please ensure 'best_seg_model.pth' is in the root folder")
        st.stop()

    model = segmentation_models.deeplabv3_resnet50(weights=None, num_classes=CONFIG["num_classes"])
    checkpoint = torch.load(path, map_location=torch.device(CONFIG["device"]))

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(CONFIG["device"])
    model.eval()
    return model

def preprocess_image(img_rgb, size):
    h, w = size
    img = cv2.resize(img_rgb, (w, h)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

def postprocess_mask(prob, thresh, min_area):
    mask = (prob >= thresh).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    out = ndi.binary_fill_holes(out).astype(np.uint8)
    return (out * 255).astype(np.uint8)

@st.cache_data(show_spinner=False)
def predict_mask(_model, img_rgb, device, size):
    h, w = img_rgb.shape[:2]
    inp = preprocess_image(img_rgb, size).to(device)

    with torch.no_grad():
        out = _model(inp)['out']
        prob = F.softmax(out, dim=1).cpu().numpy()[0, 1]

    prob_resized = cv2.resize(prob, (w, h), cv2.INTER_LINEAR)
    return prob_resized

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def apply_background(orig_np, mask_bin, mode, bg_path=None, custom_color=None):
    h, w = orig_np.shape[:2]

    if mode == "Transparent":
        result = np.zeros((h, w, 4), np.uint8)
        result[..., :3] = orig_np
        result[..., 3] = mask_bin * 255
        return Image.fromarray(result, 'RGBA')

    elif mode == "Blur":
        blur = cv2.GaussianBlur(orig_np, (51, 51), 0)
        result = blur.copy()
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result)

    elif mode == "Custom Color" and custom_color:
        hex_color = custom_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        result = np.full_like(orig_np, rgb)
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result)

    elif mode in ["Background 1", "Background 2", "Background 3", "Background 4"] and bg_path:
        if os.path.exists(bg_path):
            bg = np.array(Image.open(bg_path).convert("RGB"))
            bg = cv2.resize(bg, (w, h))
            result = bg.copy()
            result[mask_bin == 1] = orig_np[mask_bin == 1]
            return Image.fromarray(result)

    elif mode == "Custom Image" and st.session_state.selected_bg is not None:
        bg = np.array(st.session_state.selected_bg.convert("RGB"))
        bg = cv2.resize(bg, (w, h))
        result = bg.copy()
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result)

    elif mode == "White":
        result = np.full_like(orig_np, 255)
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result)

    elif mode == "Black":
        result = np.zeros_like(orig_np)
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result)

    return Image.fromarray(orig_np)

def apply_filters_and_adjustments(img):
    filter_func = FILTERS.get(st.session_state.filter_type, FILTERS["None"])
    img = filter_func(img)
    img = ImageEnhance.Brightness(img).enhance(st.session_state.brightness)
    img = ImageEnhance.Contrast(img).enhance(st.session_state.contrast)
    img = ImageEnhance.Color(img).enhance(st.session_state.saturation)
    return img

def crop_image(img, preset):
    if preset == "Freeform" or CROP_PRESETS[preset] is None:
        return img

    ratio = CROP_PRESETS[preset]
    w, h = img.size
    target_ratio = ratio[0] / ratio[1]
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    return img

def get_download_button(image, format_type, quality, button_text, file_name, key):
    buf = BytesIO()
    if format_type == "PNG" and image.mode == "RGBA":
        image.save(buf, format="PNG")
    elif format_type in ["JPEG", "JPG"]:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=quality)
    elif format_type == "WEBP":
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buf, format="WEBP", quality=quality)
    else:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buf, format=format_type, quality=quality)

    return st.download_button(
        button_text,
        buf.getvalue(),
        file_name,
        f"image/{format_type.lower()}",
        key=key,
        use_container_width=True
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image_to_base64(img_array):
    img_pil = Image.fromarray(img_array)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(img_str):
    img_data = base64.b64decode(img_str)
    img_pil = Image.open(BytesIO(img_data))
    return np.array(img_pil)

# ============================================================================
# PROJECT MANAGEMENT
# ============================================================================

def save_project():
    try:
        if st.session_state.get('original_image') is None:
            return False
        
        project_name = st.session_state.get('current_project_name', '').strip()
        if not project_name:
            return False
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        original_img_b64 = image_to_base64(st.session_state.original_image)
        current_img_b64 = image_to_base64(st.session_state.current_image) if st.session_state.current_image is not None else None
        
        mask_b64 = None
        if st.session_state.mask is not None:
            mask_img = (st.session_state.mask * 255).astype(np.uint8)
            if len(mask_img.shape) == 2:
                mask_img = np.stack([mask_img] * 3, axis=-1)
            mask_b64 = image_to_base64(mask_img)
        
        project_data = {
            'name': project_name,
            'timestamp': timestamp,
            'original_image': original_img_b64,
            'current_image': current_img_b64,
            'mask': mask_b64,
            'settings': {
                'fg_thresh': float(st.session_state.get('fg_thresh', 0.4)),
                'min_area': int(st.session_state.get('min_area', 300)),
                'extraction_mode': str(st.session_state.get('extraction_mode', 'Black')),
                'filter_type': str(st.session_state.get('filter_type', 'None')),
                'brightness': float(st.session_state.get('brightness', 1.0)),
                'contrast': float(st.session_state.get('contrast', 1.0)),
                'saturation': float(st.session_state.get('saturation', 1.0)),
                'crop_preset': str(st.session_state.get('crop_preset', 'Freeform')),
                'resize_percent': int(st.session_state.get('resize_percent', 100)),
                'custom_color': str(st.session_state.get('custom_color', '#d4af37')),
            }
        }
        
        if 'saved_projects' not in st.session_state:
            st.session_state.saved_projects = []
        
        existing_index = next((i for i, p in enumerate(st.session_state.saved_projects) if p.get('name') == project_name), -1)
        
        if existing_index >= 0:
            st.session_state.saved_projects[existing_index] = project_data
        else:
            st.session_state.saved_projects.append(project_data)
        
        return True
    except Exception as e:
        st.error(f"Error saving project: {str(e)}")
        return False

def load_project(project_name):
    try:
        for proj in st.session_state.get('saved_projects', []):
            if proj.get('name') == project_name:
                if proj.get('original_image'):
                    st.session_state.original_image = base64_to_image(proj['original_image'])
                
                if proj.get('current_image'):
                    st.session_state.current_image = base64_to_image(proj['current_image'])
                else:
                    st.session_state.current_image = st.session_state.original_image.copy()
                
                if proj.get('mask'):
                    mask_img = base64_to_image(proj['mask'])
                    if len(mask_img.shape) == 3:
                        mask_img = mask_img[:, :, 0]
                    st.session_state.mask = (mask_img > 127).astype(np.uint8)
                
                settings = proj.get('settings', {})
                st.session_state.fg_thresh = float(settings.get('fg_thresh', 0.4))
                st.session_state.min_area = int(settings.get('min_area', 300))
                st.session_state.extraction_mode = str(settings.get('extraction_mode', 'Black'))
                st.session_state.filter_type = str(settings.get('filter_type', 'None'))
                st.session_state.brightness = float(settings.get('brightness', 1.0))
                st.session_state.contrast = float(settings.get('contrast', 1.0))
                st.session_state.saturation = float(settings.get('saturation', 1.0))
                st.session_state.crop_preset = str(settings.get('crop_preset', 'Freeform'))
                st.session_state.resize_percent = int(settings.get('resize_percent', 100))
                st.session_state.custom_color = str(settings.get('custom_color', '#d4af37'))
                st.session_state.current_project_name = project_name
                st.session_state.current_step = 2
                
                return True
        return False
    except Exception as e:
        st.error(f"Error loading project: {str(e)}")
        return False

def delete_project(project_name):
    try:
        st.session_state.saved_projects = [
            p for p in st.session_state.get('saved_projects', []) if p.get('name') != project_name
        ]
        
        if st.session_state.get('current_project_name') == project_name:
            st.session_state.current_project_name = ""
        
        return True
    except Exception as e:
        st.error(f"Error deleting project: {str(e)}")
        return False

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    init_session_state()

    st.set_page_config(
        page_title="OneView Pro - Premium Image Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    model = get_model()

    # Modern Header
    st.markdown("""
    <div class="modern-header">
        <h1>OneView Pro</h1>
        <p>Premium Image Studio - AI-Powered Excellence in Every Pixel</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### Configuration")
        
        with st.expander("Processing Settings", expanded=True):
            st.session_state.fg_thresh = st.slider("Detection Threshold", 0.0, 1.0, st.session_state.fg_thresh, 0.01)
            st.session_state.min_area = st.number_input("Minimum Area (px)", 1, 5000, st.session_state.min_area, 50)
        
        with st.expander("Export Settings", expanded=False):
            export_format = st.selectbox("Format", ["PNG", "JPEG", "JPG", "WEBP"])
            quality = st.slider("Quality", 1, 100, 95) if export_format in ["JPEG", "JPG"] else 95
        
        st.markdown("---")
        
        with st.expander("Project Management", expanded=False):
            project_name = st.text_input("Project Name", value=st.session_state.get('current_project_name', ''))
            
            if project_name != st.session_state.get('current_project_name', ''):
                st.session_state.current_project_name = project_name
            
            if st.button("Save Project", use_container_width=True):
                if project_name.strip() and st.session_state.get('original_image') is not None:
                    if save_project():
                        st.success(f"Project '{project_name}' saved!")
                        st.rerun()
                else:
                    st.warning("Enter project name and upload an image")
            
            st.markdown("---")
            
            if st.session_state.get('saved_projects', []):
                st.markdown("**Saved Projects**")
                for idx, proj in enumerate(st.session_state.saved_projects):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{proj['name']}**")
                            st.caption(f"{proj['timestamp']}")
                        with col2:
                            if st.button("Load", key=f"load_{idx}"):
                                if load_project(proj['name']):
                                    st.success("Loaded!")
                                    st.rerun()
                        
                        if st.button("Delete", key=f"del_{idx}", use_container_width=True):
                            delete_project(proj['name'])
                            st.success("Deleted!")
                            st.rerun()
                        st.markdown("---")

    # Process Flow Indicator
    st.markdown(f"""
    <div class="process-flow">
        <div class="flow-step {'active' if st.session_state.current_step == 1 else ''}">
            <div class="flow-icon">1</div>
            <div class="flow-label">Upload</div>
        </div>
        <div class="flow-connector"></div>
        <div class="flow-step {'active' if st.session_state.current_step == 2 else ''}">
            <div class="flow-icon">2</div>
            <div class="flow-label">Process</div>
        </div>
        <div class="flow-connector"></div>
        <div class="flow-step {'active' if st.session_state.current_step == 3 else ''}">
            <div class="flow-icon">3</div>
            <div class="flow-label">Customize</div>
        </div>
        <div class="flow-connector"></div>
        <div class="flow-step {'active' if st.session_state.current_step == 4 else ''}">
            <div class="flow-icon">4</div>
            <div class="flow-label">Export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload Section
    st.markdown("### Upload Your Image")
    
    uploaded_files = st.file_uploader(
        "Drop your images here or click to browse",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=st.session_state.batch_mode,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.session_state.batch_mode and not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        elif not st.session_state.batch_mode and isinstance(uploaded_files, list):
            uploaded_files = uploaded_files[0]

        if st.session_state.batch_mode:
            st.session_state.uploaded_images = []
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                st.session_state.uploaded_images.append(np.array(img))
            st.success(f"{len(uploaded_files)} images uploaded successfully!")
            st.session_state.current_step = 2
        else:
            img = Image.open(uploaded_files).convert("RGB")
            st.session_state.original_image = np.array(img)
            st.session_state.current_image = np.array(img)
            st.session_state.current_step = 2

            with st.spinner("AI is detecting the subject..."):
                prob = predict_mask(model, st.session_state.original_image, CONFIG["device"], CONFIG["img_size"])
                st.session_state.prob_map = prob
                mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                st.session_state.mask = (mask > 127).astype(np.uint8)
            
            st.success("Subject detected successfully!")

    # Main Editing Interface
    if st.session_state.current_image is not None and st.session_state.mask is not None:
        
        tabs = st.tabs(["Background", "Enhancement", "Crop & Resize", "Compare", "Export"])

        # Background Tab
        with tabs[0]:
            st.markdown("### Background Options")
            
            col_options, col_preview = st.columns([1, 2])
            
            with col_options:
                st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
                st.markdown("#### Choose Style")
                
                bg_options = {
                    "Transparent": "Transparent",
                    "White": "White", 
                    "Black": "Black",
                    "Blur": "Blur",
                    "Custom Color": "Custom Color",
                    "Custom Image": "Custom Image"
                }
                
                for label, mode in bg_options.items():
                    if st.button(label, key=f"bg_{mode}", use_container_width=True):
                        st.session_state.extraction_mode = mode
                        st.session_state.current_step = 3
                        st.rerun()
                
                if st.button("Preset Backgrounds", key="toggle_presets", use_container_width=True):
                    st.session_state.show_bg_presets = not st.session_state.show_bg_presets
                    st.rerun()
                
                if st.session_state.show_bg_presets:
                    st.markdown("---")
                    for bg_key, bg_path in BG_IMAGE_PATHS.items():
                        if os.path.exists(bg_path):
                            if st.button(bg_key, key=f"preset_{bg_key}", use_container_width=True):
                                st.session_state.extraction_mode = bg_key
                                st.rerun()
                
                if st.session_state.extraction_mode == "Custom Color":
                    st.markdown("---")
                    st.session_state.custom_color = st.color_picker("Pick Color", st.session_state.custom_color)
                
                if st.session_state.extraction_mode == "Custom Image":
                    st.markdown("---")
                    custom_bg = st.file_uploader("Upload Background", type=["jpg", "jpeg", "png"], key="custom_bg_upload")
                    if custom_bg:
                        st.session_state.selected_bg = Image.open(custom_bg)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_preview:
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                st.markdown('<div class="preview-header">Live Preview</div>', unsafe_allow_html=True)
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(st.session_state.current_image, st.session_state.mask, 
                                             st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                
                display_img = result_pil.copy()
                display_img.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Enhancement Tab
        with tabs[1]:
            st.markdown("### Enhancement Controls")
            
            col_controls, col_preview = st.columns([1, 2])
            
            with col_controls:
                st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
                
                st.markdown("#### Filter Presets")
                st.session_state.filter_type = st.selectbox("Select Filter", list(FILTERS.keys()))
                
                st.markdown("---")
                st.markdown("#### Adjustments")
                st.session_state.brightness = st.slider("Brightness", 0.0, 2.0, st.session_state.brightness, 0.1)
                st.session_state.contrast = st.slider("Contrast", 0.0, 2.0, st.session_state.contrast, 0.1)
                st.session_state.saturation = st.slider("Saturation", 0.0, 2.0, st.session_state.saturation, 0.1)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_preview:
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                st.markdown('<div class="preview-header">Enhanced Preview</div>', unsafe_allow_html=True)
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(st.session_state.current_image, st.session_state.mask,
                                             st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                result_pil = apply_filters_and_adjustments(result_pil)
                
                display_img = result_pil.copy()
                display_img.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Crop & Resize Tab
        with tabs[2]:
            st.markdown("### Crop & Resize")
            
            col_settings, col_preview = st.columns([1, 2])
            
            with col_settings:
                st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
                
                st.markdown("#### Crop Presets")
                st.session_state.crop_preset = st.selectbox("Select Aspect Ratio", list(CROP_PRESETS.keys()))
                
                st.markdown("---")
                st.markdown("#### Resize")
                
                orig_h, orig_w = st.session_state.current_image.shape[:2]
                st.caption(f"Original: {orig_w} × {orig_h} px")
                
                st.session_state.resize_percent = st.slider("Scale (%)", 10, 200, st.session_state.resize_percent)
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                st.caption(f"New Size: {new_w} × {new_h} px")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_preview:
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                st.markdown('<div class="preview-header">Cropped Preview</div>', unsafe_allow_html=True)
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(st.session_state.current_image, st.session_state.mask,
                                             st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                result_pil = apply_filters_and_adjustments(result_pil)
                result_pil = crop_image(result_pil, st.session_state.crop_preset)
                
                if st.session_state.resize_percent != 100:
                    result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
                
                display_img = result_pil.copy()
                display_img.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Compare Tab
        with tabs[3]:
            st.markdown("### Comparison View")
            
            bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
            result_pil = apply_background(st.session_state.current_image, st.session_state.mask,
                                         st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
            result_pil = apply_filters_and_adjustments(result_pil)
            result_pil = crop_image(result_pil, st.session_state.crop_preset)
            
            if st.session_state.resize_percent != 100:
                orig_w, orig_h = result_pil.size
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)

            st.markdown('<div class="comparison-controls">', unsafe_allow_html=True)
            comparison_mode = st.radio(
                "Comparison Mode",
                ["Side-by-Side", "Interactive Slider", "Blend View"],
                horizontal=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if comparison_mode == "Side-by-Side":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                    st.markdown('<div class="preview-header">Original</div>', unsafe_allow_html=True)
                    original_img = Image.fromarray(st.session_state.original_image)
                    display_orig = original_img.copy()
                    display_orig.thumbnail((600, 600), Image.LANCZOS)
                    st.image(display_orig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                    st.markdown('<div class="preview-header">Processed</div>', unsafe_allow_html=True)
                    display_result = result_pil.copy()
                    display_result.thumbnail((600, 600), Image.LANCZOS)
                    st.image(display_result, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            elif comparison_mode == "Interactive Slider":
                original_img = Image.fromarray(st.session_state.original_image)
                
                if result_pil.mode == 'RGBA':
                    result_rgb = Image.new('RGB', result_pil.size, (255, 255, 255))
                    result_rgb.paste(result_pil, (0, 0), result_pil)
                else:
                    result_rgb = result_pil.convert('RGB')
                
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                image_comparison(
                    img1=original_img, 
                    img2=result_rgb, 
                    label1="Original",
                    label2="Processed"
                )

            elif comparison_mode == "Blend View":
                original_img = Image.fromarray(st.session_state.original_image)
                
                if result_pil.mode == 'RGBA':
                    result_rgb = Image.new('RGB', result_pil.size, (255, 255, 255))
                    result_rgb.paste(result_pil, (0, 0), result_pil)
                else:
                    result_rgb = result_pil.convert('RGB')
                
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                blend_value = st.slider("Blend Amount", 0.0, 1.0, 0.5, 0.01)
                
                orig_rgb = original_img.convert('RGB')
                blended = Image.blend(orig_rgb, result_rgb, float(blend_value))
                
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                display_blended = blended.copy()
                display_blended.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_blended, use_container_width=True, caption=f"Blend: {int(blend_value*100)}%")
                st.markdown('</div>', unsafe_allow_html=True)

        # Export Tab
        with tabs[4]:
            st.markdown("### Export Your Work")
            
            bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
            final_result = apply_background(st.session_state.current_image, st.session_state.mask,
                                           st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
            final_result = apply_filters_and_adjustments(final_result)
            final_result = crop_image(final_result, st.session_state.crop_preset)
            
            if st.session_state.resize_percent != 100:
                orig_w, orig_h = final_result.size
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                final_result = final_result.resize((new_w, new_h), Image.LANCZOS)
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown('<div class="preview-header">Final Result</div>', unsafe_allow_html=True)
            display_final = final_result.copy()
            display_final.thumbnail((900, 700), Image.LANCZOS)
            st.image(display_final, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                get_download_button(final_result, export_format, quality, "Download Final",
                                  f"oneview_final.{export_format.lower()}", "download_final")
            
            with col2:
                buf_orig = BytesIO()
                Image.fromarray(st.session_state.original_image).save(buf_orig, format="PNG")
                st.download_button("Download Original", buf_orig.getvalue(), "original.png",
                                 "image/png", key="download_orig", use_container_width=True)
            
            with col3:
                original_img = Image.fromarray(st.session_state.original_image)
                result_rgb = final_result.convert("RGB")
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                comparison = np.concatenate([np.array(original_img), np.array(result_rgb)], axis=1)
                buf_comp = BytesIO()
                Image.fromarray(comparison).save(buf_comp, format="PNG")
                st.download_button("Download Comparison", buf_comp.getvalue(), "comparison.png",
                                 "image/png", key="download_comp", use_container_width=True)
            
            st.session_state.current_step = 4

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="modern-footer">
        <h3>OneView Pro</h3>
        <p>Premium Image Studio - AI-Powered Excellence in Every Pixel</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8; color: #d4af37;">Developed by Manusha</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
