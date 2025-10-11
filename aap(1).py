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
# MODERN PROFESSIONAL CSS - GOLD & DARK BLUE THEME
# ============================================================================
CUSTOM_CSS = """
<style>
:root {
    --primary-gold: #d4af37;
    --primary-dark: #0a0e27;
    --secondary-dark: #151932;
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
    border: 1px solid rgba(212, 175, 55, 0.2);
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

/* Styled buttons for Background options and Project Management */
.styled-option-button {
    background: rgba(21, 25, 50, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    color: #e8e8e8 !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

.styled-option-button:hover {
    background: rgba(21, 25, 50, 0.8) !important;
    border-color: rgba(212, 175, 55, 0.4) !important;
    transform: translateY(-2px) !important;
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
        top =
