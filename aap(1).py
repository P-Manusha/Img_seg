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
# ELEGANT PROFESSIONAL CSS DESIGN
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=Montserrat:wght@300;400;500;600;700&display=swap');

:root {
    --primary-gold: #D4AF37;
    --secondary-gold: #F4E4C1;
    --dark-navy: #0A1929;
    --medium-navy: #132F4C;
    --light-navy: #1E4976;
    --accent-silver: #C0C0C0;
    --text-light: #E7EBF0;
    --text-muted: #B2BAC2;
}

* {
    font-family: 'Inter', sans-serif;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stApp {
    background: linear-gradient(180deg, #0A1929 0%, #001E3C 50%, #0A1929 100%);
    background-attachment: fixed;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 15% 20%, rgba(212, 175, 55, 0.03) 0%, transparent 25%),
        radial-gradient(circle at 85% 80%, rgba(212, 175, 55, 0.02) 0%, transparent 25%),
        repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(212, 175, 55, 0.01) 2px, rgba(212, 175, 55, 0.01) 4px);
    pointer-events: none;
    z-index: 0;
}

/* Elegant Header */
.elegant-header {
    position: relative;
    text-align: center;
    padding: 4rem 2rem 3rem;
    margin-bottom: 3rem;
    overflow: hidden;
}

.elegant-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 1200px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary-gold), transparent);
}

.elegant-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary-gold), transparent);
}

.elegant-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 800;
    color: transparent;
    background: linear-gradient(135deg, #D4AF37 0%, #F4E4C1 50%, #D4AF37 100%);
    -webkit-background-clip: text;
    background-clip: text;
    letter-spacing: 8px;
    margin: 0 0 1rem 0;
    text-transform: uppercase;
    position: relative;
    text-shadow: 0 0 80px rgba(212, 175, 55, 0.5);
}

.elegant-header .subtitle {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.1rem;
    font-weight: 300;
    color: var(--text-muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 1rem;
}

.elegant-header .tagline {
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    font-weight: 400;
    color: var(--text-light);
    max-width: 700px;
    margin: 1.5rem auto 0;
    line-height: 1.8;
    opacity: 0.9;
}

/* Navigation Pills */
.nav-container {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin: 2.5rem 0;
    padding: 0 2rem;
    flex-wrap: wrap;
}

.nav-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, rgba(19, 47, 76, 0.6) 0%, rgba(10, 25, 41, 0.6) 100%);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 50px;
    color: var(--text-light);
    font-weight: 500;
    font-size: 0.95rem;
    letter-spacing: 1px;
    cursor: pointer;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.nav-pill:hover {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(19, 47, 76, 0.8) 100%);
    border-color: var(--primary-gold);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(212, 175, 55, 0.3);
}

.nav-pill.active {
    background: linear-gradient(135deg, #D4AF37 0%, #B8941F 100%);
    border-color: var(--secondary-gold);
    color: var(--dark-navy);
    font-weight: 600;
    box-shadow: 0 8px 35px rgba(212, 175, 55, 0.5);
}

/* Workflow Steps */
.workflow-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 3rem 0;
    padding: 0 2rem;
}

.workflow-step {
    position: relative;
    text-align: center;
    padding: 2rem 1.5rem;
    background: linear-gradient(135deg, rgba(19, 47, 76, 0.4) 0%, rgba(10, 25, 41, 0.4) 100%);
    border: 1px solid rgba(212, 175, 55, 0.15);
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

.workflow-step::before {
    content: attr(data-step);
    position: absolute;
    top: -15px;
    left: 50%;
    transform: translateX(-50%);
    width: 45px;
    height: 45px;
    background: linear-gradient(135deg, var(--primary-gold) 0%, var(--secondary-gold) 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--dark-navy);
    box-shadow: 0 4px 20px rgba(212, 175, 55, 0.4);
}

.workflow-step.active {
    border-color: var(--primary-gold);
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(19, 47, 76, 0.6) 100%);
    box-shadow: 0 8px 40px rgba(212, 175, 55, 0.3);
}

.workflow-step .step-icon {
    font-size: 2.5rem;
    margin: 1.5rem 0 1rem;
}

.workflow-step .step-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-light);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Content Cards */
.content-card {
    background: linear-gradient(135deg, rgba(19, 47, 76, 0.5) 0%, rgba(10, 25, 41, 0.5) 100%);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 25px;
    padding: 2.5rem;
    margin: 2rem 0;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 50px rgba(0, 0, 0, 0.4);
}

.content-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-gold);
    margin: 0 0 1.5rem 0;
    letter-spacing: 2px;
}

.content-card h4 {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-light);
    margin: 1.5rem 0 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 25, 41, 0.95) 0%, rgba(19, 47, 76, 0.95) 100%);
    border-right: 1px solid rgba(212, 175, 55, 0.3);
    backdrop-filter: blur(20px);
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 1px;
    height: 100%;
    background: linear-gradient(180deg, transparent, var(--primary-gold), transparent);
    opacity: 0.5;
}

section[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--primary-gold) !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
    letter-spacing: 2px !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.8rem !important;
    border-bottom: 1px solid rgba(212, 175, 55, 0.2) !important;
}

section[data-testid="stSidebar"] label {
    font-family: 'Montserrat', sans-serif !important;
    color: var(--text-light) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
}

section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] input[type="text"] {
    background: rgba(19, 47, 76, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    border-radius: 12px !important;
    color: var(--text-light) !important;
    font-weight: 500 !important;
    padding: 0.7rem !important;
}

section[data-testid="stSidebar"] .stNumberInput input:focus,
section[data-testid="stSidebar"] .stSelectbox select:focus {
    border-color: var(--primary-gold) !important;
    box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.2) !important;
}

/* Elegant Buttons */
.stButton > button {
    background: linear-gradient(135deg, #D4AF37 0%, #B8941F 100%);
    color: var(--dark-navy);
    border: 1px solid var(--secondary-gold);
    padding: 0.9rem 2.5rem;
    font-weight: 600;
    font-family: 'Montserrat', sans-serif;
    border-radius: 50px;
    font-size: 1rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    box-shadow: 0 8px 30px rgba(212, 175, 55, 0.4);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #F4E4C1 0%, #D4AF37 100%);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(212, 175, 55, 0.6);
    border-color: var(--primary-gold);
}

.stButton > button:active {
    transform: translateY(-1px);
    box-shadow: 0 6px 25px rgba(212, 175, 55, 0.5);
}

/* File Uploader */
section[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, rgba(19, 47, 76, 0.3) 0%, rgba(10, 25, 41, 0.3) 100%) !important;
    border: 2px dashed rgba(212, 175, 55, 0.4) !important;
    border-radius: 25px !important;
    padding: 4rem 2rem !important;
    backdrop-filter: blur(10px) !important;
}

section[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--primary-gold) !important;
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(19, 47, 76, 0.4) 100%) !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted);
    border: none;
    border-bottom: 3px solid transparent;
    font-family: 'Montserrat', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 1rem 2rem;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-light);
    background: rgba(212, 175, 55, 0.05);
}

.stTabs [aria-selected="true"] {
    background: transparent;
    color: var(--primary-gold);
    border-bottom-color: var(--primary-gold);
    font-weight: 600;
}

/* Sliders */
.stSlider {
    padding: 1.5rem 0;
}

.stSlider [data-baseweb="slider"] {
    background: rgba(19, 47, 76, 0.4);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid rgba(212, 175, 55, 0.15);
}

/* Radio Buttons */
.stRadio > div {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.stRadio label {
    background: rgba(19, 47, 76, 0.5);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    color: var(--text-light);
    font-family: 'Montserrat', sans-serif;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
}

.stRadio label:hover {
    background: rgba(212, 175, 55, 0.1);
    border-color: var(--primary-gold);
}

/* Footer */
.elegant-footer {
    margin-top: 5rem;
    padding: 4rem 2rem;
    text-align: center;
    position: relative;
    border-top: 1px solid rgba(212, 175, 55, 0.2);
}

.elegant-footer::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary-gold), transparent);
}

.elegant-footer h2 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 800;
    color: transparent;
    background: linear-gradient(135deg, #D4AF37 0%, #F4E4C1 100%);
    -webkit-background-clip: text;
    background-clip: text;
    letter-spacing: 6px;
    margin-bottom: 1rem;
}

.elegant-footer p {
    font-family: 'Montserrat', sans-serif;
    color: var(--text-muted);
    font-size: 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.elegant-footer .credits {
    margin-top: 2rem;
    font-family: 'Inter', sans-serif;
    color: var(--text-light);
    font-size: 0.95rem;
    opacity: 0.8;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: var(--dark-navy);
    border-left: 1px solid rgba(212, 175, 55, 0.1);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #D4AF37 0%, #B8941F 100%);
    border-radius: 10px;
    border: 2px solid var(--dark-navy);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #F4E4C1 0%, #D4AF37 100%);
}

/* Utilities */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

p, span, div, li {
    color: var(--text-light);
}

.stCaption {
    color: var(--text-muted) !important;
    font-style: italic;
}

/* Info/Warning/Success boxes */
.stAlert {
    background: rgba(19, 47, 76, 0.5) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    border-radius: 15px !important;
    backdrop-filter: blur(10px) !important;
}

/* Divider */
hr {
    border-color: rgba(212, 175, 55, 0.2) !important;
    margin: 2rem 0 !important;
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
        'custom_color': '#00FF00',
        'resize_percent': 100,
        'show_bg_presets': False,
        'zoom_percentage': 100,
        'blend_slider': 0.5,
        'saved_projects': [],
        'show_profile_section': False,
        'current_project_name': "",
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
        st.error(f"❌ Model file not found: {path}")
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
                'custom_color': str(st.session_state.get('custom_color', '#00FF00')),
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
                st.session_state.custom_color = str(settings.get('custom_color', '#00FF00'))
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
        page_title="OneView | Professional Image Editor",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    model = get_model()

    # Elegant Header
    st.markdown("""
    <div class="elegant-header">
        <h1>OneView</h1>
        <div class="subtitle">Professional Image Editing Suite</div>
        <div class="tagline">
            Precision AI-powered subject extraction with elegant refinement tools.
            Transform your images with sophistication and artistry.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Detection Parameters")
        st.session_state.fg_thresh = st.slider("Detection Threshold", 0.0, 1.0, st.session_state.fg_thresh, 0.01)
        st.session_state.min_area = st.number_input("Minimum Area (px)", 1, 5000, st.session_state.min_area, 50)
        
        st.markdown("---")
        st.markdown("### 📦 Export Configuration")
        export_format = st.selectbox("Format", ["PNG", "JPEG", "JPG", "WEBP"])
        quality = st.slider("Quality", 1, 100, 95) if export_format in ["JPEG", "JPG"] else 95
        
        st.markdown("---")
        st.markdown("### 🔄 Processing Mode")
        st.session_state.batch_mode = st.checkbox("Batch Processing", value=st.session_state.batch_mode)
        
        st.markdown("---")
        st.markdown("### 👤 Project Management")
        
        if st.button("📁 My Projects", use_container_width=True):
            st.session_state.show_profile_section = not st.session_state.show_profile_section
        
        if st.session_state.get('show_profile_section', False):
            st.markdown("---")
            project_name = st.text_input("📝 Project Name", value=st.session_state.get('current_project_name', ''))
            
            if project_name != st.session_state.get('current_project_name', ''):
                st.session_state.current_project_name = project_name
            
            if st.button("💾 Save Project", use_container_width=True):
                if project_name.strip() and st.session_state.get('original_image') is not None:
                    if save_project():
                        st.success(f"✅ Saved: {project_name}")
                        st.rerun()
                else:
                    st.warning("⚠️ Project name and image required")
            
            st.markdown("---")
            
            if st.session_state.get('saved_projects', []):
                st.markdown("**📚 Saved Projects**")
                for idx, proj in enumerate(st.session_state.saved_projects):
                    with st.container():
                        st.markdown(f"**{proj['name']}**")
                        st.caption(f"🕒 {proj['timestamp']}")
                        
                        if proj.get('original_image'):
                            try:
                                thumb_img = base64_to_image(proj['original_image'])
                                thumb_pil = Image.fromarray(thumb_img)
                                thumb_pil.thumbnail((150, 150), Image.LANCZOS)
                                st.image(thumb_pil, use_container_width=True)
                            except:
                                st.caption("📷 Preview unavailable")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📂 Load", key=f"load_{idx}", use_container_width=True):
                                if load_project(proj['name']):
                                    st.success("✅ Loaded!")
                                    st.rerun()
                        
                        with col2:
                            if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True):
                                delete_project(proj['name'])
                                st.success("🗑️ Deleted!")
                                st.rerun()
                        st.markdown("---")

    # Workflow Steps Indicator
    steps_data = [
        ("1", "📤", "Upload", st.session_state.current_step == 1),
        ("2", "✂️", "Extract", st.session_state.current_step == 2),
        ("3", "🎨", "Customize", st.session_state.current_step == 3),
        ("4", "💾", "Export", st.session_state.current_step == 4),
    ]
    
    st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
    for step_num, icon, title, is_active in steps_data:
        active_class = "active" if is_active else ""
        st.markdown(f"""
        <div class="workflow-step {active_class}" data-step="{step_num}">
            <div class="step-icon">{icon}</div>
            <div class="step-title">{title}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Demo Section
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("<h3>✨ Showcase Gallery</h3>", unsafe_allow_html=True)
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("#### Original Image")
        if os.path.exists(DEMO_IMAGE_PATHS["Demo Image 1"]):
            demo_orig = Image.open(DEMO_IMAGE_PATHS["Demo Image 1"])
            demo_orig.thumbnail((500, 500), Image.Resampling.LANCZOS)
            st.image(demo_orig, use_container_width=True, caption="Before Processing")
        else:
            st.info("Demo image not available")

    with demo_col2:
        st.markdown("#### Extracted Subject")
        if os.path.exists(DEMO_IMAGE_PATHS["Demo Image 2"]):
            demo_extracted = Image.open(DEMO_IMAGE_PATHS["Demo Image 2"])
            demo_extracted.thumbnail((500, 500), Image.Resampling.LANCZOS)
            st.image(demo_extracted, use_container_width=True, caption="After Processing")
        else:
            st.info("Demo image not available")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload Section
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("<h3>📤 Upload Your Images</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drag and drop your images here or click to browse",
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
            st.success(f"✅ Successfully uploaded {len(uploaded_files)} images")
            st.session_state.current_step = 2
        else:
            img = Image.open(uploaded_files).convert("RGB")
            st.session_state.original_image = np.array(img)
            st.session_state.current_image = np.array(img)
            st.session_state.current_step = 2

            with st.spinner("🔍 AI is analyzing your image..."):
                prob = predict_mask(model, st.session_state.original_image, CONFIG["device"], CONFIG["img_size"])
                st.session_state.prob_map = prob
                mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                st.session_state.mask = (mask > 127).astype(np.uint8)
            
            st.success("✅ Subject detected successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Main Editing Interface
    if st.session_state.current_image is not None or st.session_state.batch_mode:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("<h3>🎨 Creative Studio</h3>", unsafe_allow_html=True)
        
        tabs = st.tabs(["🖼️ Background", "🎛️ Filters & Effects", "✂️ Crop & Resize", "📊 Final Preview"])

        # TAB 1: Background
        with tabs[0]:
            st.markdown("#### Background Selection")
            
            bg_col1, bg_col2 = st.columns([1, 2])
            
            with bg_col1:
                st.markdown("**Quick Actions**")
                
                if st.button("✨ Transparent PNG", key="bg_transparent", use_container_width=True):
                    st.session_state.extraction_mode = "Transparent"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("⚪ White Background", key="bg_white", use_container_width=True):
                    st.session_state.extraction_mode = "White"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("⚫ Black Background", key="bg_black", use_container_width=True):
                    st.session_state.extraction_mode = "Black"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("💫 Blur Effect", key="bg_blur", use_container_width=True):
                    st.session_state.extraction_mode = "Blur"
                    st.session_state.current_step = 3
                    st.rerun()

                st.markdown("---")
                st.markdown("**Custom Options**")

                if st.button("🎨 Color Picker", key="bg_custom_color", use_container_width=True):
                    st.session_state.extraction_mode = "Custom Color"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("🖼️ Upload Background", key="bg_custom_image", use_container_width=True):
                    st.session_state.extraction_mode = "Custom Image"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("📚 Preset Library", key="toggle_presets", use_container_width=True):
                    st.session_state.show_bg_presets = not st.session_state.show_bg_presets
                    st.rerun()

                if st.session_state.show_bg_presets:
                    st.markdown("---")
                    st.markdown("**Presets**")
                    for bg_key, bg_path in BG_IMAGE_PATHS.items():
                        if os.path.exists(bg_path):
                            if st.button(bg_key, key=f"preset_{bg_key}", use_container_width=True):
                                st.session_state.extraction_mode = bg_key
                                st.rerun()

                if st.session_state.extraction_mode == "Custom Color":
                    st.markdown("---")
                    st.session_state.custom_color = st.color_picker("Select Color", st.session_state.custom_color)

                if st.session_state.extraction_mode == "Custom Image":
                    st.markdown("---")
                    custom_bg = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], key="custom_bg_upload")
                    if custom_bg:
                        st.session_state.selected_bg = Image.open(custom_bg)

            with bg_col2:
                if st.session_state.current_image is not None and st.session_state.mask is not None:
                    st.markdown("#### Live Preview")
                    bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                    result_pil = apply_background(st.session_state.current_image, st.session_state.mask, 
                                                 st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                    
                    display_img = result_pil.copy()
                    display_img.thumbnail((800, 600), Image.LANCZOS)
                    st.image(display_img, use_container_width=True)
                    
                    st.markdown("---")
                    get_download_button(result_pil, export_format, quality, "⬇️ Download",  
                                      f"background_preview.{export_format.lower()}", "download_bg")

        # TAB 2: Filters & Effects
        with tabs[1]:
            st.markdown("#### Filters & Adjustments")
            
            filter_col1, filter_col2 = st.columns([1, 2])

            with filter_col1:
                st.markdown("**Filter Presets**")
                st.session_state.filter_type = st.selectbox("Choose Filter", list(FILTERS.keys()))
                
                st.markdown("---")
                st.markdown("**Fine-tune Adjustments**")
                st.session_state.brightness = st.slider("☀️ Brightness", 0.0, 2.0, st.session_state.brightness, 0.1)
                st.session_state.contrast = st.slider("🔆 Contrast", 0.0, 2.0, st.session_state.contrast, 0.1)
                st.session_state.saturation = st.slider("🌈 Saturation", 0.0, 2.0, st.session_state.saturation, 0.1)

            with filter_col2:
                if st.session_state.current_image is not None:
                    st.markdown("#### Live Preview")
                    bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                    result_pil = apply_background(st.session_state.current_image, st.session_state.mask,
                                                 st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                    result_pil = apply_filters_and_adjustments(result_pil)
                    
                    display_img = result_pil.copy()
                    display_img.thumbnail((800, 600), Image.LANCZOS)
                    st.image(display_img, use_container_width=True)
                    
                    st.markdown("---")
                    get_download_button(result_pil, export_format, quality, "⬇️ Download",
                                      f"filtered_image.{export_format.lower()}", "download_filter")

        # TAB 3: Crop & Resize
        with tabs[2]:
            st.markdown("#### Crop & Resize Tools")
            
            crop_col1, crop_col2 = st.columns([1, 2])

            with crop_col1:
                st.markdown("**Crop Presets**")
                st.session_state.crop_preset = st.selectbox("Aspect Ratio", list(CROP_PRESETS.keys()))
                
                st.markdown("---")
                st.markdown("**Resize Options**")
                
                if st.session_state.current_image is not None:
                    orig_h, orig_w = st.session_state.current_image.shape[:2]
                    st.caption(f"Original: {orig_w} × {orig_h} px")
                    
                    st.session_state.resize_percent = st.slider("Scale (%)", 10, 200, st.session_state.resize_percent)
                    new_w = int(orig_w * st.session_state.resize_percent / 100)
                    new_h = int(orig_h * st.session_state.resize_percent / 100)
                    st.caption(f"New: {new_w} × {new_h} px")

            with crop_col2:
                if st.session_state.current_image is not None:
                    st.markdown("#### Live Preview")
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
                    
                    st.markdown("---")
                    get_download_button(result_pil, export_format, quality, "⬇️ Download",
                                      f"cropped_image.{export_format.lower()}", "download_crop")

        # TAB 4: Final Preview
        with tabs[3]:
            st.markdown("#### Final Preview & Export")

            if st.session_state.current_image is not None and st.session_state.mask is not None:
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

                st.markdown("**Comparison View**")
                comparison_mode = st.radio(
                    "Select Mode",
                    ["Interactive Slider", "Side-by-Side", "Blend View", "Grid View"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                if comparison_mode == "Interactive Slider":
                    original_img = Image.fromarray(st.session_state.original_image)
                    
                    if result_pil.mode == 'RGBA':
                        result_rgb = Image.new('RGB', result_pil.size, (255, 255, 255))
                        result_rgb.paste(result_pil, (0, 0), result_pil)
                    else:
                        result_rgb = result_pil.convert('RGB')
                    
                    if original_img.size != result_rgb.size:
                        result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                    
                    zoom_value = st.slider("🔍 Zoom Level (%)", 50, 200, st.session_state.zoom_percentage, 5, key="zoom_slider")
                    st.session_state.zoom_percentage = zoom_value
                    
                    scale = zoom_value / 100.0
                    new_w = max(1, int(original_img.size[0] * scale))
                    new_h = max(1, int(original_img.size[1] * scale))
                    
                    zoomed_orig = original_img.resize((new_w, new_h), Image.LANCZOS)
                    zoomed_result = result_rgb.resize((new_w, new_h), Image.LANCZOS)
                    
                    image_comparison(
                        img1=zoomed_orig, 
                        img2=zoomed_result, 
                        label1="Original",
                        label2="Processed"
                    )

                elif comparison_mode == "Side-by-Side":
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("**📷 Original**")
                        original_img = Image.fromarray(st.session_state.original_image)
                        display_orig = original_img.copy()
                        display_orig.thumbnail((600, 600), Image.LANCZOS)
                        st.image(display_orig, use_container_width=True)
                        
                        orig_w, orig_h = original_img.size
                        st.caption(f"Size: {orig_w} × {orig_h} px | Mode: {original_img.mode}")
                    
                    with comp_col2:
                        st.markdown("**✨ Processed**")
                        display_result = result_pil.copy()
                        display_result.thumbnail((600, 600), Image.LANCZOS)
                        st.image(display_result, use_container_width=True)
                        
                        res_w, res_h = result_pil.size
                        st.caption(f"Size: {res_w} × {res_h} px | Mode: {result_pil.mode}")

                elif comparison_mode == "Blend View":
                    original_img = Image.fromarray(st.session_state.original_image)
                    
                    if result_pil.mode == 'RGBA':
                        result_rgb = Image.new('RGB', result_pil.size, (255, 255, 255))
                        result_rgb.paste(result_pil, (0, 0), result_pil)
                    else:
                        result_rgb = result_pil.convert('RGB')
                    
                    if original_img.size != result_rgb.size:
                        result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                    
                    blend_value = st.slider("Blend Ratio", 0.0, 1.0, 0.5, 0.01, key="blend_amount")
                    
                    orig_rgb = original_img.convert('RGB')
                    blended = Image.blend(orig_rgb, result_rgb, float(blend_value))
                    
                    display_blended = blended.copy()
                    display_blended.thumbnail((900, 700), Image.LANCZOS)
                    st.image(display_blended, use_container_width=True, 
                            caption=f"Blend: {int((1-blend_value)*100)}% Original + {int(blend_value*100)}% Processed")

                elif comparison_mode == "Grid View":
                    original_img = Image.fromarray(st.session_state.original_image)
                    
                    if result_pil.mode == 'RGBA':
                        result_rgb = Image.new('RGB', result_pil.size, (255, 255, 255))
                        result_rgb.paste(result_pil, (0, 0), result_pil)
                    else:
                        result_rgb = result_pil.convert('RGB')
                    
                    if original_img.size != result_rgb.size:
                        result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                    
                    orig_arr = np.array(original_img.convert('RGB'))
                    result_arr = np.array(result_rgb)
                    diff_arr = np.abs(orig_arr.astype(float) - result_arr.astype(float)).astype(np.uint8)
                    diff_img = Image.fromarray(diff_arr)
                    
                    blend_50 = Image.blend(original_img.convert('RGB'), result_rgb, 0.5)
                    
                    grid_col1, grid_col2 = st.columns(2)
                    
                    with grid_col1:
                        st.markdown("**📷 Original**")
                        display_orig = original_img.copy()
                        display_orig.thumbnail((450, 450), Image.LANCZOS)
                        st.image(display_orig, use_container_width=True)
                        
                        st.markdown("**🔍 Difference Map**")
                        display_diff = diff_img.copy()
                        display_diff.thumbnail((450, 450), Image.LANCZOS)
                        st.image(display_diff, use_container_width=True)
                    
                    with grid_col2:
                        st.markdown("**✨ Processed**")
                        display_result = result_rgb.copy()
                        display_result.thumbnail((450, 450), Image.LANCZOS)
                        st.image(display_result, use_container_width=True)
                        
                        st.markdown("**🎨 50% Blend**")
                        display_blend = blend_50.copy()
                        display_blend.thumbnail((450, 450), Image.LANCZOS)
                        st.image(display_blend, use_container_width=True)

                st.markdown("---")
                st.markdown("**📥 Export Options**")
                
                export_col1, export_col2, export_col3 = st.columns(3)

                with export_col1:
                    get_download_button(result_pil, export_format, quality, "⬇️ Final Image",
                                      f"oneview_final.{export_format.lower()}", "download_final")

                with export_col2:
                    buf_orig = BytesIO()
                    Image.fromarray(st.session_state.original_image).save(buf_orig, format="PNG")
                    st.download_button("📥 Original", buf_orig.getvalue(), "original.png",
                                     "image/png", key="download_orig", use_container_width=True)

                with export_col3:
                    original_img = Image.fromarray(st.session_state.original_image)
                    result_rgb = result_pil.convert("RGB")
                    if original_img.size != result_rgb.size:
                        result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                    
                    comparison = np.concatenate([np.array(original_img), np.array(result_rgb)], axis=1)
                    buf_comp = BytesIO()
                    Image.fromarray(comparison).save(buf_comp, format="PNG")
                    st.download_button("📊 Comparison", buf_comp.getvalue(), "comparison.png",
                                     "image/png", key="download_comp", use_container_width=True)

                st.session_state.current_step = 4

        st.markdown('</div>', unsafe_allow_html=True)

    # Batch Processing Section
    if st.session_state.batch_mode and len(st.session_state.uploaded_images) > 0:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>🔄 Batch Processing</h3>", unsafe_allow_html=True)
        st.info(f"Ready to process {len(st.session_state.uploaded_images)} images with current settings")

        if st.button("▶️ Process All Images", use_container_width=True):
            progress_bar = st.progress(0)
            batch_results = []

            for idx, img_array in enumerate(st.session_state.uploaded_images):
                with st.spinner(f"Processing image {idx + 1}/{len(st.session_state.uploaded_images)}..."):
                    prob = predict_mask(model, img_array, CONFIG["device"], CONFIG["img_size"])
                    mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                    mask_bin = (mask > 127).astype(np.uint8)

                    bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                    result_pil = apply_background(img_array, mask_bin, st.session_state.extraction_mode, 
                                                 bg_path, st.session_state.custom_color)
                    result_pil = apply_filters_and_adjustments(result_pil)
                    result_pil = crop_image(result_pil, st.session_state.crop_preset)

                    if st.session_state.resize_percent != 100:
                        orig_w, orig_h = result_pil.size
                        new_w = int(orig_w * st.session_state.resize_percent / 100)
                        new_h = int(orig_h * st.session_state.resize_percent / 100)
                        result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)

                    batch_results.append(result_pil)
                    progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))

            st.success("✅ All images processed successfully!")

            st.markdown("**📸 Results Gallery**")
            cols = st.columns(min(3, len(batch_results)))
            for idx, result in enumerate(batch_results):
                with cols[idx % len(cols)]:
                    display_result = result.copy()
                    display_result.thumbnail((350, 300), Image.LANCZOS)
                    st.image(display_result, caption=f"Image {idx + 1}", use_container_width=True)

            st.markdown("---")
            st.markdown("**📥 Download All Results**")
            download_cols = st.columns(min(len(batch_results), 4))
            for idx, result in enumerate(batch_results):
                with download_cols[idx % len(download_cols)]:
                    buf = BytesIO()
                    if export_format == "PNG" and result.mode == "RGBA":
                        result.save(buf, format="PNG")
                    elif export_format in ["JPEG", "JPG"]:
                        result.convert("RGB").save(buf, format="JPEG", quality=quality)
                    else:
                        result.convert("RGB").save(buf, format=export_format, quality=quality)

                    st.download_button(f"⬇️ Image {idx + 1}", buf.getvalue(), 
                                     f"batch_{idx + 1}.{export_format.lower()}",
                                     f"image/{export_format.lower()}", 
                                     key=f"batch_dl_{idx}", 
                                     use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Elegant Footer
    st.markdown("""
    <div class="elegant-footer">
        <h2>OneView</h2>
        <p>Professional Image Editing Suite</p>
        <div class="credits">
            Crafted with precision by <strong>Manusha</strong><br>
            Powered by Advanced AI Technology
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
