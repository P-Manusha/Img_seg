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

# Paths
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

# Crop Presets
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

# Filter Presets
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
# CUSTOM CSS - PROFESSIONAL CLASSIC DESIGN
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=Montserrat:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Montserrat', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
}

.main-header {
    background: linear-gradient(135deg, rgba(15, 52, 96, 0.95), rgba(26, 26, 46, 0.95));
    padding: 3rem 2.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(236, 179, 101, 0.2);
    margin-bottom: 2rem;
    border: 2px solid rgba(236, 179, 101, 0.3);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at top right, rgba(236, 179, 101, 0.1), transparent 50%);
    pointer-events: none;
}

.main-header h1 {
    margin: 0;
    background: linear-gradient(135deg, #ecb365 0%, #ffd89b 50%, #ecb365 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 800;
    font-family: 'Playfair Display', serif;
    letter-spacing: 2px;
    text-transform: uppercase;
    position: relative;
    text-shadow: 0 0 30px rgba(236, 179, 101, 0.3);
}

.main-header p {
    margin: 1rem 0 0 0;
    color: #e8dcc4;
    font-size: 1.15rem;
    font-weight: 400;
    letter-spacing: 1px;
    position: relative;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: linear-gradient(135deg, rgba(15, 52, 96, 0.9), rgba(26, 26, 46, 0.9));
    padding: 1.2rem;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(236, 179, 101, 0.2);
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(236, 179, 101, 0.1), rgba(236, 179, 101, 0.05));
    border: 1px solid rgba(236, 179, 101, 0.3);
    color: #ecb365 !important;
    padding: 0.8rem 1.8rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, rgba(236, 179, 101, 0.2), rgba(236, 179, 101, 0.1));
    border-color: #ecb365;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(236, 179, 101, 0.3);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #ecb365, #d4a574) !important;
    color: #1a1a2e !important;
    border-color: #ecb365 !important;
    box-shadow: 0 8px 20px rgba(236, 179, 101, 0.4);
}

button[kind="primary"], button[kind="secondary"], .stButton button {
    background: linear-gradient(135deg, #ecb365 0%, #d4a574 100%) !important;
    color: #1a1a2e !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 6px 20px rgba(236, 179, 101, 0.3) !important;
    text-transform: uppercase;
}

button[kind="primary"]:hover, button[kind="secondary"]:hover, .stButton button:hover {
    background: linear-gradient(135deg, #ffd89b 0%, #ecb365 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(236, 179, 101, 0.5) !important;
}

.stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
    background: rgba(15, 52, 96, 0.6) !important;
    border-radius: 12px !important;
}

.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: rgba(26, 26, 46, 0.8) !important;
    border: 1px solid rgba(236, 179, 101, 0.3) !important;
    color: #e8dcc4 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div:hover, .stTextInput > div > div > input:hover, .stNumberInput > div > div > input:hover {
    border-color: #ecb365 !important;
    box-shadow: 0 0 15px rgba(236, 179, 101, 0.2) !important;
}

label, .stMarkdown p, .stMarkdown li {
    color: #e8dcc4 !important;
    font-weight: 500 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ecb365 !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

.stSlider {
    padding: 1rem 0;
}

.stSlider > div > div > div {
    background: linear-gradient(90deg, #ecb365, #d4a574) !important;
}

.stSlider > div > div > div > div {
    background: #ecb365 !important;
    border: 3px solid #1a1a2e !important;
    box-shadow: 0 4px 12px rgba(236, 179, 101, 0.4) !important;
}

.uploadedFile {
    background: rgba(15, 52, 96, 0.6) !important;
    border: 2px solid rgba(236, 179, 101, 0.3) !important;
    border-radius: 12px !important;
    color: #e8dcc4 !important;
}

.stAlert {
    background: rgba(15, 52, 96, 0.8) !important;
    border-left: 4px solid #ecb365 !important;
    border-radius: 10px !important;
    color: #e8dcc4 !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #ecb365, #ffd89b) !important;
}

.stExpander {
    background: rgba(15, 52, 96, 0.6) !important;
    border: 1px solid rgba(236, 179, 101, 0.3) !important;
    border-radius: 12px !important;
}

.stExpander summary {
    color: #ecb365 !important;
    font-weight: 600 !important;
}

.stCheckbox label {
    color: #e8dcc4 !important;
}

.stCheckbox input:checked ~ span {
    background: #ecb365 !important;
    border-color: #ecb365 !important;
}

.stRadio label {
    color: #e8dcc4 !important;
}

.stRadio input:checked ~ div {
    background: #ecb365 !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16213e 0%, #0f3460 100%) !important;
    border-right: 2px solid rgba(236, 179, 101, 0.2) !important;
}

div[data-testid="stSidebar"] h1, 
div[data-testid="stSidebar"] h2, 
div[data-testid="stSidebar"] h3 {
    color: #ecb365 !important;
}

div[data-testid="stSidebar"] label, 
div[data-testid="stSidebar"] p {
    color: #e8dcc4 !important;
}

.footer-professional {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, rgba(15, 52, 96, 0.9), rgba(26, 26, 46, 0.9));
    border-radius: 20px;
    margin-top: 3rem;
    border: 2px solid rgba(236, 179, 101, 0.3);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.footer-professional h3 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    background: linear-gradient(135deg, #ecb365 0%, #ffd89b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
}

.footer-professional p {
    color: #e8dcc4;
    font-size: 1rem;
    margin: 0.5rem 0;
}

.stDownloadButton button {
    background: linear-gradient(135deg, #ecb365 0%, #d4a574 100%) !important;
    color: #1a1a2e !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 20px rgba(236, 179, 101, 0.3) !important;
}

.stDownloadButton button:hover {
    background: linear-gradient(135deg, #ffd89b 0%, #ecb365 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(236, 179, 101, 0.5) !important;
}

hr {
    border-color: rgba(236, 179, 101, 0.3) !important;
    margin: 2rem 0 !important;
}

.stSpinner > div {
    border-top-color: #ecb365 !important;
}

::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: rgba(26, 26, 46, 0.5);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #ecb365, #d4a574);
    border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #ffd89b, #ecb365);
}

.demo-section {
    background: linear-gradient(135deg, rgba(15, 52, 96, 0.6), rgba(26, 26, 46, 0.6));
    padding: 2rem;
    border-radius: 20px;
    border: 2px solid rgba(236, 179, 101, 0.3);
    margin: 2rem 0;
}

.step-indicator {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, rgba(15, 52, 96, 0.9), rgba(26, 26, 46, 0.9));
    padding: 1.5rem;
    border-radius: 15px;
    margin: 2rem 0;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    border: 2px solid rgba(236, 179, 101, 0.3);
}

.step {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
}

.step-number {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: linear-gradient(135deg, #ecb365 0%, #d4a574 100%);
    color: #1a1a2e;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}

.step-number.active {
    box-shadow: 0 0 30px rgba(236, 179, 101, 0.8);
    transform: scale(1.15);
}

.step-label {
    color: #e8dcc4;
    font-weight: 600;
    font-size: 0.9rem;
}

.step-connector {
    height: 3px;
    background: linear-gradient(90deg, #ecb365 0%, #d4a574 100%);
    flex: 1;
    margin: 0 1rem;
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
        'extraction_mode': "Transparent",
        'selected_bg': None,
        'current_step': 1,
        'crop_preset': "Freeform",
        'filter_type': "None",
        'brightness': 1.0,
        'contrast': 1.0,
        'saturation': 1.0,
        'sharpness': 1.0,
        'batch_mode': False,
        'custom_color': (255, 255, 255),
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
# MODEL FUNCTIONS - FIXED
# ============================================================================

@st.cache_resource
def get_model(path=CONFIG["model_path"]):
    if not Path(path).exists():
        st.error(f"❌ Model file not found: {path}")
        st.info("Please ensure 'best_seg_model.pth' is in the root folder")
        st.stop()

    model = segmentation_models.deeplabv3_resnet50(weights=None, num_classes=CONFIG["num_classes"])
    
    try:
        checkpoint = torch.load(path, map_location=torch.device(CONFIG["device"]))

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # If checkpoint has keys like 'epoch', 'optimizer_state', etc., it's a training checkpoint
                # Try to extract just the model weights
                state_dict = {k: v for k, v in checkpoint.items() 
                            if not k in ['epoch', 'optimizer_state', 'mean_iou']}
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Could not load custom model: {str(e)}")
        st.info("Loading pretrained model instead...")
        model = segmentation_models.deeplabv3_resnet50(weights='DEFAULT')

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
        result = np.full_like(orig_np, custom_color)
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
    img = ImageEnhance.Sharpness(img).enhance(st.session_state.sharpness)
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
                'fg_thresh': float(st.session_state.get('fg_thresh', 0.3)),
                'min_area': int(st.session_state.get('min_area', 300)),
                'extraction_mode': str(st.session_state.get('extraction_mode', 'Transparent')),
                'filter_type': str(st.session_state.get('filter_type', 'None')),
                'brightness': float(st.session_state.get('brightness', 1.0)),
                'contrast': float(st.session_state.get('contrast', 1.0)),
                'saturation': float(
