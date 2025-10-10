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
# PROFESSIONAL CSS - PURPLE/INDIGO GRADIENT THEME
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e1b4b 100%);
    color: #e2e8f0;
}

/* Main Title Styling */
h1 {
    background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    text-align: center;
    font-size: 3.5rem !important;
    letter-spacing: 2px;
    margin-bottom: 0.5rem !important;
}

h2 {
    color: #c4b5fd !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    margin-top: 2rem !important;
}

h3 {
    color: #ddd6fe !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}

/* Button Styling - Professional Purple Gradient */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #c026d3 100%) !important;
    box-shadow: 0 6px 25px rgba(139, 92, 246, 0.6) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4) !important;
}

/* Download Button Styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 6px 25px rgba(16, 185, 129, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Slider Styling */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
}

.stSlider > div > div > div > div {
    background-color: #a78bfa !important;
    border: 3px solid #1e293b !important;
    box-shadow: 0 0 15px rgba(167, 139, 250, 0.5) !important;
}

/* Input Fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 0.75rem !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}

/* Select Box Styling */
.stSelectbox > div > div {
    background-color: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* File Uploader */
.stFileUploader {
    background: rgba(51, 65, 85, 0.4) !important;
    border: 2px dashed rgba(139, 92, 246, 0.4) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}

.stFileUploader:hover {
    border-color: rgba(139, 92, 246, 0.7) !important;
    background: rgba(51, 65, 85, 0.6) !important;
}

/* Cards and Containers */
.element-container, .stMarkdown {
    color: #cbd5e1 !important;
}

div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(139, 92, 246, 0.2);
    margin-bottom: 1rem;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(30, 41, 59, 0.4);
    border-radius: 12px;
    padding: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #94a3b8;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
}

/* Success/Warning Boxes */
.success-box {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #86efac;
    padding: 1rem;
    border-radius: 12px;
    margin: 1rem 0;
}

.warning-box {
    background: rgba(251, 146, 60, 0.15);
    border: 1px solid rgba(251, 146, 60, 0.4);
    color: #fdba74;
    padding: 1rem;
    border-radius: 12px;
    margin: 1rem 0;
}

/* Info Box */
.info-box {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #c4b5fd;
    padding: 1rem;
    border-radius: 12px;
    margin: 1rem 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: #1e293b;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 10px;
    border: 2px solid #1e293b;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #7c3aed 0%, #a78bfa 100%);
}

/* Preview Cards */
.preview-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(139, 92, 246, 0.3);
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.preview-header {
    color: #c4b5fd;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 1rem;
    font-family: 'Space Grotesk', sans-serif;
}

/* Step Indicators */
.step-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 2rem 0;
    padding: 2rem;
    background: rgba(30, 41, 59, 0.4);
    border-radius: 16px;
    border: 1px solid rgba(139, 92, 246, 0.2);
}

.step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

.step-number {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}

.step-number.inactive {
    background: rgba(51, 65, 85, 0.6);
    box-shadow: none;
}

.step-label {
    color: #c4b5fd;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.9rem;
    letter-spacing: 1px;
}

/* General text colors */
p, span, div, label {
    color: #cbd5e1;
}

h4, h5, h6 {
    color: #f1f5f9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stCaption {
    color: #94a3b8 !important;
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
    background: rgba(51, 65, 85, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    color: #e2e8f0 !important;
}

/* Modern Footer */
.modern-footer {
    text-align: center;
    padding: 2rem;
    background: rgba(30, 41, 59, 0.4);
    border-radius: 16px;
    border: 1px solid rgba(139, 92, 246, 0.2);
    margin-top: 3rem;
}

.modern-footer h3 {
    background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.modern-footer p {
    color: #94a3b8;
    font-size: 1rem;
}

/* Hero Section */
.hero-section {
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(30, 41, 59, 0.3);
    border-radius: 20px;
    border: 1px solid rgba(139, 92, 246, 0.3);
    margin: 2rem 0;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.2rem;
    margin-top: 1rem;
    line-height: 1.6;
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
        'custom_color': '#6366f1',
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
        blurred = cv2.GaussianBlur(orig_np, (51, 51), 0)
        result = np.where(mask_bin[..., None] == 1, orig_np, blurred)
        return Image.fromarray(result)

    elif mode in ["Black", "White"]:
        bg_val = 0 if mode == "Black" else 255
        bg = np.full((h, w, 3), bg_val, dtype=np.uint8)
        result = np.where(mask_bin[..., None] == 1, orig_np, bg)
        return Image.fromarray(result)

    elif mode == "Custom Color" and custom_color:
        hex_color = custom_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bg = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
        result = np.where(mask_bin[..., None] == 1, orig_np, bg)
        return Image.fromarray(result)

    elif mode.startswith("Background") and bg_path:
        if Path(bg_path).exists():
            bg_img = cv2.imread(bg_path)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (w, h))
            result = np.where(mask_bin[..., None] == 1, orig_np, bg_img)
            return Image.fromarray(result)

    return Image.fromarray(orig_np)

def apply_filters_and_adjustments(pil_img):
    img = FILTERS[st.session_state.filter_type](pil_img)
    img = ImageEnhance.Brightness(img).enhance(st.session_state.brightness)
    img = ImageEnhance.Contrast(img).enhance(st.session_state.contrast)
    img = ImageEnhance.Color(img).enhance(st.session_state.saturation)
    return img

def crop_image(pil_img, preset):
    if preset == "Freeform" or CROP_PRESETS[preset] is None:
        return pil_img
    
    w, h = pil_img.size
    target_w, target_h = CROP_PRESETS[preset]
    
    current_ratio = w / h
    target_ratio = target_w / target_h
    
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return pil_img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return pil_img.crop((0, top, w, top + new_h))

def get_download_button(pil_img, fmt, quality, label, filename, key):
    buf = BytesIO()
    if fmt == "PNG":
        pil_img.save(buf, format="PNG")
    elif fmt == "JPEG":
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        pil_img.save(buf, format="JPEG", quality=quality)
    elif fmt == "WEBP":
        pil_img.save(buf, format="WEBP", quality=quality)
    
    st.download_button(label, buf.getvalue(), filename, 
                      f"image/{fmt.lower()}", key=key, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="OneView - Professional Image Processing",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()

    # Hero Header
    st.markdown("""
    <div class="hero-section">
        <h1>ONEVIEW</h1>
        <p class="hero-subtitle">Create pro-level cutouts by removing distractions and letting your subject shine in every image.</p>
    </div>
    """, unsafe_allow_html=True)

    # Step Indicators
    st.markdown("""
    <div class="step-container">
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-label">UPLOAD</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-label">EXTRACT</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-label">CUSTOMIZE</div>
        </div>
        <div class="step">
            <div class="step-number">4</div>
            <div class="step-label">EXPORT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - Settings
    with st.sidebar:
        st.markdown("## ⚙️ Detection Settings")
        
        st.session_state.fg_thresh = st.slider(
            "Detection Threshold",
            0.0, 1.0, st.session_state.fg_thresh, 0.05,
            help="Lower = more inclusive, Higher = more precise"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.min_area = st.number_input(
                "Minimum Area (px)",
                50, 5000, st.session_state.min_area, 50
            )
        with col2:
            if st.button("−", key="dec_area"):
                st.session_state.min_area = max(50, st.session_state.min_area - 50)
                st.rerun()
            if st.button("+", key="inc_area"):
                st.session_state.min_area = min(5000, st.session_state.min_area + 50)
                st.rerun()

        st.markdown("---")
        st.markdown("## 📦 Export Settings")
        
        export_format = st.selectbox("Format", ["PNG", "JPEG", "WEBP"])
        quality = st.slider("Quality", 50, 100, 95) if export_format != "PNG" else 100

    # Main Content Tabs
    tabs = st.tabs(["📤 Upload", "✂️ Extract", "🎨 Customize", "🖼️ Preview", "💾 Export"])

    # Upload Tab
    with tabs[0]:
        st.markdown("### Upload Your Image")
        
        uploaded_files = st.file_uploader(
            "Drop your images here or click to browse",
            type=['png', 'jpg', 'jpeg', 'webp'],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
            st.session_state.uploaded_images = []
            for uf in uploaded_files:
                img = Image.open(uf).convert('RGB')
                st.session_state.uploaded_images.append(np.array(img))
            
            st.session_state.original_image = st.session_state.uploaded_images[0]
            st.session_state.current_image = st.session_state.original_image.copy()
            
            st.markdown('<div class="success-box">✅ Image uploaded successfully!</div>', 
                       unsafe_allow_html=True)
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown('<div class="preview-header">📸 Uploaded Image</div>', unsafe_allow_html=True)
            display_img = Image.fromarray(st.session_state.original_image)
            display_img.thumbnail((800, 600), Image.LANCZOS)
            st.image(display_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Extract Tab
    with tabs[1]:
        if st.session_state.original_image is not None:
            st.markdown("### AI-Powered Background Removal")
            
            if st.button("🚀 Extract Subject", type="primary", use_container_width=True):
                with st.spinner("🔮 AI is analyzing your image..."):
                    model = get_model()
                    prob_map = predict_mask(
                        model,
                        st.session_state.current_image,
                        CONFIG["device"],
                        CONFIG["img_size"]
                    )
                    st.session_state.prob_map = prob_map
                    
                    mask = postprocess_mask(
                        prob_map,
                        st.session_state.fg_thresh,
                        st.session_state.min_area
                    )
                    st.session_state.mask = (mask > 0).astype(np.uint8)
                    
                    st.markdown('<div class="success-box">✨ Extraction complete!</div>', 
                               unsafe_allow_html=True)
            
            if st.session_state.mask is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                    st.markdown('<div class="preview-header">Original</div>', unsafe_allow_html=True)
                    display_orig = Image.fromarray(st.session_state.original_image)
                    display_orig.thumbnail((400, 400), Image.LANCZOS)
                    st.image(display_orig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                    st.markdown('<div class="preview-header">Extracted</div>', unsafe_allow_html=True)
                    result = apply_background(
                        st.session_state.current_image,
                        st.session_state.mask,
                        "Transparent"
                    )
                    display_result = result.copy()
                    display_result.thumbnail((400, 400), Image.LANCZOS)
                    st.image(display_result, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ Please upload an image first</div>', 
                       unsafe_allow_html=True)

    # Customize Tab
    with tabs[2]:
        if st.session_state.mask is not None:
            st.markdown("### Customize Your Image")
            
            st.markdown("#### 🎨 Background Options")
            st.session_state.extraction_mode = st.radio(
                "Background Style",
                ["Black", "White", "Transparent", "Blur", "Custom Color"] + list(BG_IMAGE_PATHS.keys()),
                horizontal=True
            )
            
            if st.session_state.extraction_mode == "Custom Color":
                st.session_state.custom_color = st.color_picker(
                    "Pick a color", st.session_state.custom_color
                )
            
            st.markdown("---")
            st.markdown("#### ✨ Filters & Adjustments")
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.filter_type = st.selectbox("Filter", list(FILTERS.keys()))
                st.session_state.brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            
            with col2:
                st.session_state.contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                st.session_state.saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
            
            st.markdown("---")
            st.markdown("#### 📐 Crop & Resize")
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.crop_preset = st.selectbox("Crop Preset", list(CROP_PRESETS.keys()))
            with col2:
                st.session_state.resize_percent = st.slider("Resize %", 10, 200, 100, 10)
        else:
            st.markdown('<div class="info-box">ℹ️ Please extract the subject first</div>', 
                       unsafe_allow_html=True)

    # Preview Tab
    with tabs[3]:
        if st.session_state.mask is not None:
            st.markdown("### Live Preview")
            
            bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
            result = apply_background(
                st.session_state.current_image,
                st.session_state.mask,
                st.session_state.extraction_mode,
                bg_path,
                st.session_state.custom_color
            )
            
            result = apply_filters_and_adjustments(result)
            result = crop_image(result, st.session_state.crop_preset)
            
            if st.session_state.resize_percent != 100:
                orig_w, orig_h = result.size
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                result = result.resize((new_w, new_h), Image.LANCZOS)
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown('<div class="preview-header">🎬 Final Preview</div>', unsafe_allow_html=True)
            display_preview = result.copy()
            display_preview.thumbnail((900, 700), Image.LANCZOS)
            st.image(display_preview, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### 🔍 Compare with Original")
            blend_value = st.slider("Blend Amount", 0.0, 1.0, 0.5, 0.1)
            
            if result.mode != 'RGB':
                result_rgb = result.convert('RGB')
            else:
                result_rgb = result
            
            original_img = Image.fromarray(st.session_state.original_image)
            if original_img.size != result_rgb.size:
                result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
            
            orig_rgb = original_img.convert('RGB')
            blended = Image.blend(orig_rgb, result_rgb, float(blend_value))
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            display_blended = blended.copy()
            display_blended.thumbnail((800, 600), Image.LANCZOS)
            st.image(display_blended, use_container_width=True, caption=f"Blend: {int(blend_value*100)}%")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ Please extract the subject first</div>', 
                       unsafe_allow_html=True)

    # Export Tab
    with tabs[4]:
        if st.session_state.mask is not None:
            st.markdown("### Export Your Work")
            
            bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
            final_result = apply_background(
                st.session_state.current_image,
                st.session_state.mask,
                st.session_state.extraction_mode,
                bg_path,
                st.session_state.custom_color
            )
            final_result = apply_filters_and_adjustments(final_result)
            final_result = crop_image(final_result, st.session_state.crop_preset)
            
            if st.session_state.resize_percent != 100:
                orig_w, orig_h = final_result.size
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                final_result = final_result.resize((new_w, new_h), Image.LANCZOS)
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown('<div class="preview-header">🎉 Final Result</div>', unsafe_allow_html=True)
            display_final = final_result.copy()
            display_final.thumbnail((900, 700), Image.LANCZOS)
            st.image(display_final, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📥 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                get_download_button(
                    final_result, export_format, quality,
                    "⬇️ Download Final",
                    f"oneview_final.{export_format.lower()}",
                    "download_final"
                )
            
            with col2:
                buf_orig = BytesIO()
                Image.fromarray(st.session_state.original_image).save(buf_orig, format="PNG")
                st.download_button(
                    "📥 Download Original",
                    buf_orig.getvalue(),
                    "original.png",
                    "image/png",
                    key="download_orig",
                    use_container_width=True
                )
            
            with col3:
                original_img = Image.fromarray(st.session_state.original_image)
                result_rgb = final_result.convert("RGB")
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                comparison = np.concatenate([np.array(original_img), np.array(result_rgb)], axis=1)
                buf_comp = BytesIO()
                Image.fromarray(comparison).save(buf_comp, format="PNG")
                st.download_button(
                    "📊 Download Comparison",
                    buf_comp.getvalue(),
                    "comparison.png",
                    "image/png",
                    key="download_comp",
                    use_container_width=True
                )
        else:
            st.markdown('<div class="info-box">ℹ️ Please extract the subject first</div>', 
                       unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="modern-footer">
        <h3>OneView</h3>
        <p>Professional AI-Powered Image Processing Solution</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.7;">Developed by Manusha</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
