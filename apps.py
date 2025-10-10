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
# MODERN PROFESSIONAL CSS - ENHANCED BUTTON STYLING
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #f1f5f9;
}

/* ============================================
   ENHANCED BUTTON STYLING - PRIMARY FOCUS
   ============================================ */

/* Primary Buttons (Download, Process, etc.) */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4), 
                0 4px 10px rgba(139, 92, 246, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
    transition: left 0.5s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c026d3 100%) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 12px 28px rgba(99, 102, 241, 0.5), 
                0 6px 14px rgba(139, 92, 246, 0.4),
                0 0 30px rgba(168, 85, 247, 0.3) !important;
}

.stButton > button:hover:before {
    left: 100% !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
}

/* Download Buttons - Special Styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4), 
                0 4px 10px rgba(5, 150, 105, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 50%, #065f46 100%) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 12px 28px rgba(16, 185, 129, 0.5), 
                0 6px 14px rgba(5, 150, 105, 0.4),
                0 0 30px rgba(16, 185, 129, 0.3) !important;
}

.stDownloadButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
}

/* File Uploader Button */
.stFileUploader > div > button {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #b45309 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 8px 20px rgba(245, 158, 11, 0.4) !important;
    transition: all 0.3s ease !important;
}

.stFileUploader > div > button:hover {
    background: linear-gradient(135deg, #d97706 0%, #b45309 50%, #92400e 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 28px rgba(245, 158, 11, 0.5) !important;
}

/* ============================================
   REST OF THE STYLING (UNCHANGED)
   ============================================ */

.main-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
}

.main-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    text-shadow: 0 0 40px rgba(99, 102, 241, 0.3);
}

.main-header p {
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-top: 0.5rem;
}

.feature-card {
    background: rgba(51, 65, 85, 0.5);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    transition: all 0.3s ease;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
    background: rgba(51, 65, 85, 0.7);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.preview-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    margin: 1rem 0;
}

.preview-header {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #a855f7;
    font-family: 'Space Grotesk', sans-serif;
}

.modern-footer {
    text-align: center;
    padding: 2rem;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    margin-top: 3rem;
}

.modern-footer h3 {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.metric-card {
    background: rgba(51, 65, 85, 0.5);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-3px);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a855f7;
    font-family: 'Space Grotesk', sans-serif;
}

.metric-label {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}

.info-box {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    color: #93c5fd;
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

p, span, div, label {
    color: #cbd5e1;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

h4, h5, h6 {
    color: #f1f5f9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stCaption {
    color: #94a3b8 !important;
    font-style: italic;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stRadio > div {
    flex-direction: row;
    gap: 1rem;
}

.stRadio [role="radiogroup"] {
    gap: 1rem;
}

.streamlit-expanderHeader {
    background: rgba(51, 65, 85, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.1);
}

.project-card {
    background: rgba(51, 65, 85, 0.5);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.project-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(51, 65, 85, 0.7);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30, 41, 59, 0.5);
    padding: 0.5rem;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(51, 65, 85, 0.5);
    border-radius: 8px;
    color: #cbd5e1;
    font-weight: 500;
    padding: 0.75rem 1.5rem;
    border: 1px solid rgba(148, 163, 184, 0.1);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border-color: transparent;
}

.stSlider {
    padding: 1rem 0;
}

[data-testid="stSlider"] [role="slider"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
}

.stSelectbox > div > div {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 8px;
    color: #cbd5e1;
}

.stTextInput > div > div > input {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 8px;
    color: #cbd5e1;
}

.stNumberInput > div > div > input {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 8px;
    color: #cbd5e1;
}

[data-testid="stColorPicker"] > div > div {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 8px;
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

    elif mode == "Black":
        result = np.zeros((h, w, 3), np.uint8)
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result, 'RGB')

    elif mode == "White":
        result = np.full((h, w, 3), 255, np.uint8)
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result, 'RGB')

    elif mode == "Custom Color":
        hex_color = custom_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        result = np.full((h, w, 3), rgb, np.uint8)
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result, 'RGB')

    elif mode == "Blur":
        bg_blur = cv2.GaussianBlur(orig_np, (51, 51), 0)
        result = bg_blur.copy()
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result, 'RGB')

    elif mode in BG_IMAGE_PATHS and bg_path:
        if Path(bg_path).exists():
            bg_img = cv2.imread(bg_path)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (w, h))
            result = bg_img.copy()
            result[mask_bin == 1] = orig_np[mask_bin == 1]
            return Image.fromarray(result, 'RGB')
        else:
            result = np.zeros((h, w, 3), np.uint8)
            result[mask_bin == 1] = orig_np[mask_bin == 1]
            return Image.fromarray(result, 'RGB')

    else:
        result = np.zeros((h, w, 3), np.uint8)
        result[mask_bin == 1] = orig_np[mask_bin == 1]
        return Image.fromarray(result, 'RGB')

def apply_filters_and_adjustments(img):
    img = FILTERS[st.session_state.filter_type](img)
    img = ImageEnhance.Brightness(img).enhance(st.session_state.brightness)
    img = ImageEnhance.Contrast(img).enhance(st.session_state.contrast)
    img = ImageEnhance.Color(img).enhance(st.session_state.saturation)
    return img

def crop_image(img, preset_name):
    if preset_name == "Freeform" or CROP_PRESETS[preset_name] is None:
        return img

    w, h = img.size
    aspect_w, aspect_h = CROP_PRESETS[preset_name]
    target_aspect = aspect_w / aspect_h
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        new_h = h
    else:
        new_w = w
        new_h = int(w / target_aspect)

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))

def get_download_button(img, fmt, quality, label, filename, key):
    buf = BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() in ["JPEG", "JPG"]:
        save_kwargs["quality"] = quality
        img = img.convert("RGB")
    img.save(buf, **save_kwargs)
    st.download_button(label, buf.getvalue(), filename, f"image/{fmt.lower()}", 
                      key=key, use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="OneView - AI Image Processing", 
                      page_icon="🎨", 
                      layout="wide",
                      initial_sidebar_state="expanded")
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    st.markdown("""
    <div class="main-header">
        <h1>🎨 OneView</h1>
        <p>Professional AI-Powered Image Background Processing</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📤 Upload", "🎯 Extract", "🎨 Customize", "👁️ Preview", "📥 Export"])

    with tabs[0]:
        st.markdown("### Upload Your Images")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader("Choose images", 
                                            type=['png', 'jpg', 'jpeg', 'webp'],
                                            accept_multiple_files=True,
                                            key="file_uploader")
            
            if uploaded_files:
                st.session_state.uploaded_images = uploaded_files
                st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
                
                if len(uploaded_files) > 1:
                    selected_img_name = st.selectbox("Select image to process", 
                                                    [f.name for f in uploaded_files])
                    st.session_state.current_image_idx = [f.name for f in uploaded_files].index(selected_img_name)
                
                current_file = st.session_state.uploaded_images[st.session_state.current_image_idx]
                img = Image.open(current_file).convert('RGB')
                img_np = np.array(img)
                st.session_state.original_image = img_np
                st.session_state.current_image = img_np.copy()
                
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                st.markdown('<div class="preview-header">Preview</div>', unsafe_allow_html=True)
                display_img = img.copy()
                display_img.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Quick Info")
            if st.session_state.original_image is not None:
                h, w = st.session_state.original_image.shape[:2]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{w} × {h}</div>
                    <div class="metric-label">Resolution</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.original_image.shape[2]}</div>
                    <div class="metric-label">Channels</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Tips")
            st.markdown("""
            <div class="info-box">
            • Use high-resolution images for best results<br>
            • Ensure good contrast between subject and background<br>
            • Supported formats: PNG, JPG, JPEG, WEBP
            </div>
            """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### AI Background Extraction")
        
        if st.session_state.original_image is None:
            st.warning("⚠️ Please upload an image first!")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🎛️ Extraction Settings")
                
                fg_thresh = st.slider("Foreground Threshold", 0.0, 1.0, 
                                     st.session_state.fg_thresh, 0.05,
                                     help="Higher values = stricter foreground detection")
                st.session_state.fg_thresh = fg_thresh
                
                min_area = st.slider("Minimum Object Area", 100, 5000, 
                                    st.session_state.min_area, 100,
                                    help="Remove small artifacts")
                st.session_state.min_area = min_area
                
                if st.button("🚀 Extract Background", use_container_width=True):
                    with st.spinner("🔮 AI is processing your image..."):
                        model = get_model()
                        prob_map = predict_mask(model, st.session_state.current_image, 
                                              CONFIG["device"], CONFIG["img_size"])
                        st.session_state.prob_map = prob_map
                        
                        mask = postprocess_mask(prob_map, fg_thresh, min_area)
                        st.session_state.mask = (mask // 255).astype(np.uint8)
                        
                        st.success("✅ Extraction complete!")
                        st.session_state.current_step = 2
            
            with col2:
                st.markdown("#### 🎭 Results")
                if st.session_state.mask is not None:
                    result = apply_background(st.session_state.current_image, 
                                            st.session_state.mask, "Black")
                    
                    st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                    display_result = result.copy()
                    display_result.thumbnail((800, 600), Image.LANCZOS)
                    st.image(display_result, use_container_width=True, caption="Extracted Subject")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Accept", use_container_width=True):
                            st.success("Extraction accepted! Move to Customize tab.")
                    with col_b:
                        if st.button("🔄 Retry", use_container_width=True):
                            st.session_state.mask = None
                            st.rerun()

    with tabs[2]:
        st.markdown("### Customize Your Image")
        
        if st.session_state.mask is None:
            st.warning("⚠️ Please extract the background first!")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🎨 Background Options")
                
                bg_options = ["Black", "White", "Transparent", "Blur", "Custom Color"] + list(BG_IMAGE_PATHS.keys())
                extraction_mode = st.selectbox("Background Type", bg_options, 
                                              index=bg_options.index(st.session_state.extraction_mode))
                st.session_state.extraction_mode = extraction_mode
                
                if extraction_mode == "Custom Color":
                    custom_color = st.color_picker("Pick Background Color", st.session_state.custom_color)
                    st.session_state.custom_color = custom_color
                
                st.markdown("---")
                st.markdown("#### 🎭 Filters & Effects")
                
                filter_type = st.selectbox("Filter", list(FILTERS.keys()), 
                                          index=list(FILTERS.keys()).index(st.session_state.filter_type))
                st.session_state.filter_type = filter_type
                
                st.markdown("#### 🔧 Adjustments")
                brightness = st.slider("Brightness", 0.5, 2.0, st.session_state.brightness, 0.1)
                st.session_state.brightness = brightness
                
                contrast = st.slider("Contrast", 0.5, 2.0, st.session_state.contrast, 0.1)
                st.session_state.contrast = contrast
                
                saturation = st.slider("Saturation", 0.0, 2.0, st.session_state.saturation, 0.1)
                st.session_state.saturation = saturation
                
                st.markdown("#### ✂️ Crop Preset")
                crop_preset = st.selectbox("Crop Ratio", list(CROP_PRESETS.keys()),
                                          index=list(CROP_PRESETS.keys()).index(st.session_state.crop_preset))
                st.session_state.crop_preset = crop_preset
                
                st.markdown("#### 📏 Resize")
                resize_percent = st.slider("Scale (%)", 10, 200, st.session_state.resize_percent, 5)
                st.session_state.resize_percent = resize_percent
            
            with col2:
                st.markdown("#### 👁️ Live Preview")
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result = apply_background(st.session_state.current_image, st.session_state.mask,
                                        st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                result = apply_filters_and_adjustments(result)
                result = crop_image(result, st.session_state.crop_preset)
                
                if st.session_state.resize_percent != 100:
                    orig_w, orig_h = result.size
                    new_w = int(orig_w * st.session_state.resize_percent / 100)
                    new_h = int(orig_h * st.session_state.resize_percent / 100)
                    result = result.resize((new_w, new_h), Image.LANCZOS)
                
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                display_preview = result.copy()
                display_preview.thumbnail((900, 700), Image.LANCZOS)
                st.image(display_preview, use_container_width=True, caption="Customized Result")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("✨ Apply Changes", use_container_width=True):
                    st.success("✅ Changes applied! Check the Preview tab.")
                    st.session_state.current_step = 3

    with tabs[3]:
        st.markdown("### Preview & Compare")
        
        if st.session_state.mask is None:
            st.warning("⚠️ Please extract the background first!")
        else:
            export_format = st.selectbox("Export Format", ["PNG", "JPEG", "WEBP"], index=0)
            quality = 95
            if export_format == "JPEG":
                quality = st.slider("JPEG Quality", 50, 100, 95, 5)
            
            bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
            result = apply_background(st.session_state.current_image, st.session_state.mask,
                                     st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
            result = apply_filters_and_adjustments(result)
            result = crop_image(result, st.session_state.crop_preset)
            
            if st.session_state.resize_percent != 100:
                orig_w, orig_h = result.size
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                result = result.resize((new_w, new_h), Image.LANCZOS)
            
            st.markdown("#### 🔄 Side-by-Side Comparison")
            
            original_img = Image.fromarray(st.session_state.original_image)
            result_rgb = result.convert("RGB")
            
            if original_img.size != result_rgb.size:
                result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                display_orig = original_img.copy()
                display_orig.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_orig, use_container_width=True, caption="Original")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                display_result = result_rgb.copy()
                display_result.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_result, use_container_width=True, caption="Processed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### 🎚️ Interactive Blend")
            
            blend_value = st.slider("Blend Original ↔ Processed", 0.0, 1.0, 0.5, 0.05)
            
            if original_img.mode != 'RGB':
                orig_rgb = original_img.convert('RGB')
            else:
                orig_rgb = original_img
            
            if orig_rgb.size != result_rgb.size:
                result_rgb_resized = result_rgb.resize(orig_rgb.size, Image.LANCZOS)
            else:
                result_rgb_resized = result_rgb
            
            blended = Image.blend(orig_rgb, result_rgb_resized, float(blend_value))
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            display_blended = blended.copy()
            display_blended.thumbnail((800, 600), Image.LANCZOS)
            st.image(display_blended, use_container_width=True, caption=f"Blend: {int(blend_value*100)}%")
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("### Export Your Work")
        
        if st.session_state.mask is None:
            st.warning("⚠️ Please extract the background first!")
        else:
            export_format = st.selectbox("Select Export Format", ["PNG", "JPEG", "WEBP"], 
                                        index=0, key="export_format_final")
            quality = 95
            if export_format == "JPEG":
                quality = st.slider("Image Quality", 50, 100, 95, 5, key="export_quality")
            
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
            st.markdown('<div class="preview-header">🎉 Final Result</div>', unsafe_allow_html=True)
            display_final = final_result.copy()
            display_final.thumbnail((900, 700), Image.LANCZOS)
            st.image(display_final, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📥 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                get_download_button(final_result, export_format, quality, "⬇️ Download Final",
                                  f"oneview_final.{export_format.lower()}", "download_final")
            
            with col2:
                buf_orig = BytesIO()
                Image.fromarray(st.session_state.original_image).save(buf_orig, format="PNG")
                st.download_button("📥 Download Original", buf_orig.getvalue(), "original.png",
                                 "image/png", key="download_orig", use_container_width=True)
            
            with col3:
                original_img = Image.fromarray(st.session_state.original_image)
                result_rgb = final_result.convert("RGB")
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                comparison = np.concatenate([np.array(original_img), np.array(result_rgb)], axis=1)
                buf_comp = BytesIO()
                Image.fromarray(comparison).save(buf_comp, format="PNG")
                st.download_button("📊 Download Comparison", buf_comp.getvalue(), "comparison.png",
                                 "image/png", key="download_comp", use_container_width=True)
            
            st.session_state.current_step = 4

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
