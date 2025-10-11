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

/* Special styling for specific buttons */
button[key="bg_transparent"],
button[key="bg_blur"],
button[key="bg_white"],
button[key="bg_black"],
button[key="bg_custom_color"],
button[key="bg_custom_image"],
button[key^="bg_Background"],
button[key="save_project"],
button[key="load_project"],
button[key="delete_project"] {
    background: rgba(21, 25, 50, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    color: #e8e8e8 !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 16px !important;
    margin-bottom: 1.5rem !important;
}

button[key="bg_transparent"]:hover,
button[key="bg_blur"]:hover,
button[key="bg_white"]:hover,
button[key="bg_black"]:hover,
button[key="bg_custom_color"]:hover,
button[key="bg_custom_image"]:hover,
button[key^="bg_Background"]:hover,
button[key="save_project"]:hover,
button[key="load_project"]:hover,
button[key="delete_project"]:hover {
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
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(155, 89, 182, 0.2) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    border: 2px solid rgba(212, 175, 55, 0.3);
}

.flow-step.active .flow-icon {
    background: linear-gradient(135deg, #d4af37 0%, #c9a227 100%);
    border-color: #d4af37;
    box-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
}

.flow-label {
    color: #cbd5e1;
    font-size: 0.9rem;
    font-weight: 500;
}

.flow-step.active .flow-label {
    color: #d4af37;
    font-weight: 600;
}

/* Preview cards */
.preview-card {
    background: rgba(21, 25, 50, 0.4);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 20px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.preview-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #d4af37;
    margin-bottom: 1rem;
    font-family: 'Space Grotesk', sans-serif;
}

/* Modern footer */
.modern-footer {
    text-align: center;
    padding: 2rem;
    background: rgba(21, 25, 50, 0.4);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    margin-top: 3rem;
}

.modern-footer h3 {
    color: #d4af37;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.modern-footer p {
    color: #cbd5e1;
    margin: 0.25rem 0;
}

/* Metrics styling */
div[data-testid="stMetricValue"] {
    color: #d4af37 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* Sliders */
.stSlider > div > div > div {
    background: rgba(212, 175, 55, 0.3) !important;
}

.stSlider > div > div > div > div {
    background: #d4af37 !important;
}

/* Select boxes and inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(21, 25, 50, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 12px !important;
    color: #e8e8e8 !important;
    backdrop-filter: blur(10px) !important;
}

.stSelectbox > div > div:hover,
.stTextInput > div > div > input:hover,
.stNumberInput > div > div > input:hover {
    border-color: rgba(212, 175, 55, 0.4) !important;
}

/* Color picker */
.stColorPicker > div > div {
    background: rgba(21, 25, 50, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 12px !important;
}

/* Radio buttons */
.stRadio > div {
    background: rgba(21, 25, 50, 0.4);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(212, 175, 55, 0.2);
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(21, 25, 50, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 12px !important;
    color: #e8e8e8 !important;
    backdrop-filter: blur(10px) !important;
}

.streamlit-expanderHeader:hover {
    border-color: rgba(212, 175, 55, 0.4) !important;
}

/* Success/Info/Warning boxes */
.stAlert {
    background: rgba(21, 25, 50, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: rgba(21, 25, 50, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    color: #e8e8e8 !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 12px !important;
}

.stDownloadButton > button:hover {
    background: rgba(21, 25, 50, 0.8) !important;
    border-color: rgba(212, 175, 55, 0.4) !important;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2) !important;
}
</style>
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the segmentation model"""
    model = getattr(segmentation_models, CONFIG["model_name"])(
        pretrained=False, num_classes=CONFIG["num_classes"]
    )
    
    if Path(CONFIG["model_path"]).exists():
        try:
            state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
            
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Try to load with strict=False to handle mismatches
            model.load_state_dict(state_dict, strict=False)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load model weights: {str(e)}. Using pretrained model instead.")
            # Load pretrained model as fallback
            model = getattr(segmentation_models, CONFIG["model_name"])(
                pretrained=True, num_classes=21  # Standard COCO classes
            )
    else:
        st.warning("⚠️ Model weights not found. Using pretrained model.")
        # Load pretrained model as fallback
        model = getattr(segmentation_models, CONFIG["model_name"])(
            pretrained=True, num_classes=21  # Standard COCO classes
        )
    
    model.to(CONFIG["device"])
    model.eval()
    return model

def preprocess_image(image_array):
    """Preprocess image for model input"""
    img = cv2.resize(image_array, CONFIG["img_size"])
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(CONFIG["device"])
    return img_tensor

def generate_mask(model, image_array):
    """Generate segmentation mask"""
    with torch.no_grad():
        img_tensor = preprocess_image(image_array)
        output = model(img_tensor)['out']
        probs = F.softmax(output, dim=1)
        fg_prob = probs[0, 1].cpu().numpy()
        
    fg_prob_resized = cv2.resize(fg_prob, (image_array.shape[1], image_array.shape[0]))
    mask = (fg_prob_resized > CONFIG["fg_thresh"]).astype(np.uint8) * 255
    return mask

def refine_mask(mask, kernel_size=5, iterations=2):
    """Refine mask using morphological operations"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def apply_background(image, mask, mode, bg_path=None, custom_color=(255, 255, 255)):
    """Apply background based on selected mode"""
    pil_img = Image.fromarray(image)
    pil_mask = Image.fromarray(mask).convert('L')
    
    if mode == "Transparent":
        result = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        result.paste(pil_img, mask=pil_mask)
        return result
    
    elif mode == "Blur Background":
        blurred = pil_img.filter(ImageFilter.GaussianBlur(15))
        result = Image.composite(pil_img, blurred, pil_mask)
        return result
    
    elif mode in ["White Background", "Black Background"]:
        color = (255, 255, 255) if mode == "White Background" else (0, 0, 0)
        bg = Image.new("RGB", pil_img.size, color)
        result = Image.composite(pil_img, bg, pil_mask)
        return result
    
    elif mode == "Custom Color":
        bg = Image.new("RGB", pil_img.size, custom_color)
        result = Image.composite(pil_img, bg, pil_mask)
        return result
    
    elif mode == "Custom Image" and bg_path:
        bg = Image.open(bg_path).convert("RGB")
        bg = bg.resize(pil_img.size, Image.LANCZOS)
        result = Image.composite(pil_img, bg, pil_mask)
        return result
    
    elif mode in BG_IMAGE_PATHS.keys() and bg_path:
        if Path(bg_path).exists():
            bg = Image.open(bg_path).convert("RGB")
            bg = bg.resize(pil_img.size, Image.LANCZOS)
            result = Image.composite(pil_img, bg, pil_mask)
            return result
    
    return pil_img

def apply_sepia(img):
    """Apply sepia filter"""
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    img_array = np.array(img)
    sepia_img = cv2.transform(img_array, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_img)

def apply_vintage(img):
    """Apply vintage filter"""
    img = ImageEnhance.Contrast(img).enhance(0.8)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Color(img).enhance(0.7)
    return img

def apply_cool_tone(img):
    """Apply cool tone filter"""
    img_array = np.array(img).astype(np.float32)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.9, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def apply_warm_tone(img):
    """Apply warm tone filter"""
    img_array = np.array(img).astype(np.float32)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1, 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.9, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def crop_image(img, preset):
    """Crop image based on preset"""
    if preset is None:
        return img
    
    aspect_ratio = preset
    w, h = img.size
    target_aspect = aspect_ratio[0] / aspect_ratio[1]
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    
    return img

def get_download_button(img, format_type, quality, label, filename, key):
    """Create download button for image"""
    buf = BytesIO()
    
    if format_type == "PNG":
        img_to_save = img.convert("RGBA") if img.mode != "RGBA" else img
        img_to_save.save(buf, format="PNG")
        mime = "image/png"
    elif format_type == "JPEG":
        img_to_save = img.convert("RGB") if img.mode != "RGB" else img
        img_to_save.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    elif format_type == "WEBP":
        img_to_save = img.convert("RGB") if img.mode == "RGBA" else img
        img_to_save.save(buf, format="WEBP", quality=quality)
        mime = "image/webp"
    
    buf.seek(0)
    st.download_button(label, buf.getvalue(), filename, mime, key=key, use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="OneView Pro - AI Image Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="modern-header">
        <h1>OneView Pro</h1>
        <p>AI-Powered Professional Image Studio</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'mask' not in st.session_state:
        st.session_state.mask = None
    if 'extraction_mode' not in st.session_state:
        st.session_state.extraction_mode = "Transparent"
    if 'custom_color' not in st.session_state:
        st.session_state.custom_color = (255, 255, 255)
    if 'brightness' not in st.session_state:
        st.session_state.brightness = 1.0
    if 'contrast' not in st.session_state:
        st.session_state.contrast = 1.0
    if 'saturation' not in st.session_state:
        st.session_state.saturation = 1.0
    if 'filter_type' not in st.session_state:
        st.session_state.filter_type = "None"
    if 'crop_preset' not in st.session_state:
        st.session_state.crop_preset = None
    if 'resize_percent' not in st.session_state:
        st.session_state.resize_percent = 100
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'custom_bg_path' not in st.session_state:
        st.session_state.custom_bg_path = None
    if 'saved_projects' not in st.session_state:
        st.session_state.saved_projects = {}
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎨 Control Panel")
        
        # Project Management
        st.markdown("---")
        st.markdown("### 💾 Manage Projects")
        
        project_name = st.text_input("Project Name", placeholder="Enter project name...")
        
        if st.button("💾 Save Project", key="save_project", use_container_width=True):
            if project_name and st.session_state.original_image is not None:
                st.session_state.saved_projects[project_name] = {
                    'original_image': st.session_state.original_image.copy(),
                    'mask': st.session_state.mask.copy() if st.session_state.mask is not None else None,
                    'extraction_mode': st.session_state.extraction_mode,
                    'custom_color': st.session_state.custom_color,
                    'brightness': st.session_state.brightness,
                    'contrast': st.session_state.contrast,
                    'saturation': st.session_state.saturation,
                    'filter_type': st.session_state.filter_type,
                    'crop_preset': st.session_state.crop_preset,
                    'resize_percent': st.session_state.resize_percent,
                }
                st.success(f"✅ Project '{project_name}' saved!")
            else:
                st.error("❌ Please enter a name and load an image first")
        
        if st.session_state.saved_projects:
            selected_project = st.selectbox(
                "Select Project",
                options=list(st.session_state.saved_projects.keys()),
                key="project_selector"
            )
            
            # Load and Delete buttons side by side
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📂 Load", key="load_project", use_container_width=True):
                    if selected_project:
                        project = st.session_state.saved_projects[selected_project]
                        st.session_state.original_image = project['original_image']
                        st.session_state.current_image = project['original_image']
                        st.session_state.mask = project['mask']
                        st.session_state.extraction_mode = project['extraction_mode']
                        st.session_state.custom_color = project['custom_color']
                        st.session_state.brightness = project['brightness']
                        st.session_state.contrast = project['contrast']
                        st.session_state.saturation = project['saturation']
                        st.session_state.filter_type = project['filter_type']
                        st.session_state.crop_preset = project['crop_preset']
                        st.session_state.resize_percent = project['resize_percent']
                        st.success(f"✅ Loaded project '{selected_project}'")
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Delete", key="delete_project", use_container_width=True):
                    if selected_project:
                        del st.session_state.saved_projects[selected_project]
                        st.success(f"✅ Deleted project '{selected_project}'")
                        st.rerun()
        
        # Export Settings
        st.markdown("---")
        st.markdown("### 📤 Export Settings")
        export_format = st.selectbox("Format", ["PNG", "JPEG", "WEBP"])
        quality = st.slider("Quality", 1, 100, 95) if export_format != "PNG" else 95
        
        # Quick Actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Load model
    model = load_model()
    
    # Main content area - Horizontal Workflow Tabs
    tabs = st.tabs(["📸 Upload", "🎯 Extract", "🎨 Background", "✨ Enhance", "💾 Export"])
    
    # Helper function for filters and adjustments
    def apply_filters_and_adjustments(img):
        """Apply filters and adjustments to image"""
        if st.session_state.filter_type != "None":
            img = FILTERS[st.session_state.filter_type](img)
        
        if st.session_state.brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(st.session_state.brightness)
        
        if st.session_state.contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(st.session_state.contrast)
        
        if st.session_state.saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(st.session_state.saturation)
        
        return img
    
    # Upload Tab
    with tabs[0]:
        st.markdown("### Upload Your Image")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="Upload an image to start processing"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_array = np.array(image.convert('RGB'))
                st.session_state.original_image = image_array
                st.session_state.current_image = image_array
                st.success("✅ Image uploaded successfully!")
        
        with col2:
            st.markdown("### 🎭 Try Demo Images")
            for demo_name, demo_path in DEMO_IMAGE_PATHS.items():
                if st.button(demo_name, key=f"demo_{demo_name}", use_container_width=True):
                    if Path(demo_path).exists():
                        image = Image.open(demo_path)
                        image_array = np.array(image.convert('RGB'))
                        st.session_state.original_image = image_array
                        st.session_state.current_image = image_array
                        st.success(f"✅ Loaded {demo_name}")
                        st.rerun()
                    else:
                        st.error(f"Demo image not found: {demo_path}")
        
        if st.session_state.original_image is not None:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown('<div class="preview-header">Original Image</div>', unsafe_allow_html=True)
            display_img = Image.fromarray(st.session_state.original_image).copy()
            display_img.thumbnail((800, 600), Image.LANCZOS)
            st.image(display_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.current_step = 0
    
    # Extract Tab
    with tabs[1]:
        if st.session_state.original_image is None:
            st.warning("⚠️ Please upload an image first in the Upload tab")
        else:
            st.markdown("### AI-Powered Subject Extraction")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Extraction Controls")
                
                if st.button("🚀 Extract Subject", key="extract_btn", use_container_width=True):
                    with st.spinner("🔮 AI is processing your image..."):
                        mask = generate_mask(model, st.session_state.original_image)
                        st.session_state.mask = mask
                        st.success("✅ Extraction complete!")
                        st.rerun()
                
                if st.session_state.mask is not None:
                    st.markdown("---")
                    st.markdown("#### Refine Mask")
                    
                    kernel_size = st.slider("Smoothness", 1, 15, 5, step=2, key="kernel_slider")
                    iterations = st.slider("Iterations", 1, 5, 2, key="iter_slider")
                    
                    if st.button("🔧 Refine Mask", key="refine_btn", use_container_width=True):
                        st.session_state.mask = refine_mask(
                            st.session_state.mask, kernel_size, iterations
                        )
                        st.success("✅ Mask refined!")
                        st.rerun()
                    
                    threshold = st.slider("Threshold", 0.0, 1.0, CONFIG["fg_thresh"], 0.05, key="thresh_slider")
                    if threshold != CONFIG["fg_thresh"]:
                        CONFIG["fg_thresh"] = threshold
                        if st.button("🎯 Re-extract", key="reextract_btn", use_container_width=True):
                            mask = generate_mask(model, st.session_state.original_image)
                            st.session_state.mask = mask
                            st.success("✅ Re-extracted with new threshold!")
                            st.rerun()
            
            with col2:
                if st.session_state.mask is not None:
                    st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                    st.markdown('<div class="preview-header">Extraction Preview</div>', unsafe_allow_html=True)
                    
                    preview_img = apply_background(
                        st.session_state.original_image,
                        st.session_state.mask,
                        "Transparent",
                        None,
                        None
                    )
                    
                    display_preview = preview_img.copy()
                    display_preview.thumbnail((800, 600), Image.LANCZOS)
                    st.image(display_preview, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("👆 Click 'Extract Subject' to see the preview")
            
            st.session_state.current_step = 1
    
    # Background Tab
    with tabs[2]:
        if st.session_state.mask is None:
            st.warning("⚠️ Please extract the subject first in the Extract tab")
        else:
            st.markdown("### Choose Your Background")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Background Options")
                
                # Background mode buttons with special styling
                if st.button("🔲 Transparent", key="bg_transparent", use_container_width=True):
                    st.session_state.extraction_mode = "Transparent"
                    st.rerun()
                
                if st.button("💫 Blur Background", key="bg_blur", use_container_width=True):
                    st.session_state.extraction_mode = "Blur Background"
                    st.rerun()
                
                if st.button("⚪ White Background", key="bg_white", use_container_width=True):
                    st.session_state.extraction_mode = "White Background"
                    st.rerun()
                
                if st.button("⚫ Black Background", key="bg_black", use_container_width=True):
                    st.session_state.extraction_mode = "Black Background"
                    st.rerun()
                
                st.markdown("---")
                
                # Custom color with color picker
                st.markdown("#### 🎨 Custom Color")
                custom_color = st.color_picker("Pick a color", "#FFFFFF", key="color_picker")
                st.session_state.custom_color = tuple(int(custom_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                if st.button("🎨 Apply Custom Color", key="bg_custom_color", use_container_width=True):
                    st.session_state.extraction_mode = "Custom Color"
                    st.rerun()
                
                st.markdown("---")
                
                # Custom image upload
                st.markdown("#### 🖼️ Custom Image")
                custom_bg = st.file_uploader("Upload background", type=['png', 'jpg', 'jpeg'], key="custom_bg_upload")
                
                if custom_bg:
                    st.session_state.custom_bg_path = custom_bg
                    if st.button("📷 Apply Custom Image", key="bg_custom_image", use_container_width=True):
                        st.session_state.extraction_mode = "Custom Image"
                        st.rerun()
                
                st.markdown("---")
                
                # Preset backgrounds
                st.markdown("#### 🌅 Preset Backgrounds")
                for bg_name in BG_IMAGE_PATHS.keys():
                    if st.button(bg_name, key=f"bg_{bg_name}", use_container_width=True):
                        st.session_state.extraction_mode = bg_name
                        st.rerun()
            
            with col2:
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="preview-header">Preview: {st.session_state.extraction_mode}</div>', unsafe_allow_html=True)
                
                bg_path = None
                if st.session_state.extraction_mode in BG_IMAGE_PATHS:
                    bg_path = BG_IMAGE_PATHS[st.session_state.extraction_mode]
                elif st.session_state.extraction_mode == "Custom Image" and st.session_state.custom_bg_path:
                    bg_path = st.session_state.custom_bg_path
                
                result_img = apply_background(
                    st.session_state.original_image,
                    st.session_state.mask,
                    st.session_state.extraction_mode,
                    bg_path,
                    st.session_state.custom_color
                )
                
                display_result = result_img.copy()
                display_result.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_result, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.session_state.current_step = 2
    
    # Enhance Tab
    with tabs[3]:
        if st.session_state.mask is None:
            st.warning("⚠️ Please extract the subject first")
        else:
            st.markdown("### Enhance Your Image")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🎨 Filters")
                st.session_state.filter_type = st.selectbox(
                    "Choose Filter",
                    options=list(FILTERS.keys()),
                    index=list(FILTERS.keys()).index(st.session_state.filter_type),
                    key="filter_select"
                )
                
                st.markdown("---")
                st.markdown("#### 🎛️ Adjustments")
                
                st.session_state.brightness = st.slider(
                    "Brightness",
                    0.5, 2.0,
                    st.session_state.brightness,
                    0.1,
                    key="brightness_slider"
                )
                
                st.session_state.contrast = st.slider(
                    "Contrast",
                    0.5, 2.0,
                    st.session_state.contrast,
                    0.1,
                    key="contrast_slider"
                )
                
                st.session_state.saturation = st.slider(
                    "Saturation",
                    0.0, 2.0,
                    st.session_state.saturation,
                    0.1,
                    key="saturation_slider"
                )
                
                st.markdown("---")
                st.markdown("#### ✂️ Crop & Resize")
                
                crop_option = st.selectbox(
                    "Crop Preset",
                    options=list(CROP_PRESETS.keys()),
                    key="crop_select"
                )
                st.session_state.crop_preset = CROP_PRESETS[crop_option]
                
                st.session_state.resize_percent = st.slider(
                    "Resize %",
                    10, 200,
                    st.session_state.resize_percent,
                    5,
                    key="resize_slider"
                )
            
            with col2:
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                if st.session_state.extraction_mode == "Custom Image" and st.session_state.custom_bg_path:
                    bg_path = st.session_state.custom_bg_path
                
                result_img = apply_background(
                    st.session_state.original_image,
                    st.session_state.mask,
                    st.session_state.extraction_mode,
                    bg_path,
                    st.session_state.custom_color
                )
                
                result_img = apply_filters_and_adjustments(result_img)
                result_img = crop_image(result_img, st.session_state.crop_preset)
                
                if st.session_state.resize_percent != 100:
                    orig_w, orig_h = result_img.size
                    new_w = int(orig_w * st.session_state.resize_percent / 100)
                    new_h = int(orig_h * st.session_state.resize_percent / 100)
                    result_img = result_img.resize((new_w, new_h), Image.LANCZOS)
                
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                st.markdown('<div class="preview-header">Enhanced Preview</div>', unsafe_allow_html=True)
                display_enhanced = result_img.copy()
                display_enhanced.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_enhanced, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Comparison slider
                st.markdown("---")
                st.markdown("#### 🔄 Compare with Original")
                
                blend_value = st.slider("Blend Original/Result", 0.0, 1.0, 0.5, 0.1, key="blend_slider")
                
                original_img = Image.fromarray(st.session_state.original_image)
                result_rgb = result_img.convert("RGB")
                
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                orig_rgb = original_img.convert('RGB')
                blended = Image.blend(orig_rgb, result_rgb, float(blend_value))
                
                st.markdown('<div class="preview-card">', unsafe_allow_html=True)
                display_blended = blended.copy()
                display_blended.thumbnail((800, 600), Image.LANCZOS)
                st.image(display_blended, use_container_width=True, caption=f"Blend: {int(blend_value*100)}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.session_state.current_step = 3
    
    # Export Tab
    with tabs[4]:
        if st.session_state.mask is None:
            st.warning("⚠️ Please complete the previous steps first")
        else:
            st.markdown("### Export Your Work")
            
            bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
            if st.session_state.extraction_mode == "Custom Image" and st.session_state.custom_bg_path:
                bg_path = st.session_state.custom_bg_path
            
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
