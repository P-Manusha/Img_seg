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

# Paths - Updated for deployment
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

/* Custom scrollbar */
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
</style>
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_sepia(img):
    """Apply sepia tone filter"""
    img_array = np.array(img)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(img_array, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_img)

def apply_vintage(img):
    """Apply vintage filter"""
    img = ImageEnhance.Contrast(img).enhance(0.9)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Color(img).enhance(0.8)
    return img

def apply_cool_tone(img):
    """Apply cool tone filter"""
    img_array = np.array(img)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def apply_warm_tone(img):
    """Apply warm tone filter"""
    img_array = np.array(img)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

@st.cache_resource
def load_model(model_path, device, num_classes=2):
    """Load the segmentation model"""
    try:
        model = getattr(segmentation_models, CONFIG["model_name"])(pretrained=False)
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            st.warning("⚠️ Model file not found. Using pretrained model.")
            model = getattr(segmentation_models, CONFIG["model_name"])(pretrained=True)
        
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_mask(model, img_array, device, img_size):
    """Generate segmentation mask"""
    if model is None:
        return np.zeros(img_size)
    
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else img_array
    img_resized = cv2.resize(img_rgb, img_size)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)['out']
        prob = F.softmax(output, dim=1)[0, 1].cpu().numpy()
    
    return prob

def postprocess_mask(prob, fg_thresh=0.5, min_area=500):
    """Post-process the predicted mask"""
    mask = (prob > fg_thresh).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        mask_filtered = np.zeros_like(mask)
        cv2.drawContours(mask_filtered, filtered_contours, -1, 255, thickness=cv2.FILLED)
        mask = mask_filtered
    
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def apply_background(img_array, mask_bin, mode, bg_path=None, custom_color=(255, 255, 255)):
    """Apply background based on selected mode"""
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_bin * 255).convert("L")
    
    if mode == "Transparent":
        img_rgba = img_pil.convert("RGBA")
        img_rgba.putalpha(mask_pil)
        return img_rgba
    
    elif mode == "White":
        bg = Image.new("RGB", img_pil.size, (255, 255, 255))
    elif mode == "Black":
        bg = Image.new("RGB", img_pil.size, (0, 0, 0))
    elif mode == "Custom Color":
        bg = Image.new("RGB", img_pil.size, custom_color)
    elif mode == "Blur":
        bg = img_pil.filter(ImageFilter.GaussianBlur(25))
    elif mode.startswith("Background") and bg_path:
        try:
            bg = Image.open(bg_path).convert("RGB").resize(img_pil.size, Image.LANCZOS)
        except:
            bg = Image.new("RGB", img_pil.size, (255, 255, 255))
    else:
        bg = Image.new("RGB", img_pil.size, (255, 255, 255))
    
    result = Image.composite(img_pil.convert("RGB"), bg, mask_pil)
    return result

def apply_filters_and_adjustments(img):
    """Apply filters and adjustments from session state"""
    if "selected_filter" in st.session_state:
        img = FILTERS[st.session_state.selected_filter](img)
    
    if "brightness" in st.session_state:
        img = ImageEnhance.Brightness(img).enhance(st.session_state.brightness)
    if "contrast" in st.session_state:
        img = ImageEnhance.Contrast(img).enhance(st.session_state.contrast)
    if "saturation" in st.session_state:
        img = ImageEnhance.Color(img).enhance(st.session_state.saturation)
    if "sharpness" in st.session_state:
        img = ImageEnhance.Sharpness(img).enhance(st.session_state.sharpness)
    
    return img

def crop_image(img, preset_name):
    """Crop image based on preset"""
    if preset_name == "Freeform" or CROP_PRESETS[preset_name] is None:
        return img
    
    aspect_ratio = CROP_PRESETS[preset_name]
    w, h = img.size
    target_ratio = aspect_ratio[0] / aspect_ratio[1]
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
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="OneView - Professional Image Processing", 
                       page_icon="🎨", 
                       layout="wide",
                       initial_sidebar_state="expanded")
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>OneView</h1>
        <p>Professional AI-Powered Image Processing Suite</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "batch_mode" not in st.session_state:
        st.session_state.batch_mode = False
    
# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="OneView - Professional Image Processing", 
                       page_icon="🎨", 
                       layout="wide",
                       initial_sidebar_state="expanded")
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>OneView</h1>
        <p>Professional AI-Powered Image Processing Suite</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "batch_mode" not in st.session_state:
        st.session_state.batch_mode = False
    if "fg_thresh" not in st.session_state:
        st.session_state.fg_thresh = CONFIG["fg_thresh"]
    if "min_area" not in st.session_state:
        st.session_state.min_area = 500
    if "extraction_mode" not in st.session_state:
        st.session_state.extraction_mode = "Transparent"
    if "custom_color" not in st.session_state:
        st.session_state.custom_color = (255, 255, 255)
    if "selected_filter" not in st.session_state:
        st.session_state.selected_filter = "None"
    if "brightness" not in st.session_state:
        st.session_state.brightness = 1.0
    if "contrast" not in st.session_state:
        st.session_state.contrast = 1.0
    if "saturation" not in st.session_state:
        st.session_state.saturation = 1.0
    if "sharpness" not in st.session_state:
        st.session_state.sharpness = 1.0
    if "crop_preset" not in st.session_state:
        st.session_state.crop_preset = "Freeform"
    if "resize_percent" not in st.session_state:
        st.session_state.resize_percent = 100
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model Loading
        model = load_model(CONFIG["model_path"], CONFIG["device"], CONFIG["num_classes"])
        
        # Batch Mode Toggle
        st.session_state.batch_mode = st.checkbox("🔄 Batch Processing Mode", value=st.session_state.batch_mode)
        
        st.markdown("---")
        
        # Background Settings
        st.markdown("### 🎨 Background Settings")
        extraction_options = ["Transparent", "White", "Black", "Custom Color", "Blur"] + list(BG_IMAGE_PATHS.keys())
        st.session_state.extraction_mode = st.selectbox("Background Type", extraction_options, 
                                                         index=extraction_options.index(st.session_state.extraction_mode))
        
        if st.session_state.extraction_mode == "Custom Color":
            color = st.color_picker("Pick a color", "#ffffff")
            st.session_state.custom_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        st.markdown("---")
        
        # Segmentation Settings
        st.markdown("### 🎯 Segmentation Settings")
        st.session_state.fg_thresh = st.slider("Foreground Threshold", 0.0, 1.0, st.session_state.fg_thresh, 0.05)
        st.session_state.min_area = st.slider("Minimum Area", 0, 2000, st.session_state.min_area, 100)
        
        st.markdown("---")
        
        # Filters & Adjustments
        st.markdown("### ✨ Filters & Adjustments")
        st.session_state.selected_filter = st.selectbox("Filter", list(FILTERS.keys()), 
                                                         index=list(FILTERS.keys()).index(st.session_state.selected_filter))
        st.session_state.brightness = st.slider("Brightness", 0.5, 2.0, st.session_state.brightness, 0.1)
        st.session_state.contrast = st.slider("Contrast", 0.5, 2.0, st.session_state.contrast, 0.1)
        st.session_state.saturation = st.slider("Saturation", 0.0, 2.0, st.session_state.saturation, 0.1)
        st.session_state.sharpness = st.slider("Sharpness", 0.0, 2.0, st.session_state.sharpness, 0.1)
        
        st.markdown("---")
        
        # Crop & Resize
        st.markdown("### ✂️ Crop & Resize")
        st.session_state.crop_preset = st.selectbox("Crop Preset", list(CROP_PRESETS.keys()), 
                                                     index=list(CROP_PRESETS.keys()).index(st.session_state.crop_preset))
        st.session_state.resize_percent = st.slider("Resize %", 10, 200, st.session_state.resize_percent, 5)
    
    # Main Content - Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🖼️ Demo Images", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 📤 Upload Your Images")
        
        if st.session_state.batch_mode:
            uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], 
                                             accept_multiple_files=True, key="batch_uploader")
            if uploaded_files:
                st.session_state.uploaded_images = []
                for file in uploaded_files:
                    img = Image.open(file).convert("RGB")
                    st.session_state.uploaded_images.append(np.array(img))
                st.success(f"✅ Loaded {len(uploaded_files)} images")
        else:
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="single_uploader")
            if uploaded_file:
                original_img = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(original_img)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Image")
                    st.image(original_img, use_container_width=True)
                
                if st.button("🚀 Process Image", use_container_width=True):
                    with st.spinner("Processing..."):
                        # Predict mask
                        prob = predict_mask(model, img_array, CONFIG["device"], CONFIG["img_size"])
                        mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                        mask_bin = (mask > 127).astype(np.uint8)
                        
                        # Apply background
                        bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                        result_pil = apply_background(img_array, mask_bin, st.session_state.extraction_mode, 
                                                     bg_path, st.session_state.custom_color)
                        
                        # Apply filters
                        result_pil = apply_filters_and_adjustments(result_pil)
                        
                        # Crop
                        result_pil = crop_image(result_pil, st.session_state.crop_preset)
                        
                        # Resize
                        if st.session_state.resize_percent != 100:
                            w, h = result_pil.size
                            new_w = int(w * st.session_state.resize_percent / 100)
                            new_h = int(h * st.session_state.resize_percent / 100)
                            result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
                        
                        st.session_state.result_image = result_pil
                        st.session_state.mask_image = mask
                
                if "result_image" in st.session_state:
                    with col2:
                        st.markdown("#### Processed Image")
                        st.image(st.session_state.result_image, use_container_width=True)
                    
                    # Download options
                    st.markdown("---")
                    st.markdown("### 📥 Download Options")
                    
                    dcol1, dcol2, dcol3 = st.columns(3)
                    
                    with dcol1:
                        export_format = st.selectbox("Format", ["PNG", "JPEG", "WEBP"])
                    with dcol2:
                        quality = st.slider("Quality", 1, 100, 95) if export_format != "PNG" else 95
                    with dcol3:
                        st.write("")
                    
                    # Download buttons
                    buf = BytesIO()
                    result_img = st.session_state.result_image
                    
                    if export_format == "PNG" and result_img.mode == "RGBA":
                        result_img.save(buf, format="PNG")
                    elif export_format in ["JPEG", "JPG"]:
                        result_img.convert("RGB").save(buf, format="JPEG", quality=quality)
                    else:
                        result_img.convert("RGB").save(buf, format=export_format, quality=quality)
                    
                    st.download_button("⬇️ Download Processed Image", buf.getvalue(), 
                                      f"oneview_processed.{export_format.lower()}", 
                                      f"image/{export_format.lower()}", use_container_width=True)
    
    with tab2:
        st.markdown("### 🖼️ Demo Images")
        st.info("👇 Try our demo images to see OneView in action!")
        
        demo_cols = st.columns(len(DEMO_IMAGE_PATHS))
        for idx, (name, path) in enumerate(DEMO_IMAGE_PATHS.items()):
            with demo_cols[idx]:
                if os.path.exists(path):
                    st.image(path, caption=name, use_container_width=True)
                else:
                    st.warning(f"Demo image not found: {name}")
    
    with tab3:
        st.markdown("### ℹ️ About OneView")
        st.markdown("""
        **OneView** is a professional AI-powered image processing application that provides:
        
        - 🎯 **Advanced Segmentation**: AI-powered background removal
        - 🎨 **Custom Backgrounds**: Transparent, solid colors, blur, or custom images
        - ✨ **Professional Filters**: Sepia, vintage, cool/warm tones, and more
        - ⚙️ **Fine-tuned Adjustments**: Brightness, contrast, saturation, sharpness
        - ✂️ **Smart Cropping**: Presets for social media platforms
        - 🔄 **Batch Processing**: Process multiple images at once
        - 📥 **Multiple Export Formats**: PNG, JPEG, WEBP
        
        **Technology Stack:**
        - Deep Learning: PyTorch & DeepLabV3
        - Image Processing: OpenCV & PIL
        - UI Framework: Streamlit
        """)
    
    # Batch Processing Section
    if st.session_state.batch_mode and len(st.session_state.uploaded_images) > 0:
        st.markdown("---")
        st.markdown("### 🔄 Batch Processing")
        st.info(f"📊 {len(st.session_state.uploaded_images)} images ready for processing")
        
        if st.button("▶️ Process All Images", use_container_width=True):
            progress_bar = st.progress(0)
            batch_results = []
            
            for idx, img_array in enumerate(st.session_state.uploaded_images):
                prob = predict_mask(model, img_array, CONFIG["device"], CONFIG["img_size"])
                mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                mask_bin = (mask > 127).astype(np.uint8)
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(img_array, mask_bin, st.session_state.extraction_mode, 
                                             bg_path, st.session_state.custom_color)
                result_pil = apply_filters_and_adjustments(result_pil)
                result_pil = crop_image(result_pil, st.session_state.crop_preset)
                
                if st.session_state.resize_percent != 100:
                    w, h = result_pil.size
                    new_w = int(w * st.session_state.resize_percent / 100)
                    new_h = int(h * st.session_state.resize_percent / 100)
                    result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
                
                batch_results.append(result_pil)
                progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
            
            st.success("✅ All images processed successfully!")
            
            # Display results
            st.markdown("#### 📸 Results Preview")
            cols = st.columns(min(3, len(batch_results)))
            for idx, result in enumerate(batch_results):
                with cols[idx % len(cols)]:
                    display_img = result.copy()
                    display_img.thumbnail((300, 300), Image.LANCZOS)
                    st.image(display_img, caption=f"Image {idx + 1}", use_container_width=True)
            
            # Download section
            st.markdown("#### 📥 Download Results")
            export_format = st.selectbox("Export Format", ["PNG", "JPEG", "WEBP"], key="batch_format")
            quality = st.slider("Quality", 1, 100, 95, key="batch_quality") if export_format != "PNG" else 95
            
            download_cols = st.columns(min(4, len(batch_results)))
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
                                      key=f"batch_dl_{idx}", use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-professional">
        <h3>OneView</h3>
        <p>Professional AI-Powered Image Processing Solution</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">Developed by Manusha</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
