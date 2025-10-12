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
# PROFESSIONAL CUSTOM CSS
# ============================================================================
CUSTOM_CSS = """
<style>
/* Import Professional Fonts */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Crimson Text', 'Georgia', serif;
    box-sizing: border-box;
}

/* Main App Background - Professional Gradient */
.stApp {
    background: linear-gradient(135deg, #1a2332 0%, #2c3e50 50%, #34495e 100%) !important;
    min-height: 100vh;
}

/* Alternative gradient options (uncomment to use):
   Navy to Charcoal: background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #363636 100%) !important;
   Deep Blue Classic: background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
   Sophisticated Gray: background: linear-gradient(135deg, #434343 0%, #2c2c2c 100%) !important;
*/

/* Main Header */
.main-header {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(248, 249, 250, 0.95));
    padding: 3rem 2.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    margin-bottom: 2.5rem;
    border: 1px solid rgba(52, 73, 94, 0.1);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #2c3e50 0%, #3498db 50%, #2c3e50 100%);
}

.main-header h1 {
    margin: 0;
    color: #1a2332;
    font-size: 3.2rem;
    font-weight: 700;
    font-family: 'Playfair Display', 'Georgia', serif;
    letter-spacing: -0.5px;
    text-transform: uppercase;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.main-header p {
    margin: 1rem 0 0 0;
    color: #546e7a;
    font-size: 1.15rem;
    font-weight: 500;
    font-family: 'Montserrat', sans-serif;
    letter-spacing: 0.5px;
}

/* Sidebar Styling */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
}

.css-1d391kg .block-container, [data-testid="stSidebar"] .block-container {
    padding: 2rem 1rem;
}

[data-testid="stSidebar"] label {
    color: #ecf0f1 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stSlider label {
    color: #ecf0f1 !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: #bdc3c7 !important;
    font-family: 'Crimson Text', serif !important;
    font-weight: 400 !important;
    text-transform: none !important;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(255, 255, 255, 0.95);
    padding: 1.2rem;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(52, 73, 94, 0.1);
}

.stTabs [data-baseweb="tab"] {
    height: 3rem;
    padding: 0 2rem;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    color: #2c3e50;
    border-radius: 10px;
    border: 1px solid rgba(52, 73, 94, 0.2);
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 5px 20px rgba(44, 62, 80, 0.4) !important;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Montserrat', sans-serif;
    letter-spacing: 0.5px;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
    transition: all 0.3s ease;
    text-transform: uppercase;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
}

/* Download Button Special Styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'Montserrat', sans-serif;
    letter-spacing: 0.5px;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
    transition: all 0.3s ease;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #229954 0%, #1e8449 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(39, 174, 96, 0.4);
}

/* Input Fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(52, 73, 94, 0.3);
    border-radius: 8px;
    color: #2c3e50;
    font-family: 'Crimson Text', serif;
    font-size: 1rem;
    padding: 0.6rem;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
}

/* Slider Styling */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
}

.stSlider > div > div > div > div {
    background: white;
    border: 2px solid #2c3e50;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    color: #2c3e50;
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    border-radius: 10px;
    border: 1px solid rgba(52, 73, 94, 0.2);
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
}

/* Info, Success, Warning, Error Messages */
.stAlert {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    border-left: 4px solid #3498db;
    font-family: 'Crimson Text', serif;
    color: #2c3e50;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
}

/* Metric Styling */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.95);
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid rgba(52, 73, 94, 0.2);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

[data-testid="metric-container"] label {
    color: #546e7a;
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2c3e50;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 2rem;
}

/* Footer Styling */
.footer-professional {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(248, 249, 250, 0.95));
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-top: 3rem;
    box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(52, 73, 94, 0.1);
}

.footer-professional h3 {
    color: #2c3e50;
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 1px;
}

.footer-professional p {
    color: #546e7a;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.95rem;
    margin: 0.5rem 0 0 0;
}

/* Column Headers */
h1, h2, h3, h4, h5, h6 {
    color: #ecf0f1 !important;
    font-family: 'Playfair Display', serif !important;
}

/* Markdown Text */
.stMarkdown {
    color: #ecf0f1;
}

/* Upload Area */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.95);
    border: 2px dashed rgba(52, 73, 94, 0.3);
    border-radius: 10px;
    transition: all 0.3s ease;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #3498db;
    background: rgba(255, 255, 255, 1);
    box-shadow: 0 4px 15px rgba(52, 152, 219, 0.2);
}

/* Radio Button Custom Styling */
.stRadio > div {
    background: rgba(255, 255, 255, 0.05);
    padding: 1rem;
    border-radius: 10px;
}

/* Checkbox Styling */
.stCheckbox {
    color: #ecf0f1;
}

/* Color Picker */
[data-testid="stColorPicker"] > div {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 8px;
    padding: 0.5rem;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(52, 73, 94, 0.1);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #3498db 0%, #2980b9 100%);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #2980b9 0%, #21618c 100%);
}
</style>
"""

# ============================================================================
# MODEL FUNCTIONS (keeping your existing functions)
# ============================================================================

@st.cache_resource
def load_model():
    """Load the segmentation model"""
    model = getattr(segmentation_models, CONFIG["model_name"])(pretrained=False, num_classes=CONFIG["num_classes"])
    
    if os.path.exists(CONFIG["model_path"]):
        checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(CONFIG["device"])
    model.eval()
    return model

def preprocess_image(image, size):
    """Preprocess image for model input"""
    img = cv2.resize(image, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def predict_mask(model, image, device, size):
    """Generate segmentation mask"""
    input_tensor = preprocess_image(image, size).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)["out"]
        prob = F.softmax(output, dim=1)[0, 1]
    
    return prob.cpu().numpy()

def postprocess_mask(prob, threshold=0.5, min_area=500):
    """Post-process the predicted mask"""
    mask = (prob > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (prob.shape[1], prob.shape[0]))
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    
    # Edge refinement
    mask = cv2.GaussianBlur(mask, (5, 5), 1)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask

def apply_background(image, mask, mode, bg_path=None, custom_color=None):
    """Apply background to extracted foreground"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert("L")
    
    if mode == "Transparent":
        result = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        result.paste(img_pil, mask=mask_pil)
    elif mode == "Custom Color" and custom_color:
        result = Image.new("RGB", img_pil.size, custom_color)
        result.paste(img_pil, mask=mask_pil)
    elif bg_path and os.path.exists(bg_path):
        bg = Image.open(bg_path).convert("RGB")
        bg = bg.resize(img_pil.size, Image.LANCZOS)
        result = bg.copy()
        result.paste(img_pil, mask=mask_pil)
    else:
        result = Image.new("RGB", img_pil.size, (255, 255, 255))
        result.paste(img_pil, mask=mask_pil)
    
    return result

# Filter functions
def apply_sepia(img):
    """Apply sepia filter"""
    img_array = np.array(img)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    sepia = cv2.transform(img_array, sepia_filter)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia)

def apply_vintage(img):
    """Apply vintage filter"""
    img = ImageEnhance.Color(img).enhance(0.7)
    img = ImageEnhance.Contrast(img).enhance(0.9)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    return apply_sepia(img)

def apply_cool_tone(img):
    """Apply cool tone filter"""
    img_array = np.array(img)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.9, 0, 255)  # Reduce red
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)  # Increase blue
    return Image.fromarray(img_array)

def apply_warm_tone(img):
    """Apply warm tone filter"""
    img_array = np.array(img)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1, 0, 255)  # Increase red
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.9, 0, 255)  # Reduce blue
    return Image.fromarray(img_array)

def apply_filters_and_adjustments(img):
    """Apply selected filters and adjustments"""
    # Apply filter
    if st.session_state.filter != "None":
        img = FILTERS[st.session_state.filter](img)
    
    # Apply adjustments
    img = ImageEnhance.Brightness(img).enhance(st.session_state.brightness / 100)
    img = ImageEnhance.Contrast(img).enhance(st.session_state.contrast / 100)
    img = ImageEnhance.Color(img).enhance(st.session_state.saturation / 100)
    img = ImageEnhance.Sharpness(img).enhance(st.session_state.sharpness / 100)
    
    return img

def crop_image(img, preset):
    """Crop image based on preset"""
    if preset == "Freeform" or CROP_PRESETS[preset] is None:
        return img
    
    ratio = CROP_PRESETS[preset]
    img_w, img_h = img.size
    target_ratio = ratio[0] / ratio[1]
    current_ratio = img_w / img_h
    
    if current_ratio > target_ratio:
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        img = img.crop((0, top, img_w, top + new_h))
    
    return img

# ============================================================================
# STREAMLIT APP
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'current_step': 1,
        'uploaded_image': None,
        'uploaded_images': [],
        'processed_mask': None,
        'result_image': None,
        'extraction_mode': 'Transparent',
        'custom_color': '#FFFFFF',
        'filter': 'None',
        'brightness': 100,
        'contrast': 100,
        'saturation': 100,
        'sharpness': 100,
        'crop_preset': 'Freeform',
        'resize_percent': 100,
        'fg_thresh': 0.3,
        'min_area': 500,
        'batch_mode': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
    initialize_session_state()
    
    # Load model
    model = load_model()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>OneView</h1>
        <p>Professional AI-Powered Background Removal & Image Enhancement</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Processing Settings
        with st.expander("🎯 Processing Settings", expanded=True):
            st.session_state.fg_thresh = st.slider(
                "Foreground Threshold",
                0.1, 0.9, st.session_state.fg_thresh, 0.05
            )
            st.session_state.min_area = st.slider(
                "Minimum Area",
                100, 2000, st.session_state.min_area, 100
            )
            st.session_state.batch_mode = st.checkbox(
                "Batch Processing Mode",
                st.session_state.batch_mode
            )
        
        # Background Settings
        with st.expander("🎨 Background Settings", expanded=True):
            st.session_state.extraction_mode = st.radio(
                "Background Type",
                ["Transparent", "Custom Color"] + list(BG_IMAGE_PATHS.keys()),
                index=["Transparent", "Custom Color", *BG_IMAGE_PATHS.keys()].index(st.session_state.extraction_mode)
            )
            
            if st.session_state.extraction_mode == "Custom Color":
                st.session_state.custom_color = st.color_picker(
                    "Select Color",
                    st.session_state.custom_color
                )
        
        # Enhancement Settings
        with st.expander("✨ Enhancement Settings"):
            st.session_state.filter = st.selectbox(
                "Filter",
                list(FILTERS.keys()),
                index=list(FILTERS.keys()).index(st.session_state.filter)
            )
            
            st.session_state.brightness = st.slider(
                "Brightness", 50, 150, st.session_state.brightness
            )
            st.session_state.contrast = st.slider(
                "Contrast", 50, 150, st.session_state.contrast
            )
            st.session_state.saturation = st.slider(
                "Saturation", 0, 200, st.session_state.saturation
            )
            st.session_state.sharpness = st.slider(
                "Sharpness", 50, 150, st.session_state.sharpness
            )
        
        # Crop & Resize Settings
        with st.expander("📐 Crop & Resize"):
            st.session_state.crop_preset = st.selectbox(
                "Crop Preset",
                list(CROP_PRESETS.keys()),
                index=list(CROP_PRESETS.keys()).index(st.session_state.crop_preset)
            )
            st.session_state.resize_percent = st.slider(
                "Resize %", 25, 200, st.session_state.resize_percent
            )
        
        # Export Settings
        st.markdown("### 💾 Export Settings")
        export_format = st.selectbox(
            "Format",
            ["PNG", "JPEG", "WEBP"],
            index=0
        )
        
        quality = 95
        if export_format in ["JPEG", "WEBP"]:
            quality = st.slider("Quality", 50, 100, 95)
    
    # Main Content Area
    tabs = st.tabs(["📤 Upload", "🔬 Process", "🎨 Results", "📊 Compare"])
    
    with tabs[0]:
        st.markdown("### Upload Your Image")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            upload_type = st.radio(
                "Choose upload method:",
                ["Upload File", "Use Demo Image"],
                horizontal=True
            )
            
            if upload_type == "Upload File":
                if st.session_state.batch_mode:
                    uploaded_files = st.file_uploader(
                        "Choose images...",
                        type=["jpg", "jpeg", "png", "bmp", "tiff"],
                        accept_multiple_files=True
                    )
                    
                    if uploaded_files:
                        st.session_state.uploaded_images = []
                        for file in uploaded_files:
                            img = Image.open(file)
                            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            st.session_state.uploaded_images.append(img_array)
                        st.success(f"✅ {len(uploaded_files)} images uploaded successfully!")
                else:
                    uploaded_file = st.file_uploader(
                        "Choose an image...",
                        type=["jpg", "jpeg", "png", "bmp", "tiff"]
                    )
                    
                    if uploaded_file:
                        img = Image.open(uploaded_file)
                        st.session_state.uploaded_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        st.success("✅ Image uploaded successfully!")
            
            else:  # Use Demo Image
                demo_choice = st.selectbox(
                    "Select a demo image:",
                    list(DEMO_IMAGE_PATHS.keys())
                )
                
                if st.button("Load Demo Image", use_container_width=True):
                    demo_path = DEMO_IMAGE_PATHS[demo_choice]
                    if os.path.exists(demo_path):
                        img = Image.open(demo_path)
                        st.session_state.uploaded_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        st.success("✅ Demo image loaded successfully!")
                    else:
                        st.error(f"Demo image not found at {demo_path}")
        
        with col2:
            if st.session_state.uploaded_image is not None:
                st.markdown("#### Preview")
                display_img = cv2.cvtColor(st.session_state.uploaded_image, cv2.COLOR_BGR2RGB)
                display_img = Image.fromarray(display_img)
                display_img.thumbnail((300, 300), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                
                # Image info
                h, w = st.session_state.uploaded_image.shape[:2]
                st.markdown(f"**Dimensions:** {w} × {h} px")
                st.markdown(f"**Aspect Ratio:** {w/h:.2f}")
    
    with tabs[1]:
        st.markdown("### Image Processing")
        
        if st.session_state.uploaded_image is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 Process Image", use_container_width=True, type="primary"):
                    with st.spinner("Processing... Please wait."):
                        # Generate mask
                        prob = predict_mask(model, st.session_state.uploaded_image, 
                                          CONFIG["device"], CONFIG["img_size"])
                        mask = postprocess_mask(prob, st.session_state.fg_thresh, 
                                              st.session_state.min_area)
                        st.session_state.processed_mask = mask
                        
                        # Apply background
                        mask_bin = (mask > 127).astype(np.uint8)
                        bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                        result_pil = apply_background(
                            st.session_state.uploaded_image, mask_bin,
                            st.session_state.extraction_mode, bg_path,
                            st.session_state.custom_color
                        )
                        
                        # Apply filters and adjustments
                        result_pil = apply_filters_and_adjustments(result_pil)
                        
                        # Apply crop
                        result_pil = crop_image(result_pil, st.session_state.crop_preset)
                        
                        # Apply resize
                        if st.session_state.resize_percent != 100:
                            orig_w, orig_h = result_pil.size
                            new_w = int(orig_w * st.session_state.resize_percent / 100)
                            new_h = int(orig_h * st.session_state.resize_percent / 100)
                            result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
                        
                        st.session_state.result_image = result_pil
                        st.session_state.current_step = 3
                        st.success("✅ Processing complete!")
            
            if st.session_state.processed_mask is not None:
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Image")
                    display_orig = cv2.cvtColor(st.session_state.uploaded_image, cv2.COLOR_BGR2RGB)
                    st.image(display_orig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Segmentation Mask")
                    st.image(st.session_state.processed_mask, use_container_width=True)
                
                # Mask Statistics
                mask_bin = (st.session_state.processed_mask > 127).astype(np.uint8)
                coverage = np.sum(mask_bin) / (mask_bin.shape[0] * mask_bin.shape[1]) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Foreground Coverage", f"{coverage:.1f}%")
                with col2:
                    st.metric("Mask Quality", "High" if coverage > 20 else "Low")
                with col3:
                    st.metric("Processing Status", "Complete")
        else:
            st.info("📤 Please upload an image first in the Upload tab.")
    
    with tabs[2]:
        st.markdown("### Final Result")
        
        if st.session_state.result_image is not None:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(st.session_state.result_image, use_container_width=True)
            
            with col2:
                st.markdown("#### 📥 Download Options")
                
                # Prepare download
                buf = BytesIO()
                if export_format == "PNG" and st.session_state.result_image.mode == "RGBA":
                    st.session_state.result_image.save(buf, format="PNG")
                elif export_format in ["JPEG", "JPG"]:
                    result_rgb = st.session_state.result_image.convert("RGB")
                    result_rgb.save(buf, format="JPEG", quality=quality)
                else:
                    result_rgb = st.session_state.result_image.convert("RGB")
                    result_rgb.save(buf, format=export_format, quality=quality)
                
                st.download_button(
                    f"⬇️ Download {export_format}",
                    buf.getvalue(),
                    f"processed.{export_format.lower()}",
                    f"image/{export_format.lower()}",
                    key="download_result",
                    use_container_width=True
                )
                
                # Result Info
                st.markdown("---")
                st.markdown("#### 📊 Image Info")
                final_w, final_h = st.session_state.result_image.size
                st.markdown(f"**Size:** {final_w} × {final_h} px")
                st.markdown(f"**Format:** {export_format}")
                if export_format != "PNG":
                    st.markdown(f"**Quality:** {quality}%")
                
                # Quick Actions
                st.markdown("---")
                st.markdown("#### ⚡ Quick Actions")
                if st.button("🔄 Process Another", use_container_width=True):
                    st.session_state.uploaded_image = None
                    st.session_state.processed_mask = None
                    st.session_state.result_image = None
                    st.session_state.current_step = 1
                    st.rerun()
        else:
            st.info("🔬 Please process an image first in the Process tab.")
    
    with tabs[3]:
        st.markdown("### Image Comparison")
        
        if st.session_state.result_image is not None and st.session_state.uploaded_image is not None:
            comparison_type = st.radio(
                "Comparison Type:",
                ["Side by Side", "Slider", "Overlay"],
                horizontal=True
            )
            
            original_img = Image.fromarray(cv2.cvtColor(st.session_state.uploaded_image, cv2.COLOR_BGR2RGB))
            result_rgb = st.session_state.result_image.convert("RGB") if st.session_state.result_image.mode == "RGBA" else st.session_state.result_image
            
            if comparison_type == "Side by Side":
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Original")
                    st.image(original_img, use_container_width=True)
                with col2:
                    st.markdown("#### Processed")
                    st.image(result_rgb, use_container_width=True)
            
            elif comparison_type == "Slider":
                try:
                    # Resize result to match original for comparison
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                    image_comparison(
                        img1=original_img,
                        img2=result_rgb,
                        label1="Original",
                        label2="Processed",
                        width=700,
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True
                    )
                except:
                    st.warning("Slider comparison not available. Showing side by side instead.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_img, caption="Original", use_container_width=True)
                    with col2:
                        st.image(result_rgb, caption="Processed", use_container_width=True)
            
            else:  # Overlay
                alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.05)
                result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                overlay = Image.blend(original_img, result_rgb, alpha)
                st.image(overlay, use_container_width=True)
            
            # Download comparison
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col2:
                if comparison_type == "Side by Side":
                    # Create side-by-side comparison image
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                    comparison = np.concatenate([np.array(original_img), np.array(result_rgb)], axis=1)
                    buf_comp = BytesIO()
                    Image.fromarray(comparison).save(buf_comp, format="PNG")
                    st.download_button("📊 Download Comparison", buf_comp.getvalue(), "comparison.png",
                                     "image/png", key="download_comp", use_container_width=True)
                
                st.session_state.current_step = 4
        else:
            st.info("🎨 Please process an image first to see comparisons.")
    
    # Batch Processing Section
    if st.session_state.batch_mode and len(st.session_state.uploaded_images) > 0:
        st.markdown("---")
        st.markdown("### 🔄 Batch Processing")
        st.info(f"📁 {len(st.session_state.uploaded_images)} images ready for processing")
        
        if st.button("▶️ Process All Images", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            batch_results = []
            
            for idx, img_array in enumerate(st.session_state.uploaded_images):
                # Process each image
                prob = predict_mask(model, img_array, CONFIG["device"], CONFIG["img_size"])
                mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                mask_bin = (mask > 127).astype(np.uint8)
                
                # Apply background and filters
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(img_array, mask_bin, st.session_state.extraction_mode, 
                                             bg_path, st.session_state.custom_color)
                result_pil = apply_filters_and_adjustments(result_pil)
                result_pil = crop_image(result_pil, st.session_state.crop_preset)
                
                # Resize if needed
                if st.session_state.resize_percent != 100:
                    orig_w, orig_h = result_pil.size
                    new_w = int(orig_w * st.session_state.resize_percent / 100)
                    new_h = int(orig_h * st.session_state.resize_percent / 100)
                    result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
                
                batch_results.append(result_pil)
                progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
            
            st.success("✅ All images processed successfully!")
            
            # Display results preview
            st.markdown("#### 📸 Results Preview")
            cols = st.columns(3)
            for idx, result in enumerate(batch_results):
                with cols[idx % 3]:
                    display_result = result.copy()
                    display_result.thumbnail((300, 250), Image.LANCZOS)
                    st.image(display_result, caption=f"Image {idx + 1}", use_container_width=True)
            
            # Download options
            st.markdown("#### 📥 Download Batch Results")
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
                                     key=f"batch_dl_{idx}", use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-professional">
        <h3>OneView</h3>
        <p>Professional AI-Powered Image Processing Solution</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.7;">Developed by Manusha</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
