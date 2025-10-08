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
    "Demo Image 1": "demo/orginal.jpg",
    "Demo Image 2": "demo/extracted.jpg"
}

BG_IMAGE_PATHS = {
    "Bg1": "backgrounds/bg1.jpg",
    "Bg2": "backgrounds/bg2.jpg",
    "Bg3": "backgrounds/bg3.jpg",
    "Bg4": "backgrounds/bg4.jpg",
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
# CUSTOM CSS - FIXED SIDEBAR TEXT VISIBILITY
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.main-header {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(249, 250, 251, 0.98));
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.5);
}

.main-header h1 {
    margin: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    letter-spacing: -1px;
}

.main-header p {
    margin: 0.8rem 0 0 0;
    color: #4a5568;
    font-size: 1.1rem;
    font-weight: 500;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.95);
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
}

.stTabs [data-baseweb="tab"] {
    height: 60px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0 2rem;
    border: none;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    box-shadow: 0 5px 25px rgba(118, 75, 162, 0.5);
    transform: scale(1.05);
}

/* FIXED: Sidebar text visibility */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%) !important;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] textarea {
    color: #1a202c !important;
    background-color: #f7fafc !important;
    border: 1px solid #cbd5e0 !important;
}

section[data-testid="stSidebar"] .stNumberInput input::placeholder,
section[data-testid="stSidebar"] input[type="text"]::placeholder {
    color: #718096 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    filter: brightness(1.1);
}

section[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: 3px dashed rgba(255, 255, 255, 0.6) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
}

.step-indicator {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(249, 250, 251, 0.98));
    padding: 1.5rem;
    border-radius: 15px;
    margin: 2rem 0;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.5);
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
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}

.step-number.active {
    box-shadow: 0 0 30px rgba(102, 126, 234, 0.8);
    transform: scale(1.15);
}

.step-label {
    color: #4a5568;
    font-weight: 600;
    font-size: 0.9rem;
}

.step-connector {
    height: 3px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    flex: 1;
    margin: 0 1rem;
}

.footer-professional {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(249, 250, 251, 0.98));
    border-radius: 15px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.12);
    margin-top: 3rem;
    border: 1px solid rgba(255, 255, 255, 0.5);
}

.footer-professional h3 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Poppins', sans-serif;
    margin: 0;
    font-size: 1.5rem;
}

.footer-professional p {
    color: #4a5568 !important;
    margin-top: 0.5rem;
    font-weight: 500;
    background: transparent !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
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
    arr = np.array(img)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                              [0.349, 0.686, 0.168],
                              [0.272, 0.534, 0.131]])
    sepia_arr = cv2.transform(arr, sepia_filter)
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

    elif mode in ["Bg1", "Bg2", "Bg3", "Bg4"] and bg_path:
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
# HELPER FUNCTIONS FOR IMAGE ENCODING/DECODING
# ============================================================================

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img_pil = Image.fromarray(img_array)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(img_str):
    """Convert base64 string to numpy array"""
    img_data = base64.b64decode(img_str)
    img_pil = Image.open(BytesIO(img_data))
    return np.array(img_pil)

# ============================================================================
# PROJECT MANAGEMENT - REWRITTEN TO SAVE IMAGES
# ============================================================================

def save_project():
    """Save project with images encoded as base64"""
    try:
        if st.session_state.get('original_image') is None:
            return False
        
        project_name = st.session_state.get('current_project_name', '').strip()
        if not project_name:
            return False
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Encode images as base64
        original_img_b64 = image_to_base64(st.session_state.original_image)
        current_img_b64 = image_to_base64(st.session_state.current_image) if st.session_state.current_image is not None else None
        
        # Encode mask if it exists
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
    """Load project and decode images from base64"""
    try:
        for proj in st.session_state.get('saved_projects', []):
            if proj.get('name') == project_name:
                # Decode images
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
                
                # Load settings
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
    """Delete a project"""
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
        page_title="One View",
        page_icon="🔷",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Load model
    model = get_model()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>OneView</h1>
        <p>Create pro-level cutouts by removing distractions and letting your subject shine in every image.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        st.session_state.fg_thresh = st.slider("Detection Threshold", 0.0, 1.0, st.session_state.fg_thresh, 0.01)
        st.session_state.min_area = st.number_input("Minimum Area (px)", 1, 5000, st.session_state.min_area, 50)
        
        st.markdown("---")
        st.markdown("### 📦 Export Settings")
        export_format = st.selectbox("Format", ["PNG", "JPEG", "JPG", "WEBP"])
        quality = st.slider("Quality", 1, 100, 95) if export_format in ["JPEG", "JPG"] else 95
        
        st.markdown("---")
        st.markdown("### 🔄 Processing Mode")
        st.session_state.batch_mode = st.checkbox("Batch Processing", value=st.session_state.batch_mode)
        
        st.markdown("---")
        st.markdown("### 👤 Profile & Projects")
        
        if st.button("📁 Manage Projects", use_container_width=True):
            st.session_state.show_profile_section = not st.session_state.show_profile_section
        
        if st.session_state.get('show_profile_section', False):
            st.markdown("---")
            project_name = st.text_input("📝 Project Name", value=st.session_state.get('current_project_name', ''))
            
            if project_name != st.session_state.get('current_project_name', ''):
                st.session_state.current_project_name = project_name
            
            if st.button("💾 Save", use_container_width=True):
                if project_name.strip() and st.session_state.get('original_image') is not None:
                    if save_project():
                        st.success(f"✅ '{project_name}' saved!")
                        st.rerun()
                else:
                    st.warning("⚠️ Enter project name and upload an image first")
            
            st.markdown("---")
            
            if st.session_state.get('saved_projects', []):
                st.markdown("**📚 Saved Projects**")
                for idx, proj in enumerate(st.session_state.saved_projects):
                    with st.container():
                        st.markdown(f"**{idx + 1}. {proj['name']}**")
                        st.caption(f"🕒 {proj['timestamp']}")
                        
                        # Show thumbnail preview if available
                        if proj.get('original_image'):
                            try:
                                thumb_img = base64_to_image(proj['original_image'])
                                thumb_pil = Image.fromarray(thumb_img)
                                thumb_pil.thumbnail((150, 150), Image.LANCZOS)
                                st.image(thumb_pil, use_container_width=True)
                            except:
                                st.caption("📷 Image preview unavailable")
                        
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

    # Step Indicator
    st.markdown(f"""
    <div class="step-indicator">
        <div class="step">
            <div class="step-number {'active' if st.session_state.current_step == 1 else ''}">1</div>
            <div class="step-label">Upload</div>
        </div>
        <div class="step-connector"></div>
        <div class="step">
            <div class="step-number {'active' if st.session_state.current_step == 2 else ''}">2</div>
            <div class="step-label">Extract</div>
        </div>
        <div class="step-connector"></div>
        <div class="step">
            <div class="step-number {'active' if st.session_state.current_step == 3 else ''}">3</div>
            <div class="step-label">Customize</div>
        </div>
        <div class="step-connector"></div>
        <div class="step">
            <div class="step-number {'active' if st.session_state.current_step == 4 else ''}">4</div>
            <div class="step-label">Export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Demo Section
    st.markdown("### 📸 See What's Possible")
    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        st.markdown("#### Original Image")
        if os.path.exists(DEMO_IMAGE_PATHS["Demo Image 1"]):
            demo_orig = Image.open(DEMO_IMAGE_PATHS["Demo Image 1"])
            demo_orig.thumbnail((400, 300), Image.LANCZOS)
            st.image(demo_orig, use_container_width=True, caption="Before")
        else:
            st.info("Demo image not available")

    with demo_col2:
        st.markdown("#### Extracted Subject")
        if os.path.exists(DEMO_IMAGE_PATHS["Demo Image 2"]):
            demo_extracted = Image.open(DEMO_IMAGE_PATHS["Demo Image 2"])
            demo_extracted.thumbnail((400, 300), Image.LANCZOS)
            st.image(demo_extracted, use_container_width=True, caption="After")
        else:
            st.info("Demo image not available")

    st.markdown("---")

    # Upload Section
    st.markdown("### 📤 Upload Your Images")
    uploaded_files = st.file_uploader(
        "Drop your images here",
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
            st.success(f"✅ {len(uploaded_files)} images uploaded!")
            st.session_state.current_step = 2
        else:
            img = Image.open(uploaded_files).convert("RGB")
            st.session_state.original_image = np.array(img)
            st.session_state.current_image = np.array(img)
            st.session_state.current_step = 2

            with st.spinner("🔍 Detecting subject..."):
                prob = predict_mask(model, st.session_state.original_image, CONFIG["device"], CONFIG["img_size"])
                st.session_state.prob_map = prob
                mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                st.session_state.mask = (mask > 127).astype(np.uint8)

    # Main Editor Tabs
    if st.session_state.current_image is not None or st.session_state.batch_mode:
        tabs = st.tabs(["🎨 Background", "🎛️ Filters & Effects", "✂️ Crop & Resize", "🖼️ Final Preview"])

        # Tab 1: Background
        with tabs[0]:
            st.markdown("### Background Options")
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("#### Choose Background")
                
                if st.button("✨ Transparent", key="bg_transparent", use_container_width=True):
                    st.session_state.extraction_mode = "Transparent"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("⚪ White", key="bg_white", use_container_width=True):
                    st.session_state.extraction_mode = "White"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("⚫ Black", key="bg_black", use_container_width=True):
                    st.session_state.extraction_mode = "Black"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("💫 Blur", key="bg_blur", use_container_width=True):
                    st.session_state.extraction_mode = "Blur"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("🎨 Custom Color", key="bg_custom_color", use_container_width=True):
                    st.session_state.extraction_mode = "Custom Color"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("🖼️ Custom Image", key="bg_custom_image", use_container_width=True):
                    st.session_state.extraction_mode = "Custom Image"
                    st.session_state.current_step = 3
                    st.rerun()

                if st.button("📚 Preset Backgrounds", key="toggle_presets", use_container_width=True):
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
                    st.session_state.custom_color = st.color_picker("Pick a color", st.session_state.custom_color)

                if st.session_state.extraction_mode == "Custom Image":
                    st.markdown("---")
                    custom_bg = st.file_uploader("Upload background", type=["jpg", "jpeg", "png"], key="custom_bg_upload")
                    if custom_bg:
                        st.session_state.selected_bg = Image.open(custom_bg)

            with col_right:
                if st.session_state.current_image is not None and st.session_state.mask is not None:
                    st.markdown("#### Preview")
                    bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                    result_pil = apply_background(st.session_state.current_image, st.session_state.mask, 
                                                 st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                    
                    display_img = result_pil.copy()
                    display_img.thumbnail((700, 500), Image.LANCZOS)
                    st.image(display_img, use_container_width=True)
                    
                    st.markdown("---")
                    get_download_button(result_pil, export_format, quality, "⬇️ Download Preview", 
                                      f"background_preview.{export_format.lower()}", "download_bg")

        # Tab 2: Filters
        with tabs[1]:
            st.markdown("### Filters & Effects")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Filter Presets")
                st.session_state.filter_type = st.selectbox("Select Filter", list(FILTERS.keys()))
                
                st.markdown("---")
                st.markdown("#### Adjustments")
                st.session_state.brightness = st.slider("☀️ Brightness", 0.0, 2.0, st.session_state.brightness, 0.1)
                st.session_state.contrast = st.slider("🔆 Contrast", 0.0, 2.0, st.session_state.contrast, 0.1)
                st.session_state.saturation = st.slider("🌈 Saturation", 0.0, 2.0, st.session_state.saturation, 0.1)

            with col2:
                if st.session_state.current_image is not None:
                    st.markdown("#### Preview with Effects")
                    bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                    result_pil = apply_background(st.session_state.current_image, st.session_state.mask,
                                                 st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                    result_pil = apply_filters_and_adjustments(result_pil)
                    
                    display_img = result_pil.copy()
                    display_img.thumbnail((700, 500), Image.LANCZOS)
                    st.image(display_img, use_container_width=True)
                    
                    st.markdown("---")
                    get_download_button(result_pil, export_format, quality, "⬇️ Download Filtered",
                                      f"filtered_image.{export_format.lower()}", "download_filter")

        # Tab 3: Crop & Resize
        with tabs[2]:
            st.markdown("### Crop & Resize")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Crop Presets")
                st.session_state.crop_preset = st.selectbox("Select Preset", list(CROP_PRESETS.keys()))
                
                st.markdown("---")
                st.markdown("#### Resize")
                
                if st.session_state.current_image is not None:
                    orig_h, orig_w = st.session_state.current_image.shape[:2]
                    st.caption(f"Original: {orig_w} × {orig_h} px")
                    
                    st.session_state.resize_percent = st.slider("Scale (%)", 10, 200, st.session_state.resize_percent)
                    new_w = int(orig_w * st.session_state.resize_percent / 100)
                    new_h = int(orig_h * st.session_state.resize_percent / 100)
                    st.caption(f"New: {new_w} × {new_h} px")

            with col2:
                if st.session_state.current_image is not None:
                    st.markdown("#### Preview")
                    bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                    result_pil = apply_background(st.session_state.current_image, st.session_state.mask,
                                                 st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                    result_pil = apply_filters_and_adjustments(result_pil)
                    result_pil = crop_image(result_pil, st.session_state.crop_preset)
                    
                    if st.session_state.resize_percent != 100:
                        result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
                    
                    display_img = result_pil.copy()
                    display_img.thumbnail((700, 500), Image.LANCZOS)
                    st.image(display_img, use_container_width=True)
                    
                    st.markdown("---")
                    get_download_button(result_pil, export_format, quality, "⬇️ Download Cropped",
                                      f"cropped_image.{export_format.lower()}", "download_crop")

        # Tab 4: Final Preview - REWRITTEN TO SHOW IMAGE IMMEDIATELY
        with tabs[3]:
            st.markdown("### Final Preview & Export")

            if st.session_state.current_image is not None and st.session_state.mask is not None:
                # Generate final result
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

                st.markdown("<h3 style='text-align:center;'>🔄 Interactive View</h3>", unsafe_allow_html=True)
                
                _, view_container, _ = st.columns([1, 5, 1])
                
                with view_container:
                    view_col, adjust_col = st.columns([2, 1])
                    
                    with adjust_col:
                        st.markdown("#### 🎛️ Adjust View")
                        blend_value = st.slider("Blend", 0.0, 1.0, st.session_state.blend_slider, 0.05, key="blend_slider_control")
                        zoom_value = st.slider("Zoom (%)", 50, 200, st.session_state.zoom_percentage, 5, key="zoom_slider_control")
                        
                        # Update session state
                        st.session_state.blend_slider = blend_value
                        st.session_state.zoom_percentage = zoom_value
                    
                    with view_col:
                        # Prepare images for comparison
                        original_img = Image.fromarray(st.session_state.original_image)
                        
                        # Handle RGBA mode
                        if result_pil.mode == 'RGBA':
                            result_rgb = Image.new('RGB', result_pil.size, (0, 0, 0))
                            result_rgb.paste(result_pil, (0, 0), result_pil)
                        else:
                            result_rgb = result_pil.convert('RGB')
                        
                        # Resize if needed to match dimensions
                        if original_img.size != result_rgb.size:
                            result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                        
                        # Apply zoom
                        scale = zoom_value / 100.0
                        new_w = max(1, int(original_img.size[0] * scale))
                        new_h = max(1, int(original_img.size[1] * scale))
                        
                        zoomed_orig = original_img.resize((new_w, new_h), Image.LANCZOS)
                        zoomed_result = result_rgb.resize((new_w, new_h), Image.LANCZOS)
                        
                        orig_rgb = zoomed_orig.convert('RGB')
                        
                        # Create blended image
                        blended = Image.blend(orig_rgb, zoomed_result, float(blend_value))
                        
                        # Display comparison
                        image_comparison(
                            img1=orig_rgb, 
                            img2=blended, 
                            label1="Original",
                            label2=f"Blended ({int(blend_value*100)}%)"
                        )

                st.markdown("---")
                st.markdown("### 📥 Export Options")
                
                export_col1, export_col2, export_col3 = st.columns(3)

                with export_col1:
                    get_download_button(result_pil, export_format, quality, "⬇️ Download Final",
                                      f"visionextract_result.{export_format.lower()}", "download_final")

                with export_col2:
                    buf_orig = BytesIO()
                    Image.fromarray(st.session_state.original_image).save(buf_orig, format="PNG")
                    st.download_button("📥 Download Original", buf_orig.getvalue(), "original.png",
                                     "image/png", key="download_orig", use_container_width=True)

                with export_col3:
                    original_img = Image.fromarray(st.session_state.original_image)
                    result_rgb = result_pil.convert("RGB")
                    if original_img.size != result_rgb.size:
                        result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                    
                    comparison = np.concatenate([np.array(original_img), np.array(result_rgb)], axis=1)
                    buf_comp = BytesIO()
                    Image.fromarray(comparison).save(buf_comp, format="PNG")
                    st.download_button("📊 Download Comparison", buf_comp.getvalue(), "comparison.png",
                                     "image/png", key="download_comp", use_container_width=True)

                st.session_state.current_step = 4

    # Batch Processing
    if st.session_state.batch_mode and len(st.session_state.uploaded_images) > 0:
        st.markdown("---")
        st.markdown("### 🔄 Batch Processing")
        st.info(f"Processing {len(st.session_state.uploaded_images)} images")

        if st.button("▶️ Process All Images", use_container_width=True):
            progress_bar = st.progress(0)
            batch_results = []

            for idx, img_array in enumerate(st.session_state.uploaded_images):
                prob = predict_mask(model, img_array, CONFIG["device"], CONFIG["img_size"])
                mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                mask_bin = (mask > 127).astype(np.uint8)

                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(img_array, mask_bin, st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
                result_pil = apply_filters_and_adjustments(result_pil)
                result_pil = crop_image(result_pil, st.session_state.crop_preset)

                if st.session_state.resize_percent != 100:
                    orig_w, orig_h = result_pil.size
                    new_w = int(orig_w * st.session_state.resize_percent / 100)
                    new_h = int(orig_h * st.session_state.resize_percent / 100)
                    result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)

                batch_results.append(result_pil)
                progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))

            st.success("✅ All images processed!")

            st.markdown("#### 📸 Results Preview")
            cols = st.columns(3)
            for idx, result in enumerate(batch_results):
                with cols[idx % 3]:
                    display_result = result.copy()
                    display_result.thumbnail((300, 250), Image.LANCZOS)
                    st.image(display_result, caption=f"Image {idx + 1}", use_container_width=True)

            st.markdown("#### 📥 Download Results")
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

                    st.download_button(f"⬇️ {idx + 1}", buf.getvalue(), f"batch_{idx + 1}.{export_format.lower()}",
                                     f"image/{export_format.lower()}", key=f"batch_dl_{idx}", use_container_width=True)

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
