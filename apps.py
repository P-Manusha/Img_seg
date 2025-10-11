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
# PREMIUM CULT CLASSIC CSS - ULTRA PROFESSIONAL DESIGN
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

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

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    background-attachment: fixed;
}

/* Premium Hero Header */
.hero-header {
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.95), rgba(10, 14, 39, 0.95));
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    padding: 3.5rem 3rem;
    border-radius: 0 0 40px 40px;
    text-align: center;
    box-shadow: 
        0 20px 60px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(212, 175, 55, 0.2),
        0 0 100px rgba(212, 175, 55, 0.1);
    margin-bottom: 3rem;
    border: 1px solid var(--border-gold);
    border-top: none;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(212, 175, 55, 0.1), 
        transparent);
    animation: shimmer 8s infinite;
}

@keyframes shimmer {
    0%, 100% { transform: translateX(0); }
    50% { transform: translateX(50%); }
}

.hero-header h1 {
    margin: 0;
    font-size: 4.5rem;
    font-weight: 900;
    font-family: 'Playfair Display', serif;
    background: linear-gradient(135deg, #d4af37 0%, #f4d03f 30%, #d4af37 60%, #c5a028 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 30px rgba(212, 175, 55, 0.5);
    animation: titleGlow 3s ease-in-out infinite;
}

@keyframes titleGlow {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.2); }
}

.hero-subtitle {
    margin: 1.5rem 0 0 0;
    color: var(--text-muted);
    font-size: 1.3rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

.hero-tagline {
    margin: 0.8rem 0 0 0;
    color: var(--primary-gold);
    font-size: 1rem;
    font-weight: 500;
    font-style: italic;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

/* Premium Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1329 0%, #1a1f3a 100%);
    border-right: 2px solid var(--border-gold);
    box-shadow: 5px 0 30px rgba(212, 175, 55, 0.1);
}

section[data-testid="stSidebar"] * {
    color: var(--text-light) !important;
}

.sidebar-logo {
    text-align: center;
    padding: 2rem 1rem;
    border-bottom: 1px solid var(--border-gold);
    margin-bottom: 1.5rem;
}

.sidebar-logo h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
}

/* Luxury Control Panels */
.luxury-panel {
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.8), rgba(26, 31, 58, 0.8));
    backdrop-filter: blur(20px);
    padding: 1.8rem;
    border-radius: 20px;
    border: 1px solid var(--border-gold);
    margin-bottom: 1.5rem;
    box-shadow: 
        0 10px 40px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(212, 175, 55, 0.1);
    position: relative;
    overflow: hidden;
}

.luxury-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, 
        transparent, 
        var(--primary-gold), 
        transparent);
    opacity: 0.5;
}

.panel-title {
    color: var(--primary-gold) !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
    margin-bottom: 1.5rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.panel-title::before {
    content: '◆';
    color: var(--primary-gold);
    font-size: 0.8rem;
}

/* Premium Workflow Steps */
.workflow-container {
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.9), rgba(26, 31, 58, 0.9));
    backdrop-filter: blur(20px);
    padding: 2.5rem;
    border-radius: 30px;
    margin: 2.5rem 0;
    border: 2px solid var(--border-gold);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.workflow-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 2rem;
    position: relative;
    z-index: 1;
}

.workflow-step {
    text-align: center;
    padding: 2rem 1.5rem;
    background: rgba(10, 14, 39, 0.6);
    border-radius: 20px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.workflow-step::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--primary-gold), transparent);
    opacity: 0;
    transition: opacity 0.4s;
}

.workflow-step:hover::before,
.workflow-step.active::before {
    opacity: 1;
}

.workflow-step:hover {
    transform: translateY(-10px) scale(1.05);
    border-color: var(--primary-gold);
    box-shadow: 0 20px 40px rgba(212, 175, 55, 0.3);
}

.workflow-step.active {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(244, 208, 63, 0.1));
    border-color: var(--primary-gold);
    box-shadow: 0 0 40px rgba(212, 175, 55, 0.4);
}

.step-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    filter: grayscale(100%) brightness(0.7);
    transition: all 0.4s;
}

.workflow-step:hover .step-icon,
.workflow-step.active .step-icon {
    filter: grayscale(0%) brightness(1.2);
    transform: scale(1.2) rotate(5deg);
}

.step-number {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: rgba(212, 175, 55, 0.2);
    border: 1px solid var(--primary-gold);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--primary-gold);
    font-family: 'Space Mono', monospace;
}

.step-label {
    color: var(--text-light);
    font-weight: 600;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.5rem;
}

.step-desc {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 0.5rem;
    line-height: 1.4;
}

/* Premium Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-gold) 0%, #c5a028 100%);
    color: var(--primary-dark);
    border: 1px solid rgba(212, 175, 55, 0.5);
    padding: 0.9rem 2rem;
    font-weight: 700;
    border-radius: 15px;
    font-size: 0.95rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 
        0 8px 25px rgba(212, 175, 55, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.stButton > button:hover::before {
    width: 300px;
    height: 300px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #f4d03f 0%, var(--primary-gold) 100%);
    box-shadow: 
        0 12px 35px rgba(212, 175, 55, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.4);
    transform: translateY(-3px);
}

/* Premium File Uploader */
section[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.6), rgba(26, 31, 58, 0.6)) !important;
    backdrop-filter: blur(15px) !important;
    border: 3px dashed var(--border-gold) !important;
    border-radius: 25px !important;
    padding: 4rem 2rem !important;
    transition: all 0.4s ease !important;
    position: relative !important;
}

section[data-testid="stFileUploadDropzone"]::before {
    content: '✦';
    position: absolute;
    top: 2rem;
    left: 50%;
    transform: translateX(-50%);
    font-size: 3rem;
    color: var(--primary-gold);
    opacity: 0.3;
}

section[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--primary-gold) !important;
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.8), rgba(26, 31, 58, 0.8)) !important;
    box-shadow: 0 0 50px rgba(212, 175, 55, 0.3) !important;
}

/* Premium Image Preview Cards */
.premium-preview {
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.9), rgba(26, 31, 58, 0.9));
    backdrop-filter: blur(20px);
    padding: 2rem;
    border-radius: 25px;
    border: 2px solid var(--border-gold);
    box-shadow: 
        0 15px 50px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(212, 175, 55, 0.1);
    position: relative;
    overflow: hidden;
}

.premium-preview::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, 
        var(--primary-gold), 
        transparent, 
        var(--primary-gold));
    z-index: -1;
    opacity: 0;
    transition: opacity 0.4s;
}

.premium-preview:hover::before {
    opacity: 0.3;
}

.preview-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--primary-gold);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.preview-title::before {
    content: '';
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, var(--primary-gold), transparent);
}

/* Premium Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.8), rgba(26, 31, 58, 0.8));
    backdrop-filter: blur(20px);
    padding: 1.5rem;
    border-radius: 25px;
    border: 2px solid var(--border-gold);
    margin-bottom: 2.5rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}

.stTabs [data-baseweb="tab"] {
    height: auto;
    background: rgba(10, 14, 39, 0.6);
    color: var(--text-light);
    border-radius: 15px;
    font-weight: 600;
    font-size: 1rem;
    padding: 1rem 2rem;
    border: 1px solid rgba(212, 175, 55, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(212, 175, 55, 0.1);
    border-color: var(--primary-gold);
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(212, 175, 55, 0.3);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(244, 208, 63, 0.2));
    color: var(--primary-gold);
    border-color: var(--primary-gold);
    box-shadow: 0 0 30px rgba(212, 175, 55, 0.4);
}

/* Premium Sliders */
.stSlider {
    padding: 1rem 0;
}

.stSlider > div > div > div > div {
    background: var(--primary-gold) !important;
}

.stSlider > div > div > div {
    background: rgba(212, 175, 55, 0.2) !important;
}

/* Premium Select Boxes */
.stSelectbox > div > div {
    background: rgba(21, 25, 50, 0.8) !important;
    border: 1px solid var(--border-gold) !important;
    border-radius: 12px !important;
    color: var(--text-light) !important;
}

/* Premium Number Inputs */
.stNumberInput > div > div > input {
    background: rgba(21, 25, 50, 0.8) !important;
    border: 1px solid var(--border-gold) !important;
    border-radius: 12px !important;
    color: var(--text-light) !important;
}

/* Premium Color Picker */
.stColorPicker > div > div {
    background: rgba(21, 25, 50, 0.8) !important;
    border: 1px solid var(--border-gold) !important;
    border-radius: 12px !important;
}

/* Premium Expanders */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.8), rgba(26, 31, 58, 0.8));
    border: 1px solid var(--border-gold);
    border-radius: 15px;
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: var(--text-light) !important;
    transition: all 0.3s;
}

.streamlit-expanderHeader:hover {
    border-color: var(--primary-gold);
    box-shadow: 0 5px 20px rgba(212, 175, 55, 0.2);
}

/* Premium Radio Buttons */
.stRadio > div {
    gap: 1.5rem;
    flex-direction: row;
}

.stRadio [role="radiogroup"] {
    gap: 1.5rem;
}

/* Premium Footer */
.premium-footer {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, rgba(21, 25, 50, 0.95), rgba(26, 31, 58, 0.95));
    backdrop-filter: blur(20px);
    border-radius: 30px 30px 0 0;
    border: 2px solid var(--border-gold);
    border-bottom: none;
    margin-top: 4rem;
    box-shadow: 0 -20px 60px rgba(0, 0, 0, 0.4);
}

.premium-footer h2 {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 1rem 0;
}

.premium-footer p {
    color: var(--text-muted);
    font-size: 1rem;
    margin: 0.5rem 0;
}

.premium-footer .credits {
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-gold);
    color: var(--text-muted);
    font-size: 0.9rem;
}

.premium-footer .credits strong {
    color: var(--primary-gold);
    font-weight: 600;
}

/* Premium Scrollbar */
::-webkit-scrollbar {
    width: 14px;
}

::-webkit-scrollbar-track {
    background: var(--primary-dark);
    border-left: 1px solid var(--border-gold);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--primary-gold) 0%, #c5a028 100%);
    border-radius: 10px;
    border: 2px solid var(--primary-dark);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #f4d03f 0%, var(--primary-gold) 100%);
}

/* Success/Info/Warning Boxes */
.luxury-alert {
    padding: 1.5rem 2rem;
    border-radius: 15px;
    border-left: 4px solid;
    margin: 1.5rem 0;
    backdrop-filter: blur(10px);
    font-weight: 500;
}

.luxury-alert.success {
    background: rgba(46, 204, 113, 0.1);
    border-color: #2ecc71;
    color: #2ecc71;
}

.luxury-alert.info {
    background: rgba(52, 152, 219, 0.1);
    border-color: #3498db;
    color: #3498db;
}

.luxury-alert.warning {
    background: rgba(241, 196, 15, 0.1);
    border-color: #f1c40f;
    color: #f1c40f;
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* General Text Styling */
p, span, div, label {
    color: var(--text-light);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary-gold) !important;
    font-family: 'Playfair Display', serif !important;
}

/* Animation Classes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.8s ease-out;
}

/* Premium Toggle Buttons */
.toggle-group {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.toggle-btn {
    flex: 1;
    min-width: 150px;
    padding: 1rem 1.5rem;
    background: rgba(21, 25, 50, 0.8);
    border: 1px solid rgba(212, 175, 55, 0.3);
    border-radius: 12px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.toggle-btn:hover {
    background: rgba(212, 175, 55, 0.2);
    border-color: var(--primary-gold);
    transform: translateY(-2px);
}

.toggle-btn.active {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(244, 208, 63, 0.2));
    border-color: var(--primary-gold);
    color: var(--primary-gold);
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
        'current_step': 0,
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
        'show_advanced': False,
        'edge_refinement': 0,
        'feather_amount': 0,
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
                st.session_state.current_step = 1
                
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
        page_title="OneView Pro - Premium Image Editor",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    model = get_model()

    # Premium Hero Header
    st.markdown("""
    <div class="hero-header fade-in-up">
        <h1>ONEVIEW PRO</h1>
        <p class="hero-subtitle">Premium Image Studio</p>
        <p class="hero-tagline">AI-Powered Excellence in Every Pixel</p>
    </div>
    """, unsafe_allow_html=True)

    # Premium Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2>✦ ONEVIEW ✦</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">🎯 AI Processing</p>', unsafe_allow_html=True)
        st.session_state.fg_thresh = st.slider(
            "Detection Sensitivity", 
            0.0, 1.0, 
            st.session_state.fg_thresh, 
            0.01,
            help="Higher values = stricter detection"
        )
        st.session_state.min_area = st.number_input(
            "Minimum Object Size (px)", 
            1, 5000, 
            st.session_state.min_area, 
            50,
            help="Filters out small artifacts"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">⚙️ Advanced Settings</p>', unsafe_allow_html=True)
        
        if st.checkbox("🔧 Show Advanced Options", value=st.session_state.show_advanced):
            st.session_state.show_advanced = True
            st.session_state.edge_refinement = st.slider("Edge Refinement", 0, 10, st.session_state.edge_refinement)
            st.session_state.feather_amount = st.slider("Feather Edges", 0, 20, st.session_state.feather_amount)
        else:
            st.session_state.show_advanced = False
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">📦 Export Configuration</p>', unsafe_allow_html=True)
        export_format = st.selectbox("Output Format", ["PNG", "JPEG", "JPG", "WEBP"], help="PNG supports transparency")
        quality = 95
        if export_format in ["JPEG", "JPG", "WEBP"]:
            quality = st.slider("Image Quality", 1, 100, 95, help="Higher = better quality, larger file")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">💼 Project Management</p>', unsafe_allow_html=True)
        
        project_name = st.text_input(
            "Project Name", 
            value=st.session_state.get('current_project_name', ''),
            placeholder="My Premium Edit..."
        )
        
        if project_name != st.session_state.get('current_project_name', ''):
            st.session_state.current_project_name = project_name
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                if project_name.strip() and st.session_state.get('original_image') is not None:
                    if save_project():
                        st.success("✓ Saved!")
                        st.rerun()
                else:
                    st.warning("⚠ Name required")
        
        with col2:
            if st.button("🔄 New", use_container_width=True):
                for key in ['original_image', 'current_image', 'mask', 'current_project_name']:
                    if key in st.session_state:
                        st.session_state[key] = None if key != 'current_project_name' else ""
                st.session_state.current_step = 0
                st.rerun()
        
        if st.session_state.get('saved_projects', []):
            st.markdown("---")
            st.markdown("**📚 Saved Projects**")
            for idx, proj in enumerate(st.session_state.saved_projects):
                with st.expander(f"✦ {proj['name']}", expanded=False):
                    st.caption(f"🕒 {proj['timestamp']}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📂 Load", key=f"load_{idx}", use_container_width=True):
                            if load_project(proj['name']):
                                st.success("✓ Loaded!")
                                st.rerun()
                    with col_b:
                        if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True):
                            delete_project(proj['name'])
                            st.success("✓ Deleted!")
                            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Premium Workflow Steps
    steps_data = [
        {"icon": "📤", "label": "Upload", "desc": "Select your image", "number": "01"},
        {"icon": "🎯", "label": "Detect", "desc": "AI processing", "number": "02"},
        {"icon": "🎨", "label": "Customize", "desc": "Apply effects", "number": "03"},
        {"icon": "💎", "label": "Export", "desc": "Download result", "number": "04"}
    ]
    
    steps_html = '<div class="workflow-container fade-in-up"><div class="workflow-steps">'
    for idx, step in enumerate(steps_data):
        active_class = "active" if st.session_state.current_step == idx else ""
        steps_html += f'''
        <div class="workflow-step {active_class}">
            <div class="step-number">{step["number"]}</div>
            <div class="step-icon">{step["icon"]}</div>
            <div class="step-label">{step["label"]}</div>
            <div class="step-desc">{step["desc"]}</div>
        </div>
        '''
    steps_html += '</div></div>'
    st.markdown(steps_html, unsafe_allow_html=True)

    # Upload Section
    st.markdown('<div class="luxury-panel fade-in-up">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">📤 Upload Your Masterpiece</p>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drag and drop your image here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded_files:
        img = Image.open(uploaded_files).convert("RGB")
        st.session_state.original_image = np.array(img)
        st.session_state.current_image = np.array(img)
        st.session_state.current_step = 1

        with st.spinner("🔮 AI is analyzing your image..."):
            prob = predict_mask(model, st.session_state.original_image, CONFIG["device"], CONFIG["img_size"])
            st.session_state.prob_map = prob
            mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
            st.session_state.mask = (mask > 127).astype(np.uint8)
        
        st.markdown('<div class="luxury-alert success">✓ Subject detected with precision!</div>', unsafe_allow_html=True)
        st.session_state.current_step = 2

    st.markdown('</div>', unsafe_allow_html=True)

    # Main Editing Interface
    if st.session_state.current_image is not None and st.session_state.mask is not None:
        
        tabs = st.tabs(["🎨 Background Studio", "✨ Enhancement Lab", "✂️ Transform", "🔍 Compare View", "💎 Export Center"])

        # Background Studio Tab
        with tabs[0]:
            col_options, col_preview = st.columns([1, 2])
            
            with col_options:
                st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
                st.markdown('<p class="panel-title">Background Options</p>', unsafe_allow_html=True)
                
                bg_modes = [
                    ("✨ Transparent", "Transparent"),
                    ("⚪ Pure White", "White"),
                    ("⚫ Deep Black", "Black"),
                    ("💫 Artistic Blur", "Blur"),
                    ("🎨 Custom Color", "Custom Color"),
                    ("🖼️ Custom Image", "Custom Image")
                ]
                
                for label, mode in bg_modes:
                    if st.button(label, key=f"bg_{mode}", use_container_width=True):
                        st.session_state.extraction_mode = mode
                        st.rerun()
                
                st.markdown("---")
                
                if st.button("📚 Preset Backgrounds", key="toggle_bg", use_container_width=True):
                    st.session_state.show_bg_presets = not st.session_state.show_bg_presets
                    st.rerun()
                
                if st.session_state.show_bg_presets:
                    for bg_key, bg_path in BG_IMAGE_PATHS.items():
                        if os.path.exists(bg_path):
                            if st.button(f"✦ {bg_key}", key=f"preset_{bg_key}", use_container_width=True):
                                st.session_state.extraction_mode = bg_key
                                st.rerun()
                
                if st.session_state.extraction_mode == "Custom Color":
                    st.markdown("---")
                    st.session_state.custom_color = st.color_picker(
                        "Choose Your Color", 
                        st.session_state.custom_color
                    )
                
                if st.session_state.extraction_mode == "Custom Image":
                    st.markdown("---")
                    custom_bg = st.file_uploader(
                        "Upload Custom Background", 
                        type=["jpg", "jpeg", "png"], 
                        key="custom_bg_upload"
                    )
                    if custom_bg:
                        st.session_state.selected_bg = Image.open(custom_bg)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_preview:
                st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
                st.markdown('<p class="preview-title">Live Preview</p>', unsafe_allow_html=True)
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(
                    st.session_state.current_image, 
                    st.session_state.mask, 
                    st.session_state.extraction_mode, 
                    bg_path, 
                    st.session_state.custom_color
                )
                
                display_img = result_pil.copy()
                display_img.thumbnail((900, 700), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Enhancement Lab Tab
        with tabs[1]:
            col_controls, col_preview = st.columns([1, 2])
            
            with col_controls:
                st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
                st.markdown('<p class="panel-title">Enhancement Controls</p>', unsafe_allow_html=True)
                
                st.markdown("**🎭 Filter Presets**")
                st.session_state.filter_type = st.selectbox(
                    "Choose Filter", 
                    list(FILTERS.keys()),
                    label_visibility="collapsed"
                )
                
                st.markdown("---")
                st.markdown("**⚡ Fine Adjustments**")
                
                st.session_state.brightness = st.slider(
                    "☀️ Brightness", 
                    0.0, 2.0, 
                    st.session_state.brightness, 
                    0.1
                )
                st.session_state.contrast = st.slider(
                    "🔆 Contrast", 
                    0.0, 2.0, 
                    st.session_state.contrast, 
                    0.1
                )
                st.session_state.saturation = st.slider(
                    "🌈 Saturation", 
                    0.0, 2.0, 
                    st.session_state.saturation, 
                    0.1
                )
                
                if st.button("🔄 Reset All", use_container_width=True):
                    st.session_state.brightness = 1.0
                    st.session_state.contrast = 1.0
                    st.session_state.saturation = 1.0
                    st.session_state.filter_type = "None"
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_preview:
                st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
                st.markdown('<p class="preview-title">Enhanced Result</p>', unsafe_allow_html=True)
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(
                    st.session_state.current_image, 
                    st.session_state.mask,
                    st.session_state.extraction_mode, 
                    bg_path, 
                    st.session_state.custom_color
                )
                result_pil = apply_filters_and_adjustments(result_pil)
                
                display_img = result_pil.copy()
                display_img.thumbnail((900, 700), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Transform Tab
        with tabs[2]:
            col_settings, col_preview = st.columns([1, 2])
            
            with col_settings:
                st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
                st.markdown('<p class="panel-title">Transform Controls</p>', unsafe_allow_html=True)
                
                st.markdown("**✂️ Aspect Ratio**")
                st.session_state.crop_preset = st.selectbox(
                    "Crop Preset", 
                    list(CROP_PRESETS.keys()),
                    label_visibility="collapsed"
                )
                
                st.markdown("---")
                st.markdown("**📐 Resize Options**")
                
                orig_h, orig_w = st.session_state.current_image.shape[:2]
                st.caption(f"Original: {orig_w} × {orig_h} px")
                
                st.session_state.resize_percent = st.slider(
                    "Scale Percentage", 
                    10, 200, 
                    st.session_state.resize_percent
                )
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                st.caption(f"New Size: {new_w} × {new_h} px")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_preview:
                st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
                st.markdown('<p class="preview-title">Transformed Preview</p>', unsafe_allow_html=True)
                
                bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
                result_pil = apply_background(
                    st.session_state.current_image, 
                    st.session_state.mask,
                    st.session_state.extraction_mode, 
                    bg_path, 
                    st.session_state.custom_color
                )
                result_pil = apply_filters_and_adjustments(result_pil)
                result_pil = crop_image(result_pil, st.session_state.crop_preset)
                
                if st.session_state.resize_percent != 100:
                    result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
                
                display_img = result_pil.copy()
                display_img.thumbnail((900, 700), Image.LANCZOS)
                st.image(display_img, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Compare View Tab
        with tabs[3]:
            st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Comparison Tools</p>', unsafe_allow_html=True)
            
            bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
            result_pil = apply_background(
                st.session_state.current_image, 
                st.session_state.mask,
                st.session_state.extraction_mode, 
                bg_path, 
                st.session_state.custom_color
            )
            result_pil = apply_filters_and_adjustments(result_pil)
            result_pil = crop_image(result_pil, st.session_state.crop_preset)
            
            if st.session_state.resize_percent != 100:
                orig_w, orig_h = result_pil.size
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)

            comparison_mode = st.radio(
                "Comparison Mode",
                ["🔲 Side-by-Side", "🎚️ Interactive Slider", "🌊 Blend View"],
                horizontal=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if comparison_mode == "🔲 Side-by-Side":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
                    st.markdown('<p class="preview-title">Original Image</p>', unsafe_allow_html=True)
                    original_img = Image.fromarray(st.session_state.original_image)
                    display_orig = original_img.copy()
                    display_orig.thumbnail((700, 700), Image.LANCZOS)
                    st.image(display_orig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
                    st.markdown('<p class="preview-title">Processed Result</p>', unsafe_allow_html=True)
                    display_result = result_pil.copy()
                    display_result.thumbnail((700, 700), Image.LANCZOS)
                    st.image(display_result, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            elif comparison_mode == "🎚️ Interactive Slider":
                st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)

            elif comparison_mode == "🌊 Blend View":
                st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
                original_img = Image.fromarray(st.session_state.original_image)
                
                if result_pil.mode == 'RGBA':
                    result_rgb = Image.new('RGB', result_pil.size, (255, 255, 255))
                    result_rgb.paste(result_pil, (0, 0), result_pil)
                else:
                    result_rgb = result_pil.convert('RGB')
                
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                blend_value = st.slider("Blend Ratio", 0.0, 1.0, 0.5, 0.01)
                
                orig_rgb = original_img.convert('RGB')
                blended = Image.blend(orig_rgb, result_rgb, float(blend_value))
                
                display_blended = blended.copy()
                display_blended.thumbnail((1000, 800), Image.LANCZOS)
                st.image(display_blended, use_container_width=True, 
                        caption=f"Blend: {int(blend_value*100)}% Processed")
                st.markdown('</div>', unsafe_allow_html=True)

        # Export Center Tab
        with tabs[4]:
            st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Export Your Masterpiece</p>', unsafe_allow_html=True)
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="premium-preview">', unsafe_allow_html=True)
            st.markdown('<p class="preview-title">Final Result</p>', unsafe_allow_html=True)
            display_final = final_result.copy()
            display_final.thumbnail((1000, 800), Image.LANCZOS)
            st.image(display_final, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown('<div class="luxury-panel">', unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Download Options</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                get_download_button(
                    final_result, 
                    export_format, 
                    quality, 
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
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.current_step = 3

    # Premium Footer
    st.markdown("---")
    st.markdown("""
    <div class="premium-footer fade-in-up">
        <h2>✦ ONEVIEW PRO ✦</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">Professional AI-Powered Image Processing Suite</p>
        <p style="font-size: 0.95rem; margin-top: 1rem; opacity: 0.8;">Precision • Excellence • Innovation</p>
        <div class="credits">
            <p>Developed with passion by <strong>Manusha</strong></p>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">© 2025 OneView Pro. All rights reserved.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
