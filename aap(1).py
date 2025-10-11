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

# ============================================================================
# CUSTOM CSS - GOLD & DARK BLUE THEME
# ============================================================================
CUSTOM_CSS = """
<style>
:root {
    --primary-gold: #d4af37;
    --primary-dark: #0a0e27;
    --secondary-dark: #151932;
    --accent-blue: #4a90e2;
    --text-light: #e8e8e8;
    --text-muted: #a0a0a0;
}

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #151932 50%, #1a1f3a 100%);
    background-attachment: fixed;
}

.modern-header {
    background: rgba(21, 25, 50, 0.7);
    backdrop-filter: blur(20px);
    padding: 2rem 2.5rem;
    border-radius: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    margin-bottom: 2rem;
    border: 1px solid rgba(212, 175, 55, 0.2);
}

.modern-header h1 {
    color: #d4af37;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 2px;
}

.modern-footer {
    background: rgba(21, 25, 50, 0.5);
    border-top: 1px solid rgba(212, 175, 55, 0.2);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    color: #e8e8e8;
}

.modern-footer h3 {
    color: #d4af37;
    margin-bottom: 0.5rem;
}

.preview-card {
    background: rgba(21, 25, 50, 0.5);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.preview-header {
    color: #d4af37;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(21, 25, 50, 0.3);
    border-radius: 12px;
    padding: 0.5rem;
}

.stTabs [aria-selected="true"] {
    color: #d4af37;
}

</style>
"""

# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def apply_sepia(img):
    """Apply sepia tone filter"""
    img_array = np.array(img)
    sepia = np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img_array, sepia)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_img)

def apply_vintage(img):
    """Apply vintage filter"""
    img = ImageEnhance.Color(img).enhance(0.6)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Contrast(img).enhance(0.9)
    return img

def apply_cool_tone(img):
    """Apply cool blue tone"""
    img_array = np.array(img)
    img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.8, 0, 255)
    img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.2, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def apply_warm_tone(img):
    """Apply warm orange tone"""
    img_array = np.array(img)
    img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.2, 0, 255)
    img_array[:,:,2] = np.clip(img_array[:,:,2] * 0.8, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

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
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the segmentation model"""
    try:
        model = segmentation_models.deeplabv3_resnet50(weights=None)
        if CONFIG["num_classes"] != 21:
            model.classifier[-1] = torch.nn.Conv2d(256, CONFIG["num_classes"], kernel_size=(1, 1))
        
        if os.path.exists(CONFIG["model_path"]):
            checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
            model.load_state_dict(checkpoint)
        
        model.to(CONFIG["device"])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def segment_image(image_array):
    """Segment image using the model"""
    try:
        model = load_model()
        if model is None:
            return None
        
        img_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(CONFIG["device"])
        
        with torch.no_grad():
            output = model(img_tensor)['out']
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        return mask
    except Exception as e:
        st.error(f"Error in segmentation: {str(e)}")
        return None

def refine_mask(mask):
    """Refine mask using morphological operations"""
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def apply_background(image, mask, bg_option, bg_path, custom_color):
    """Apply background to extracted foreground"""
    image_rgb = Image.fromarray(image).convert('RGB') if isinstance(image, np.ndarray) else image.convert('RGB')
    
    if bg_option == "Remove (Transparent)":
        result = Image.new('RGBA', image_rgb.size)
        result.paste(image_rgb, mask=Image.fromarray(mask * 255))
        return result.convert('RGB')
    elif bg_option == "Custom Color":
        background = Image.new('RGB', image_rgb.size, custom_color)
        result = Image.new('RGBA', image_rgb.size)
        result.paste(image_rgb, mask=Image.fromarray(mask * 255))
        background.paste(result, mask=Image.fromarray(mask * 255))
        return background
    elif bg_option == "Blur":
        blurred = image_rgb.filter(ImageFilter.GaussianBlur(20))
        result = Image.new('RGBA', image_rgb.size)
        result.paste(image_rgb, mask=Image.fromarray(mask * 255))
        blurred.paste(result, mask=Image.fromarray(mask * 255))
        return blurred
    else:
        if bg_path and os.path.exists(bg_path):
            bg_img = Image.open(bg_path).convert('RGB')
            bg_img = bg_img.resize(image_rgb.size)
        else:
            bg_img = Image.new('RGB', image_rgb.size, (240, 240, 240))
        
        result = Image.new('RGBA', image_rgb.size)
        result.paste(image_rgb, mask=Image.fromarray(mask * 255))
        bg_img.paste(result, mask=Image.fromarray(mask * 255))
        return bg_img

def apply_filters_and_adjustments(image):
    """Apply selected filters and adjustments"""
    if 'selected_filter' in st.session_state:
        filter_func = FILTERS.get(st.session_state.selected_filter, lambda x: x)
        image = filter_func(image)
    
    if 'brightness_adjustment' in st.session_state:
        image = ImageEnhance.Brightness(image).enhance(1 + st.session_state.brightness_adjustment / 100)
    
    if 'contrast_adjustment' in st.session_state:
        image = ImageEnhance.Contrast(image).enhance(1 + st.session_state.contrast_adjustment / 100)
    
    if 'saturation_adjustment' in st.session_state:
        image = ImageEnhance.Color(image).enhance(1 + st.session_state.saturation_adjustment / 100)
    
    return image

def crop_image(image, crop_preset):
    """Crop image to specified aspect ratio"""
    if crop_preset is None:
        return image
    
    ratio_w, ratio_h = crop_preset
    img_w, img_h = image.size
    
    target_aspect = ratio_w / ratio_h
    current_aspect = img_w / img_h
    
    if current_aspect > target_aspect:
        new_w = int(img_h * target_aspect)
        left = (img_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / target_aspect)
        top = (img_h - new_h) // 2
        image = image.crop((0, top, img_w, top + new_h))
    
    return image

def get_download_button(image, export_format, quality, label, filename, key):
    """Generate download button for image"""
    buf = BytesIO()
    if export_format == "PNG":
        image.save(buf, format="PNG")
        mime = "image/png"
    else:
        image.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    
    buf.seek(0)
    st.download_button(label, buf.getvalue(), filename, mime, key=key, use_container_width=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'current_step': 0,
        'original_image': None,
        'current_image': None,
        'mask': None,
        'extraction_mode': 'Remove (Transparent)',
        'custom_color': (255, 255, 255),
        'selected_filter': 'None',
        'brightness_adjustment': 0,
        'contrast_adjustment': 0,
        'saturation_adjustment': 0,
        'crop_preset': 'Freeform',
        'resize_percent': 100,
        'blend_value': 0.5,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="OneView",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="modern-header">
        <h1>🎨 OneView</h1>
        <p>Professional AI-Powered Image Processing Solution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload & Process")
        uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image_array = np.array(image.resize(CONFIG["img_size"]))
            st.session_state.original_image = image_array
            st.session_state.current_image = image_array
            
            if st.button("🔍 Process Image", use_container_width=True):
                with st.spinner("Processing..."):
                    mask = segment_image(image_array)
                    if mask is not None:
                        mask = refine_mask(mask)
                        st.session_state.mask = mask
                        st.session_state.current_step = 1
                        st.success("✅ Segmentation complete!")
    
    # Main content
    if st.session_state.current_step == 0:
        st.info("👈 Please upload an image to get started")
        return
    
    tabs = st.tabs(["Original", "Segmentation", "Background", "Filters", "Export"])
    
    # Tab 1: Original Image
    with tabs[0]:
        st.markdown("### Original Image")
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        original_img = Image.fromarray(st.session_state.original_image)
        st.image(original_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Segmentation
    with tabs[1]:
        st.markdown("### Segmentation Result")
        if st.session_state.mask is not None:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            mask_display = (st.session_state.mask * 255).astype(np.uint8)
            st.image(mask_display, use_container_width=True, channels='GRAY')
            st.markdown('</div>', unsafe_allow_html=True)
            
            extraction_mode = st.radio(
                "Select extraction mode:",
                ["Remove (Transparent)", "Custom Color", "Blur"],
                key="extraction_mode"
            )
            
            if extraction_mode == "Custom Color":
                st.session_state.custom_color = st.color_picker("Pick background color", value="#ffffff")
                st.session_state.custom_color = tuple(int(st.session_state.custom_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Tab 3: Background
    with tabs[2]:
        st.markdown("### Apply Background")
        bg_option = st.selectbox(
            "Select background",
            ["Remove (Transparent)", "Custom Color", "Blur", "Solid Background"]
        )
        
        if st.session_state.mask is not None:
            bg_path = None if bg_option in ["Remove (Transparent)", "Custom Color", "Blur"] else "backgrounds/bg1.jpg"
            result = apply_background(st.session_state.original_image, st.session_state.mask, bg_option, bg_path, st.session_state.custom_color)
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.image(result, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Filters & Adjustments
    with tabs[3]:
        st.markdown("### Filters & Adjustments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.selected_filter = st.selectbox("Select filter", list(FILTERS.keys()), key="filter_select")
        
        with col2:
            st.session_state.brightness_adjustment = st.slider("Brightness", -50, 50, 0, key="brightness")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.session_state.contrast_adjustment = st.slider("Contrast", -50, 50, 0, key="contrast")
        
        with col4:
            st.session_state.saturation_adjustment = st.slider("Saturation", -50, 50, 0, key="saturation")
        
        if st.session_state.mask is not None:
            preview_img = Image.fromarray(st.session_state.original_image)
            preview_img = apply_filters_and_adjustments(preview_img)
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.image(preview_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: Export
    with tabs[4]:
        st.markdown("### Export Your Work")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.crop_preset = st.selectbox("Crop preset", list(CROP_PRESETS.keys()), key="crop_select")
            st.session_state.resize_percent = st.slider("Resize (%)", 10, 200, 100, key="resize")
        
        with col2:
            export_format = st.radio("Export format", ["PNG", "JPEG"], horizontal=True)
            quality = st.slider("JPEG Quality", 50, 100, 85) if export_format == "JPEG" else 85
        
        if st.session_state.mask is not None:
            bg_path = None
            final_result = apply_background(st.session_state.original_image, st.session_state.mask,
                                          st.session_state.extraction_mode, bg_path, st.session_state.custom_color)
            final_result = apply_filters_and_adjustments(final_result)
            final_result = crop_image(final_result, CROP_PRESETS[st.session_state.crop_preset])
            
            if st.session_state.resize_percent != 100:
                orig_w, orig_h = final_result.size
                new_w = int(orig_w * st.session_state.resize_percent / 100)
                new_h = int(orig_h * st.session_state.resize_percent / 100)
                final_result = final_result.resize((new_w, new_h), Image.LANCZOS)
            
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown('<div class="preview-header">Final Result</div>', unsafe_allow_html=True)
            st.image(final_result, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                get_download_button(final_result, export_format, quality, "Download Final", f"oneview_final.{export_format.lower()}", "download_final")
            
            with col2:
                buf_orig = BytesIO()
                Image.fromarray(st.session_state.original_image).save(buf_orig, format="PNG")
                buf_orig.seek(0)
                st.download_button("Download Original", buf_orig.getvalue(), "original.png", "image/png", key="download_orig", use_container_width=True)
            
            with col3:
                original_img = Image.fromarray(st.session_state.original_image)
                result_rgb = final_result.convert("RGB") if final_result.mode != "RGB" else final_result
                if original_img.size != result_rgb.size:
                    result_rgb = result_rgb.resize(original_img.size, Image.LANCZOS)
                
                comparison = np.concatenate([np.array(original_img), np.array(result_rgb)], axis=1)
                buf_comp = BytesIO()
                Image.fromarray(comparison).save(buf_comp, format="PNG")
                buf_comp.seek(0)
                st.download_button("Download Comparison", buf_comp.getvalue(), "comparison.png", "image/png", key="download_comp", use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="modern-footer">
        <h3>OneView</h3>
        <p>Professional AI-Powered Image Processing Solution</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8; color: #d4af37;">Developed by Manusha</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
