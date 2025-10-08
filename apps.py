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
# CUSTOM CSS
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.main-header p {
    font-size: 1.1rem;
    margin-top: 0.5rem;
    opacity: 0.95;
}

.step-indicator {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
}

.step-indicator h3 {
    color: #2d3748;
    font-weight: 700;
    margin: 0;
}

.preview-container {
    background: #f8f9fa;
    padding: 2rem;
    border-radius: 12px;
    margin: 1.5rem 0;
    border: 2px solid #e9ecef;
}

.preview-image-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 1rem 0;
}

.preview-image-wrapper img {
    max-width: 600px;
    max-height: 500px;
    width: auto !important;
    height: auto !important;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.footer-professional {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    color: white;
    margin-top: 3rem;
}

.footer-professional h3 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 800;
}

.footer-professional p {
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

div[data-testid="stExpander"] {
    background: white;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    margin: 1rem 0;
}

.comparison-container {
    max-width: 800px;
    margin: 0 auto;
}
</style>
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_sepia(img):
    img_arr = np.array(img)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(img_arr, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_img)

def apply_vintage(img):
    img = ImageEnhance.Contrast(img).enhance(0.85)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = apply_sepia(img)
    return img

def apply_cool_tone(img):
    img_arr = np.array(img).astype(float)
    img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] * 1.15, 0, 255)
    img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] * 0.9, 0, 255)
    return Image.fromarray(img_arr.astype(np.uint8))

def apply_warm_tone(img):
    img_arr = np.array(img).astype(float)
    img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] * 1.15, 0, 255)
    img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] * 0.9, 0, 255)
    return Image.fromarray(img_arr.astype(np.uint8))

@st.cache_resource
def load_model(model_path, device, num_classes=2):
    model = segmentation_models.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_mask(model, img_array, device, img_size):
    img_resized = cv2.resize(img_array, img_size)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        out = model(img_tensor)['out']
        prob = torch.sigmoid(out[:, 1, :, :]).squeeze().cpu().numpy()
    
    prob_resized = cv2.resize(prob, (img_array.shape[1], img_array.shape[0]))
    return prob_resized

def postprocess_mask(prob, fg_thresh=0.3, min_area=500):
    mask = (prob > fg_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    
    return mask

def apply_background(img_array, mask_bin, mode, bg_path=None, custom_color=(255, 255, 255)):
    fg = img_array.copy()
    alpha = mask_bin * 255
    
    if mode == "Transparent":
        result = np.dstack([fg, alpha])
        return Image.fromarray(result, mode="RGBA")
    
    elif mode == "White":
        bg = np.ones_like(img_array) * 255
    
    elif mode == "Black":
        bg = np.zeros_like(img_array)
    
    elif mode == "Custom Color":
        bg = np.ones_like(img_array) * np.array(custom_color, dtype=np.uint8)
    
    elif mode in ["Bg1", "Bg2", "Bg3", "Bg4"]:
        if bg_path and os.path.exists(bg_path):
            bg_img = Image.open(bg_path).convert("RGB")
            bg_img = bg_img.resize((img_array.shape[1], img_array.shape[0]), Image.LANCZOS)
            bg = np.array(bg_img)
        else:
            bg = np.ones_like(img_array) * 255
    
    else:
        bg = np.ones_like(img_array) * 255
    
    mask_3ch = np.stack([mask_bin] * 3, axis=-1)
    composite = fg * mask_3ch + bg * (1 - mask_3ch)
    return Image.fromarray(composite.astype(np.uint8), mode="RGB")

def apply_filters_and_adjustments(img):
    if st.session_state.selected_filter != "None":
        img = FILTERS[st.session_state.selected_filter](img)
    
    if st.session_state.brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(st.session_state.brightness)
    if st.session_state.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(st.session_state.contrast)
    if st.session_state.saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(st.session_state.saturation)
    if st.session_state.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(st.session_state.sharpness)
    
    if st.session_state.rotation != 0:
        img = img.rotate(st.session_state.rotation, expand=True, fillcolor=(255, 255, 255))
    
    if st.session_state.flip_horizontal:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if st.session_state.flip_vertical:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    return img

def crop_image(img, preset):
    if preset == "Freeform" or CROP_PRESETS[preset] is None:
        return img
    
    aspect_w, aspect_h = CROP_PRESETS[preset]
    img_w, img_h = img.size
    
    target_ratio = aspect_w / aspect_h
    img_ratio = img_w / img_h
    
    if img_ratio > target_ratio:
        new_w = int(img_h * target_ratio)
        new_h = img_h
        left = (img_w - new_w) // 2
        top = 0
    else:
        new_w = img_w
        new_h = int(img_w / target_ratio)
        left = 0
        top = (img_h - new_h) // 2
    
    return img.crop((left, top, left + new_w, top + new_h))

def get_download_button(img, format_type, quality, label, filename, key):
    buf = BytesIO()
    if format_type == "PNG" and img.mode == "RGBA":
        img.save(buf, format="PNG")
        mime = "image/png"
    elif format_type in ["JPEG", "JPG"]:
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        img.convert("RGB").save(buf, format=format_type, quality=quality)
        mime = f"image/{format_type.lower()}"
    
    st.download_button(label, buf.getvalue(), filename, mime, key=key, use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="OneView - AI Image Processing", page_icon="🎨", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✨ OneView</h1>
        <p>Professional AI-Powered Image Processing & Background Removal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processed_mask' not in st.session_state:
        st.session_state.processed_mask = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'batch_mode' not in st.session_state:
        st.session_state.batch_mode = False
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'fg_thresh' not in st.session_state:
        st.session_state.fg_thresh = CONFIG["fg_thresh"]
    if 'min_area' not in st.session_state:
        st.session_state.min_area = 500
    if 'extraction_mode' not in st.session_state:
        st.session_state.extraction_mode = "Transparent"
    if 'custom_color' not in st.session_state:
        st.session_state.custom_color = (255, 255, 255)
    if 'selected_filter' not in st.session_state:
        st.session_state.selected_filter = "None"
    if 'brightness' not in st.session_state:
        st.session_state.brightness = 1.0
    if 'contrast' not in st.session_state:
        st.session_state.contrast = 1.0
    if 'saturation' not in st.session_state:
        st.session_state.saturation = 1.0
    if 'sharpness' not in st.session_state:
        st.session_state.sharpness = 1.0
    if 'rotation' not in st.session_state:
        st.session_state.rotation = 0
    if 'flip_horizontal' not in st.session_state:
        st.session_state.flip_horizontal = False
    if 'flip_vertical' not in st.session_state:
        st.session_state.flip_vertical = False
    if 'crop_preset' not in st.session_state:
        st.session_state.crop_preset = "Freeform"
    if 'resize_percent' not in st.session_state:
        st.session_state.resize_percent = 100
    if 'blend_slider' not in st.session_state:
        st.session_state.blend_slider = 1.0
    if 'zoom_percentage' not in st.session_state:
        st.session_state.zoom_percentage = 100
    
    # Load model
    model = load_model(CONFIG["model_path"], CONFIG["device"], CONFIG["num_classes"])
    
    # Step 1: Upload
    st.markdown('<div class="step-indicator"><h3>📤 Step 1: Upload Your Image</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"], 
                                        accept_multiple_files=False)
    
    with col2:
        st.session_state.batch_mode = st.checkbox("🔄 Batch Mode", value=st.session_state.batch_mode)
        
        if st.session_state.batch_mode:
            batch_files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg", "webp"], 
                                          accept_multiple_files=True, key="batch_uploader")
            if batch_files:
                st.session_state.uploaded_images = [np.array(Image.open(f).convert("RGB")) for f in batch_files]
    
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        demo_choice = st.selectbox("Or try a demo image:", ["None"] + list(DEMO_IMAGE_PATHS.keys()))
    
    if demo_choice != "None":
        demo_path = DEMO_IMAGE_PATHS[demo_choice]
        if os.path.exists(demo_path):
            demo_img = Image.open(demo_path).convert("RGB")
            st.session_state.original_image = np.array(demo_img)
            st.session_state.current_step = 2
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_image = np.array(img)
        st.session_state.current_step = 2
    
    # Step 2: Process
    if st.session_state.original_image is not None:
        st.markdown("---")
        st.markdown('<div class="step-indicator"><h3>⚙️ Step 2: Process Image</h3></div>', unsafe_allow_html=True)
        
        with st.expander("🎚️ Processing Settings", expanded=True):
            proc_col1, proc_col2 = st.columns(2)
            with proc_col1:
                st.session_state.fg_thresh = st.slider("Foreground Threshold", 0.1, 0.9, st.session_state.fg_thresh, 0.05)
            with proc_col2:
                st.session_state.min_area = st.slider("Minimum Area", 100, 2000, st.session_state.min_area, 100)
        
        if st.button("🚀 Process Image", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                prob = predict_mask(model, st.session_state.original_image, CONFIG["device"], CONFIG["img_size"])
                st.session_state.processed_mask = postprocess_mask(prob, st.session_state.fg_thresh, st.session_state.min_area)
                st.session_state.current_step = 3
            st.success("✅ Processing complete!")
    
    # Step 3: Customize
    if st.session_state.processed_mask is not None:
        st.markdown("---")
        st.markdown('<div class="step-indicator"><h3>🎨 Step 3: Customize Result</h3></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Background", "🎭 Filters", "✨ Adjustments", "📐 Transform"])
        
        with tab1:
            st.markdown("#### Background Options")
            bg_col1, bg_col2 = st.columns([2, 1])
            
            with bg_col1:
                st.session_state.extraction_mode = st.selectbox(
                    "Background Mode",
                    ["Transparent", "White", "Black", "Custom Color", "Bg1", "Bg2", "Bg3", "Bg4"]
                )
            
            with bg_col2:
                if st.session_state.extraction_mode == "Custom Color":
                    color_choice = st.color_picker("Pick Color", "#FFFFFF")
                    st.session_state.custom_color = tuple(int(color_choice.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        with tab2:
            st.markdown("#### Filter Presets")
            filter_cols = st.columns(3)
            for idx, filter_name in enumerate(FILTERS.keys()):
                with filter_cols[idx % 3]:
                    if st.button(filter_name, use_container_width=True, key=f"filter_{filter_name}"):
                        st.session_state.selected_filter = filter_name
            
            st.info(f"Selected: **{st.session_state.selected_filter}**")
        
        with tab3:
            st.markdown("#### Image Adjustments")
            adj_col1, adj_col2 = st.columns(2)
            
            with adj_col1:
                st.session_state.brightness = st.slider("Brightness", 0.5, 2.0, st.session_state.brightness, 0.1)
                st.session_state.contrast = st.slider("Contrast", 0.5, 2.0, st.session_state.contrast, 0.1)
            
            with adj_col2:
                st.session_state.saturation = st.slider("Saturation", 0.0, 2.0, st.session_state.saturation, 0.1)
                st.session_state.sharpness = st.slider("Sharpness", 0.0, 2.0, st.session_state.sharpness, 0.1)
        
        with tab4:
            st.markdown("#### Transform Options")
            trans_col1, trans_col2, trans_col3 = st.columns(3)
            
            with trans_col1:
                st.session_state.rotation = st.slider("Rotation (°)", -180, 180, st.session_state.rotation, 15)
                st.session_state.crop_preset = st.selectbox("Crop Preset", list(CROP_PRESETS.keys()))
            
            with trans_col2:
                st.session_state.flip_horizontal = st.checkbox("Flip Horizontal", st.session_state.flip_horizontal)
                st.session_state.flip_vertical = st.checkbox("Flip Vertical", st.session_state.flip_vertical)
            
            with trans_col3:
                st.session_state.resize_percent = st.slider("Resize (%)", 25, 200, st.session_state.resize_percent, 5)
        
        if st.button("✨ Apply Customizations", use_container_width=True, type="primary"):
            st.session_state.current_step = 4
            st.rerun()
    
    # Step 4: Final Preview & Export (REWRITTEN SECTION)
    if st.session_state.current_step >= 4 and st.session_state.processed_mask is not None:
        st.markdown("---")
        st.markdown('<div class="step-indicator"><h3>🎬 Step 4: Final Preview & Export</h3></div>', unsafe_allow_html=True)
        
        # Generate final result
        mask_bin = (st.session_state.processed_mask > 127).astype(np.uint8)
        bg_path = BG_IMAGE_PATHS.get(st.session_state.extraction_mode)
        result_pil = apply_background(st.session_state.original_image, mask_bin, 
                                     st.session_state.extraction_mode, bg_path, 
                                     st.session_state.custom_color)
        result_pil = apply_filters_and_adjustments(result_pil)
        result_pil = crop_image(result_pil, st.session_state.crop_preset)
        
        if st.session_state.resize_percent != 100:
            orig_w, orig_h = result_pil.size
            new_w = int(orig_w * st.session_state.resize_percent / 100)
            new_h = int(orig_h * st.session_state.resize_percent / 100)
            result_pil = result_pil.resize((new_w, new_h), Image.LANCZOS)
        
        # Export Settings
        st.markdown("### ⚙️ Export Settings")
        export_settings_col1, export_settings_col2 = st.columns(2)
        
        with export_settings_col1:
            export_format = st.selectbox("Format", ["PNG", "JPEG", "WEBP"], index=0)
        
        with export_settings_col2:
            quality = st.slider("Quality", 1, 100, 95, 1) if export_format in ["JPEG", "WEBP"] else 95
        
        # Preview Section
        st.markdown("### 👁️ Image Preview")
        
        preview_tab1, preview_tab2 = st.tabs(["📸 Final Result", "🔄 Compare View"])
        
        with preview_tab1:
            st.markdown('<div class="preview-container">', unsafe_allow_html=True)
            st.markdown("#### Final Processed Image")
            
            # Display medium-sized image
            display_result = result_pil.copy()
            display_result.thumbnail((600, 500), Image.LANCZOS)
            
            col_center = st.columns([1, 2, 1])
            with col_center[1]:
                st.image(display_result, caption="Your processed image", use_container_width=False)
            
            # Image info
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Width", f"{result_pil.size[0]}px")
            with info_col2:
                st.metric("Height", f"{result_pil.size[1]}px")
            with info_col3:
                st.metric("Mode", result_pil.mode)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with preview_tab2:
            st.markdown('<div class="preview-container comparison-container">', unsafe_allow_html=True)
            
            controls_col, view_col = st.columns([1, 2])
            
            with controls_col:
                st.markdown("#### 🎛️ Adjust View")
                st.session_state.blend_slider = st.slider("Blend", 0.0, 1.0, 
                                                         st.session_state.blend_slider, 0.05)
                st.session_state.zoom_percentage = st.slider("Zoom (%)", 50, 200, 
                                                            st.session_state.zoom_percentage, 5)
            
            with view_col:
                @st.cache_data(show_spinner=False)
                def prepare_comparison(original_array, result_bytes, zoom, blend):
                    original_img = Image.fromarray(original_array)
                    result_img = Image.open(BytesIO(result_bytes))
                    
                    scale = zoom / 100.0
                    new_w = max(1, int(original_img.size[0] * scale))
                    new_h = max(1, int(original_img.size[1] * scale))
                    
                    zoomed_orig = original_img.resize((new_w, new_h), Image.LANCZOS)
                    zoomed_result = result_img.resize((new_w, new_h), Image.LANCZOS)
                    
                    if zoomed_result.mode == 'RGBA':
                        result_rgb = Image.new('RGB', zoomed_result.size, (0, 0, 0))
                        result_rgb.paste(zoomed_result, (0, 0), zoomed_result)
                    else:
                        result_rgb = zoomed_result.convert('RGB')
                    
                    orig_rgb = zoomed_orig.convert('RGB')
                    blended = Image.blend(orig_rgb, result_rgb, float(blend))
                    
                    return orig_rgb, blended
                
                result_bytes = BytesIO()
                result_pil.save(result_bytes, format='PNG')
                
                orig_rgb, blended = prepare_comparison(st.session_state.original_image, 
                                                      result_bytes.getvalue(),
                                                      st.session_state.zoom_percentage, 
                                                      st.session_state.blend_slider)
                
                image_comparison(img1=orig_rgb, img2=blended, label1="Original",
                               label2=f"Blended ({int(st.session_state.blend_slider*100)}%)")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
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
