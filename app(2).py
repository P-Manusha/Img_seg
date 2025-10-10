import streamlit as st

st.set_page_config(page_title="OneView Test", layout="wide")

st.title("OneView - Test")
st.write("If you see this, the app is working!")

# Test imports
try:
    import torch
    st.success("✅ PyTorch imported successfully")
except Exception as e:
    st.error(f"❌ PyTorch error: {e}")

try:
    import cv2
    st.success("✅ OpenCV imported successfully")
except Exception as e:
    st.error(f"❌ OpenCV error: {e}")

try:
    from streamlit_image_comparison import image_comparison
    st.success("✅ Image comparison imported successfully")
except Exception as e:
    st.error(f"❌ Image comparison error: {e}")
