[🔷One View link](https://imgseg-ujvhwtyywuhllklkhxnvqv.streamlit.app/) ←- UI design Link

**AI-Powered Image Background Removal & Processing Application**

A Streamlit-based web application that uses deep learning (DeepLabV3 with ResNet50) to automatically remove backgrounds from images and provides editing tools for professional-quality results.

---

## 🎯 Key Features

- **AI Background Removal**: Automatic subject detection and background extraction using trained segmentation model.
- **Multiple Background Options**: Transparent, solid colors, blur, custom background, or preset backgrounds.
- **Image Editing**: Filters (sepia, vintage, cool/warm tones), brightness/contrast/saturation adjustments.
- **Crop & Resize**: custom sizing and Social media presets (Instagram, Facebook, Twitter).
- **Comparison Views**: Interactive slider, side-by-side, blend mode, and grid comparison.
- **Batch Processing**: Process multiple images simultaneously with same settings.
- **Project Management**: Save and load editing sessions for later use.
- **Export Options**: PNG, JPEG, WEBP ,JPG with quality control.

---

## 📦 Technologies Used

- **Streamlit**: Web application framework
- **PyTorch & Torchvision**: Deep learning and segmentation model
- **OpenCV & PIL**: Image processing and manipulation
- **NumPy & SciPy**: Numerical operations and morphological processing

---

## 🚀 Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Place trained model file `best_seg_model.pth` in root directory
3. Run application: `streamlit run app.py`
4. Upload image → Choose background → Apply effects → Export

---



**Developed by Manusha**
