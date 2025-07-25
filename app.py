import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import gdown
import cv2
import requests
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="Silicon Sentinel", page_icon="🛡️", layout="wide")

# --- Title and Description ---
st.title("🛡️ Silicon Sentinel: AI Wafer Defect Detection")
st.write("An AI-powered system to detect microscopic defects on semiconductor wafers, trained on a hyper-realistic synthetic dataset.")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Downloads the model from Google Drive and loads it."""
    model_path = "best.pt"
    if not os.path.exists(model_path):
        try:
            gdrive_url = "[PASTE_YOUR_GOOGLE_DRIVE_SHARING_LINK_FOR_MODEL_HERE]"
            with st.spinner("Downloading final model... (this may take a minute on first startup)"):
                gdown.download(url=gdrive_url, output=model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None
    if not os.path.exists(model_path):
        st.error("Model file failed to download. The application cannot start.")
        return None
    model = YOLO(model_path)
    return model

model = load_model()

# --- Main App Logic ---
if model is not None:
    st.markdown("---")
    
    # --- Sample Image Section ---
    st.subheader("Try a Sample Image")
    sample_image_url = "https://github.com/Ritviks21/Silicon-Sentinel/raw/main/sample_test_images/wafer_all_defects.png"
    if st.button("Test a Sample Image with Multiple Defects"):
        try:
            response = requests.get(sample_image_url)
            st.session_state.uploaded_file = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Could not load the sample image from GitHub. Error: {e}")

    # --- File Uploader Section ---
    st.subheader("Or, Upload Your Own Image")
    uploaded_file_obj = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file_obj:
        st.session_state.uploaded_file = Image.open(uploaded_file_obj)

    # --- Prediction and Display ---
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        image = st.session_state.uploaded_file
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Selected Image", use_column_width=True)
        with st.spinner("The Sentinel is inspecting the wafer..."):
            results = model.predict(image, conf=0.4)
            result_image = results[0].plot()
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        with col2:
            st.image(result_image_rgb, caption="Analysis Result", use_column_width=True)
        num_detections = len(results[0].boxes)
        if num_detections > 0:
            st.success(f"Found {num_detections} potential defects.")
        else:
            st.success("No defects found with the current confidence threshold.")
            
    # --- Download Dataset Section ---
    st.markdown("---")
    st.subheader("Download Full Test Dataset")
    st.write("To test with more images, you can download the complete, high-quality synthetic dataset used to train this model.")
    
    # Correct, final link to your GitHub Release
    release_url = "https://github.com/Ritviks21/Silicon-Sentinel/releases/download/v1.0-data/ultimate_wafer_dataset.zip"
    
    st.markdown(f'<a href="{release_url}" target="_blank"><button style="color: white; background-color: #FF4B4B; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;">Download Full Dataset (ZIP)</button></a>', unsafe_allow_html=True)
