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
            # This is a placeholder link. Make sure to replace it with your actual Google Drive link.
            gdrive_url = "[PASTE_YOUR_GOOGLE_DRIVE_SHARING_LINK_HERE]"
            with st.spinner("Downloading final model... (this may take a minute on first startup)"):
                gdown.download(url=gdrive_url, output=model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None
    if not os.path.exists(model_path):
        st.error("Model file failed to download.")
        return None
    model = YOLO(model_path)
    return model

model = load_model()

# --- Main App Logic ---
if model is not None:
    st.markdown("---")
    
    # --- Sample Images Section ---
    st.subheader("Try a Sample Image")
    
    # CORRECTED: This URL format correctly points to the raw image files in your GitHub repo.
    base_url = "https://raw.githubusercontent.com/Ritviks21/Silicon-Sentinel/main/sample_test_images/"
    sample_images = {
        "Clean Wafer": base_url + "wafer_clean.png",
        "Scratch": base_url + "wafer_scratch.png",
        "Particles": base_url + "wafer_particles.png",
        "Blob": base_url + "wafer_blob.png"
    }

    # Create buttons for each sample image
    cols = st.columns(len(sample_images))
    for i, (caption, url) in enumerate(sample_images.items()):
        if cols[i].button(caption):
            try:
                response = requests.get(url)
                response.raise_for_status() # Raises an error for bad responses (4xx or 5xx)
                image = Image.open(BytesIO(response.content))
                st.session_state.uploaded_file = image # Use session state to hold the image
            except Exception as e:
                st.error(f"Could not load sample image. Please check the URL/filename in your repo. Error: {e}")

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
