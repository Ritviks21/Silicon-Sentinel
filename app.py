import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
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
    """Loads the model directly from the repository."""
    # MODIFIED: Looking for the specific filename 'best (2).pt'
    model_path = "best (2).pt"
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it has been uploaded to the GitHub repository with this exact name.")
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
