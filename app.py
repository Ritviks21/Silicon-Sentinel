
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2

# Page Configuration
st.set_page_config(page_title="Silicon Sentinel", page_icon="🛡️", layout="wide")

# Title and Description
st.title("🛡️ Silicon Sentinel: AI Wafer Defect Detection")
st.write("An AI-powered system to detect microscopic defects on semiconductor wafers, trained on a hyper-realistic synthetic dataset.")

# Model Loading
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("Model file 'best.pt' not found. Please ensure it has been uploaded to the Hugging Face Space.")
        return None
    model = YOLO(model_path)
    return model

model = load_model()

# Image Uploader and Prediction
st.header("Upload Wafer Image for Analysis")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Uploaded Image", use_column_width=True)
    
    with st.spinner("The Sentinel is inspecting the wafer..."):
        results = model.predict(image, conf=0.5)
        result_image = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.image(result_image_rgb, caption="Analysis Result", use_column_width=True)

    num_detections = len(results[0].boxes)
    if num_detections > 0:
        st.success(f"Found {num_detections} potential defects with confidence > 0.5.")
    else:
        st.success("No defects found with confidence > 0.5.")
