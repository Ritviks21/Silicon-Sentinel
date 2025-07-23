<p align="center">
  <img src="docs/images/banner.png" alt="Silicon Sentinel Project Banner">
</p>

<h1 align="center">Silicon Sentinel</h1>
<p align="center">
  <i>An AI-Powered System for Flawless Semiconductor Quality Control</i>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/YOLOv8-00FFFF.svg?style=for-the-badge&logo=YOLO&logoColor=black" alt="YOLOv8">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
</p>

<p align="center">
  <a href="#-project-overview">Project Overview</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-results--evaluation">Results</a> •
  <a href="#-getting-started">Getting Started</a>
</p>

---

## 📖 Project Overview

**Silicon Sentinel** is a state-of-the-art computer vision pipeline built to tackle one of the most critical challenges in the semiconductor industry: **automated defect detection**. By leveraging a fine-tuned **YOLOv8** model on a robust, custom-generated synthetic dataset, this project provides a scalable and highly accurate solution for identifying microscopic flaws like scratches, particles, and blobs on silicon wafers, directly contributing to improved manufacturing yield and quality.

---

## ✨ Key Features

- **High-Fidelity Synthetic Data**: Includes a custom data generation engine capable of creating thousands of varied training examples, including "negative" samples (clean wafers) to drastically reduce false positives.
- **Multi-Class Defect Recognition**: Accurately identifies and classifies 3 primary defect types: **scratch**, **particle**, and **blob**.
- **State-of-the-Art Accuracy**: Employs a fine-tuned YOLOv8 model with heavy data augmentation, achieving exceptional performance on unseen data.
- **End-to-End & Reproducible**: A complete, self-contained pipeline from data creation to final model evaluation, documented for easy replication.

---

## 🛠️ Tech Stack

| Python | PyTorch | YOLOv8 | OpenCV | NumPy | Colab |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="docs/images/python.png" width="48"> | <img src="docs/images/pytorch.png" width="48"> | <img src="docs/images/yolov8.png" width="48"> | <img src="docs/images/opencv.png" width="48"> | <img src="docs/images/numpy.png" width="48"> | <img src="docs/images/colab.png" width="48"> |

---

## 📊 Results & Evaluation

Our final model demonstrates a powerful ability to distinguish between clean and defective wafers, correctly identifying various defects while maintaining a low false-positive rate.

<p align="center">
  <b>Final Model Performance Chart</b><br>
  <i>(After training is complete, upload `results.png` to your `docs/images` folder and update this link)</i><br>
  <img src="docs/images/results.png" alt="Model Performance Chart" width="700">
</p>

### Prediction Examples

| Original Image (`wafer_clean.jpg`) | Model Prediction (Confidence > 0.5) |
| :---: | :---: |
| <img src="docs/images/wafer-clean.png" width="300"> | **Result: No Defects Detected ✅** <br> <i>(Upload your screenshot as `clean_prediction.png` to `docs/images` and update this link)</i><br> <img src="docs/images/clean_prediction.png" width="300"> |
| **Original Image (`wafer_all_defects.jpg`)** | **Model Prediction (Confidence > 0.5)** |
| <img src="docs/images/wafer-all-defects.png" width="300"> | **Result: All Defects Correctly Identified ✅** <br> <i>(Upload your screenshot as `defects_prediction.png` to `docs/images` and update this link)</i><br> <img src="docs/images/defects_prediction.png" width="300"> |

---

## 🚀 Getting Started

<details>
<summary>Click here for instructions to run this project yourself.</summary>

1.  **Clone the Repository**
    ```bash
    git clone [YOUR-GITHUB-REPO-LINK]
    cd Silicon-Sentinel
    ```

2.  **Install Dependencies**
    ```bash
    pip install ultralytics opencv-python numpy
    ```

3.  **Data Generation & Preparation**
    * Run the data generation script to create the `final_wafer_dataset`.
    * Run the data splitting script to create the `final_data_for_training` folder and `data.yaml`.

4.  **Train the Model**
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='final_data_for_training/data.yaml',
        epochs=50,
        imgsz=640,
        degrees=15,
        translate=0.1,
        scale=0.1,
        fliplr=0.5
    )
    ```

5.  **Evaluate the Model**
    ```python
    model = YOLO('path/to/your/best.pt')
    model.predict(source='path/to/test_images', save=True, conf=0.5)
    ```

</details>

---

## 🔗 Connect with Me

<p align="left">
<a href="[YOUR_LINKEDIN_PROFILE_URL]" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="Your LinkedIn" height="30" width="40" /></a>
<a href="[YOUR_GITHUB_PROFILE_URL]" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" alt="Your Github" height="30" width="40" /></a>
<a href="[YOUR_TWITTER_PROFILE_URL]" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="Your Twitter" height="30" width="40" /></a>
<a href="[YOUR_HUGGINGFACE_PROFILE_URL]" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/hugging-face.svg" alt="Your Hugging Face" height="30" width="40" /></a>
</p>
