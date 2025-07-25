<p align="center">
  <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/Silicon%20sentinel%20%20project%20banner.png" alt="Silicon Sentinel Project Banner">
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
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-our-journey">Our Journey</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-results">Results</a>
</p>

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://silicon-sentinel-qr94pmvkxeykxenptcsyb2.streamlit.app/)

**Click the badge above to try the primary live application, hosted on Streamlit Community Cloud.** This version uses our lightweight `yolov8n` model, which is optimized for fast and efficient performance on free hardware.

<br>

> ### ⚡️ High-Performance Demo on Render
> A second, more powerful version of this app using our larger `yolov8s` model is deployed on Render. You can explore the code for it on the **`render-deployment`** branch and try the live app here: **[https://silicon-sentinel.onrender.com/](https://silicon-sentinel.onrender.com/)**

---

## 📖 Project Overview

**Silicon Sentinel** is a state-of-the-art computer vision pipeline built to tackle one of the most critical challenges in the semiconductor industry: **automated defect detection**. By leveraging a fine-tuned **YOLOv8** model on a hyper-realistic, custom-generated synthetic dataset, this project provides a scalable and highly accurate solution for identifying microscopic flaws like scratches, particles, and blobs on silicon wafers.

---

## 🚀 Our Journey: The Story of a Smarter Model

This project is a testament to the iterative nature of building real-world AI. Our model was not built in a single step but was forged through a cycle of testing, diagnosing failures, and engineering targeted solutions.

<details>
<summary><strong>V1: The Naive Model - A Fragile Start</strong></summary>
<br>
Our first model was trained on a simple, clean dataset. It learned to detect basic defects but failed when shown anything new, like a "blob" defect.
<br><br>
💡 **Lesson Learned:** A model's ability to generalize depends entirely on the diversity of its training data.
</details>

<details>
<summary><strong>V2: The Overeager Model - A New Flaw Emerges</strong></summary>
<br>
We rebuilt the dataset with more variety, including blobs. The model could now see all defect types, but it became "trigger-happy," hallucinating defects on perfectly clean wafers (false positives).
<br><br>
💡 **Lesson Learned:** An AI must be taught what a defect *is not*. Training on "negative" (clean) examples is critical to prevent false alarms.
</details>

<details>
<summary><strong>V3: The Ultimate Sentinel - The Final, Robust Model</strong></summary>
<br>
Our previous model was still not perfect. It confused the background with scratches, missed tiny particles, and couldn't distinguish blobs from particle clusters. This final iteration was a targeted strike against these specific failures.
<ul>
  <li><strong>Hyper-Realistic Data:</strong> We engineered our final dataset with multiple, varied background textures, curved/wavy scratches, tiny "dust-speck" particles, and large, irregular "smudge" blobs to eliminate ambiguity.</li>
  <li><strong>A Bigger Brain:</strong> We upgraded from the lightweight `YOLOv8n` to the more powerful `YOLOv8s` model to better learn subtle patterns in our complex data.</li>
    <li><strong>More Patient Training:</strong> We increased the training time to 75 epochs, giving the more powerful model the time it needed to learn properly.</li>
</ul>
<br>
✅ **The Result:** A reliable and intelligent model that correctly identifies a wide range of defects. The journey demonstrates a realistic workflow for tackling complex computer vision challenges.
</details>

---

## ✨ Key Features

- **Hyper-Realistic Synthetic Data**: A data engine that creates thousands of training examples with varied backgrounds and highly distinct defect types.
- **Multi-Class Defect Recognition**: Accurately identifies and classifies 3 primary defect types: `scratch`, `particle`, and `blob`.
- **State-of-the-Art Accuracy**: Employs fine-tuned YOLOv8 models with heavy data augmentation.
- **End-to-End & Reproducible**: A complete pipeline from data creation to model training, documented for easy replication.

---

## 🛠️ Tech Stack

| Python | PyTorch | YOLOv8 | OpenCV | NumPy | Colab |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/Python.png" width="48"> | <img src="https://raw.githubusercontent.com/Ritviks21/Silicon-Sentinel/main/docs/images/Pytorch.png" width="48"> | <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/Yolov8s.png" width="48"> | <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/OpenCV.png" width="48"> | <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/Numpy.png" width="48"> | <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/Colab.png" width="48"> |

---

## 📊 Results & Evaluation
<a name="results"></a>
The final model demonstrates a powerful ability to identify various defects across challenging scenarios. The examples below showcase its capability to detect complex, overlapping patterns of scratches and particles.

<p align="center">
  <b>Prediction on "All Defects" Wafer</b><br>
  <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/wafer_all_defects.jpg" alt="Prediction on All Defects Wafer" width="450">
</p>

<p align="center">
  <b>Prediction on "Particles" Wafer</b><br>
  <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/wafer_particles.jpg" alt="Prediction on Particles Wafer" width="450">
</p>

---

## 🚀 Getting Started

<details>
<summary>Click here for instructions to run this project yourself.</summary>

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Ritviks21/Silicon-Sentinel.git](https://github.com/Ritviks21/Silicon-Sentinel.git)
    cd Silicon-Sentinel
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model**
    Run the provided Google Colab notebook to generate the data, split it, and train the model.

</details>

---

## 🔗 Connect with Me

<p align="left">
<a href="https://github.com/Ritviks21" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" alt="Your Github" height="30" width="40" /></a>
<a href="https://x.com/gemdata21" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="Your Twitter" height="30" width="40" /></a>
<a href="https://huggingface.co/srits21" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/hugging-face.svg" alt="Your Hugging Face" height="30" width="40" /></a>
</p>
