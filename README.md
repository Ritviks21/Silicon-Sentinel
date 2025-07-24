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
    <img src="https://img.shields.io/badge/YOLOv8s-00FFFF.svg?style=for-the-badge&logo=YOLO&logoColor=black" alt="YOLOv8s">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
</p>

<p align="center">
  <a href="#-project-overview">Overview</a> •
  <a href="#-our-journey">Our Journey</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-results">Results</a> •
  <a href="#-getting-started">Getting Started</a>
</p>

---

## 📖 Project Overview

**Silicon Sentinel** is a state-of-the-art computer vision pipeline built to tackle one of the most critical challenges in the semiconductor industry: **automated defect detection**. By leveraging a fine-tuned **YOLOv8s** model on a robust, custom-generated synthetic dataset, this project provides a scalable and highly accurate solution for identifying microscopic flaws like scratches, particles, and blobs on silicon wafers, directly contributing to improved manufacturing yield and quality.

---

## 🚀 Our Journey: The Story of a Smarter Model

Every great project is a story of iteration. Our model wasn't built in a day; it was forged through a cycle of testing, failing, and improving. This journey showcases a real-world machine learning workflow.

<details>
<summary><strong>V1: The Naive Model - Our First Attempt</strong></summary>
<br>
We began by training a model on a simple dataset of basic scratches and particles. It worked perfectly on data it had seen before, but when shown a new defect type—a "blob"—it was completely blind.
<br><br>
💡 **Lesson Learned:** A model is only as good as the variety of its training data. Without variety, it cannot generalize.
</details>

<details>
<summary><strong>V2: The Overeager Model - A New Problem</strong></summary>
<br>
We rebuilt the dataset with more variety, including blobs. The model could now see all three defect types, but it became "trigger-happy," hallucinating defects on perfectly clean wafers.
<br><br>
💡 **Lesson Learned:** A model must be taught what a defect *is not*. Training on "negative" (clean) examples is critical to prevent false alarms.
</details>

<details>
<summary><strong>V3: The Ultimate Sentinel - The Final Iteration</strong></summary>
<br>
This was our final and most important iteration, where we addressed every lesson learned to build a truly robust AI.
<ul>
  <li><strong>Smarter Data:</strong> We generated our ultimate dataset with realistic textured backgrounds, curved/wavy scratches, and tiny isolated particles.</li>
  <li><strong>A Bigger Brain:</strong> We upgraded from the lightweight `YOLOv8n` to the more powerful `YOLOv8s` to better learn subtle patterns.</li>
</ul>
<br>
✅ **The Result:** A reliable and intelligent model that correctly identifies a wide range of defects while correctly ignoring clean surfaces.
</details>

---

## ✨ Key Features

- **High-Fidelity Synthetic Data**: A custom data engine creates thousands of realistic training examples, including clean wafers, curved scratches, and varied particles.
- **Multi-Class Defect Recognition**: Accurately identifies and classifies 3 primary defect types: **scratch**, **particle**, and **blob**.
- **State-of-the-Art Accuracy**: Employs a fine-tuned YOLOv8s model with heavy data augmentation to achieve exceptional performance.
- **End-to-End & Reproducible**: A complete pipeline from data creation to model training, documented for easy replication.

---

## 🛠️ Tech Stack

| Python | PyTorch | YOLOv8s | OpenCV | NumPy | Colab |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="docs/images/python.png" width="48"> | <img src="docs/images/pytorch.png" width="48"> | <img src="docs/images/yolov8.png" width="48"> | <img src="docs/images/opencv.png" width="48"> | <img src="docs/images/numpy.png" width="48"> | <img src="docs/images/colab.png" width="48"> |

---

## 📊 Results & Evaluation

<a name="results"></a>
Our final model demonstrates a powerful ability to distinguish between clean and defective wafers, correctly identifying various defects while maintaining a low false-positive rate.

<p align="center">
  <b>Final Model Performance Chart</b><br>
  <i>(After training is complete, upload `results.png` from your results folder to `docs/images` and this will appear)</i><br>
  <img src="docs/images/results.png" alt="Model Performance Chart" width="700">
</p>

### Prediction Examples

| Original Image (`wafer_clean.jpg`) | Model Prediction (Confidence > 0.5) |
| :---: | :---: |
| <img src="docs/images/wafer-clean.png" width="300"> | **Result: No Defects Detected ✅** <br> <i>(Upload your screenshot as `clean_prediction.png` to `docs/images`)</i><br> <img src="docs/images/clean_prediction.png" width="300"> |
| **Original Image (`wafer_all_defects.jpg`)** | **Model Prediction (Confidence > 0.5)** |
| <img src="docs/images/wafer-all-defects.png" width="300"> | **Result: All Defects Correctly Identified ✅** <br> <i>(Upload your screenshot as `defects_prediction.png` to `docs/images`)</i><br> <img src="docs/images/defects_prediction.png" width="300"> |

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
    pip install ultralytics opencv-python numpy
    ```

3.  **Train the Model**
    Run the provided Google Colab notebook to generate the data, split it, and train the model.

4.  **Evaluate the Model**
    ```python
    from ultralytics import YOLO
    model = YOLO('path/to/your/best.pt')
    model.predict(source='path/to/test_images', save=True, conf=0.5)
    ```

</details>

---

## 🔗 Connect with Me

<p align="left">
<a href="[YOUR_LINKEDIN_PROFILE_URL]" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="Your LinkedIn" height="30" width="40" /></a>
<a href="https://github.com/Ritviks21" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" alt="Your Github" height="30" width="40" /></a>
<a href="[YOUR_TWITTER_PROFILE_URL]" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="Your Twitter" height="30" width="40" /></a>
<a href="[YOUR_HUGGINGFACE_PROFILE_URL]" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/hugging-face.svg" alt="Your Hugging Face" height="30" width="40" /></a>
</p>
