# Real-Time Pothole Detection for Intelligent Road Survey

A deep learning-based computer vision system engineered for real-time pothole detection and automated road infrastructure monitoring. This project leverages the robust features of **YOLOv8** and the cutting-edge structural advancements of **YOLO12** to achieve high-accuracy detection across varying environmental and lighting conditions.

---

## 📊 Custom Dataset & Roboflow Pipeline (Key Highlight)
Unlike standard public datasets, this project is built on a **completely custom, proprietary dataset** processed and hosted via Roboflow:
* **Data Source:** Self-captured high-definition video feeds and road survey footages.
* **Annotations:** Frame-by-frame extraction and manual bounding-box annotations mapped strictly to YOLO format standards.
* **Roboflow Integration:** The dataset is hosted dynamically on Roboflow. The configuration files (`.yaml`) securely map directly to the Roboflow data links for streamlined training and validation.
* **Real-World Challenges Covered:** Diverse lighting conditions, varying patch shadows, and distinct infrastructure contexts.

---

## 🛠️ Tech Stack & Architectures
* **Core Framework:** Python, OpenCV, PyTorch
* **Data Management:** Roboflow (Cloud-hosted dataset pipeline)
* **Detection Models:**
  * **YOLOv8:** Real-time object detection baseline with high industry reliability.
  * **YOLO12:** Next-generation architecture optimizing attention mechanisms for superior multi-scale accuracy and processing speed.

---

## 🚀 Project Structure
The repository contains configuration files and core training datasets tailored for object detection pipelines:
* `yolo8data.yaml` - Dataset and class mapping configuration optimized for the YOLOv8 pipeline (linked to Roboflow source).
* `yolo12data.yaml` - Model and category definition architecture structurally mapped for YOLO12 execution (linked to Roboflow source).
* `testing.py` - Evaluation script designed to run test inferences and analyze real-time video or image sequences.

---

## 🎬 Live Project Demo
See the real-time detection system and inference pipeline in action here:
👉 **[Watch the Live Demo on Google Drive](https://drive.google.com/file/d/1Iay8zIogXWaBlCc8xfJZgnjlaSIb8yRx/view?usp=sharing)**