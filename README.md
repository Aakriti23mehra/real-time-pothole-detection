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

## 📊 Deployment & Configuration
The project is configured to stream data directly from Roboflow. Ensure your `.yaml` structures are pointing to your active dataset URLs:

```yaml
# Inside yolo8data.yaml / yolo12data.yaml
train: [https://dataset.roboflow.com/](https://dataset.roboflow.com/)... ( Custom Roboflow URL)
val: [https://dataset.roboflow.com/](https://dataset.roboflow.com/)...
test: [https://dataset.roboflow.com/](https://dataset.roboflow.com/)...

nc: 1
names: ['pothole']