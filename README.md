# 🏗️ Tiny Object Detection: Building Detection Project

![Building Detection](https://github.com/MuhammadUmerKhan/Computer-Vision-Task/blob/main/imgs/Building%20Sample%20Imgs/Screenshot%20From%202025-02-05%2013-20-24.png)

In urban planning, disaster management, and satellite imagery analysis, detecting tiny objects like buildings is a crucial task. This project focuses on developing a deep learning model to detect buildings in aerial imagery, addressing challenges such as small object sizes and overlapping structures. 🏠✨

This repository provides everything needed for tiny object detection, including preprocessing, model training, evaluation, and deployment. By leveraging advanced deep learning techniques, we aim to improve the accuracy of identifying tiny buildings in satellite images.

## 📜 Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Data Insights](#data-insights)
- [Key Findings](#key-findings)
- [Usage Instructions](#usage-instructions)
- [Running the Project](#running-the-project)
- [License](#license)

---

## ❓ Problem Statement

Detecting tiny objects, such as buildings in satellite images, presents significant challenges due to their small size, occlusions, and variations in lighting conditions. Traditional object detection models struggle with such minute details, necessitating specialized architectures and preprocessing techniques. 

This project aims to develop an accurate and efficient model that can identify tiny buildings in aerial imagery, which is essential for urban planning, land monitoring, and disaster response. 🚀

---

## 🛠️ Methodology

1. **Data Preprocessing & Annotation Handling:**
   - Converted YOLO `.txt` annotations format 📄➡️📂
   - Removed images with no labeled buildings to improve dataset quality
   - Addressed class imbalance using augmentation techniques 📊

2. **Exploratory Data Analysis (EDA):**
   - Visualized the distribution of buildings in images 🏢📈
   - Analyzed class imbalance and bounding box sizes
   
3. **Model Training:**
   - Implemented **YOLOv8** as the baseline model 🦾
   - Optimized hyperparameters to enhance accuracy 🎯

4. **Evaluation & Performance Metrics:**
   - Measured **Mean Average Precision (mAP)** for detection performance 🏆
   - Fine-tuned model based on false positives & missed detections

---

## 🚀 Usage Instructions

### 📂 Clone the Repository
```bash
   git clone https://github.com/MuhammadUmerKhan/Tiny-Object-Detection-Buildings.git
```

### 📦 Install Dependencies
```bash
   pip install -r requirements.txt
```

### 🎯 Model Training
Train the YOLOv8 model using:
```bash
   python train.py --model yolov8l --epochs 50
```

---

## 🏃‍♂️ Running the Project

### 🌐 Start the Streamlit App
```bash
   streamlit run app.py
```

Open your browser and navigate to:
```
   http://localhost:8501/
```

---

## 🏆 Key Findings
- **Tiny buildings are challenging to detect** due to low resolution and overlapping structures.
- **YOLOv8 performed well in detecting multiple buildings**.
- **Data preprocessing significantly impacted results**, ensuring better annotations improved model performance. 🏗️

---
## 🔴 Check Live Demo:
- [Object Detector App](https://tiny-object-detection-with-yolo.streamlit.app/)
---
## 🔖 Conclusion
By leveraging deep learning techniques, our model enhances tiny object detection, crucial for urban analysis, disaster management, and remote sensing. Future improvements could include integrating **Vision Transformers (ViTs)** for even better small-object recognition! 🌍🔬

---

💡 **Feel free to contribute, raise issues, or suggest improvements!**

📌 **License:** MIT License 🔓
