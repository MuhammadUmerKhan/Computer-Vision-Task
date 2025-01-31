# **Tiny Object Detection - Building Detection Project**

🎯 **Project Goal**:
This project aims to build an object detection system to identify buildings in images. The dataset contains labeled images with bounding boxes, and we are focusing on detecting the **"Building"** class (Class 0) using deep learning models.

---

## **🔨 Process Overview**

### **Step 1: Data Preprocessing**
1. **Cleaning the Dataset**:
   - Filtered out only the "Building" class (Class 0) annotations from the dataset.
   - Removed irrelevant bounding boxes (other classes like A, B, C, etc.) and kept only those with class `0` (Buildings).

2. **Data Validation**:
   - Checked the dataset splits (`train`, `val`, `test`) to ensure they are balanced.
   - Ensured that only images with "Building" class annotations were kept and irrelevant images/annotations were removed.

---

### **Step 2: Model Selection**
1. **Pretrained YOLO Model (Recommended)**:
   - Chose **YOLO** as the object detection model, specifically YOLOv5 or YOLOv8.
   - Since the dataset is already in YOLO format, the YOLO model is a perfect fit for fast training and detection.

2. **Custom Model**:
   - Alternatively, a custom object detection model (like Faster R-CNN or SSD using TensorFlow) could be used.
   - The dataset would be converted to Pascal VOC or COCO format for compatibility with TensorFlow models.

---

### **Step 3: Model Training**
- **Next Steps**:
  - Set up and train the YOLO model on the cleaned dataset (focusing on "Building" class only).
  - Monitor key metrics such as mAP (Mean Average Precision), loss, precision, and recall.
  - Save the trained model weights for later evaluation.

---

### **Step 4: Evaluation**
- **Test the Model** on the `test` dataset.
- Calculate mAP and compare with baseline models (if multiple models are tested).
- Ensure the model is accurately detecting buildings.

---

### **Step 5: (Optional) Deployment**
- Build a **Streamlit Web App** for easy interaction, where users can upload images and get bounding box predictions for buildings.

---

## **📂 Directory Structure**

- `images/`: Contains the images split into `train`, `val`, and `test` folders.
- `labels/`: Contains the corresponding label files in YOLO format.
- `filtered_labels/`: Cleaned dataset after removing irrelevant bounding boxes.
- `notebooks/`: Jupyter notebooks for preprocessing and training.

---

## **🚀 Next Steps**
1. **Model Training**:
   - Train the YOLO model or a custom TensorFlow model on the filtered dataset.
   
2. **Evaluation**:
   - Evaluate the model's performance using the test set and compute mAP.

3. **Deployment (Optional)**:
   - Deploy the trained model via a Streamlit app for easy demonstration.

---

## **📜 License**
This project is open-source. You are free to use and modify it for your own projects.

