# 📌 IPython Notebooks (`ipynb` Folder)

## Overview  
This folder contains Jupyter notebooks (`.ipynb` files) that document and execute different stages of the tiny object detection pipeline using YOLO annotations. Each notebook focuses on a specific aspect of data preprocessing, dataset balancing, model training, or evaluation.

---

## Notebooks Overview  

1️⃣ **Annotation Filtering & Preprocessing**  
   - **File:** `Step 1: annotation_filtering.ipynb`  
   - **Purpose:** Filters images and annotations to retain only those containing buildings (class `0`).  
   - **Key Steps:**  
     - Reads YOLO annotation files and extracts valid bounding boxes.  
     - Removes images without building annotations.  
     - Saves the filtered dataset in structured folders for training, validation, and testing.  

2️⃣ **Dataset Balancing**  
   - **File:** `Step 2: dataset_balancing.ipynb`  
   - **Purpose:** Ensures the dataset is well-distributed across `train`, `val`, and `test` sets.  
   - **Key Steps:**  
     - Checks dataset statistics (image counts per split).  
     - Computes the percentage distribution of splits.  
     - Moves images and labels to balance the dataset.  

3️⃣ **YOLOv8 Model Training**  
   - **File:** `Step 3: train_yolo.ipynb`  
   - **Purpose:** Implements YOLOv8-based object detection for detecting tiny buildings in aerial images.  
   - **Key Steps:**  
     - Loads YOLOv8m model and configures hyperparameters.  
     - Trains the model for 100 epochs with data augmentation.  
     - Saves the best-performing model for evaluation.  

4️⃣ **Model Evaluation & Performance Analysis**  
   - **File:** `Step 4: evaluate_model.ipynb`  
   - **Purpose:** Evaluates the trained model’s accuracy using mAP metrics.  
   - **Key Steps:**  
     - Loads the best model checkpoint.  
     - Computes mAP50 and mAP50-95 scores to assess detection performance.  
     - Identifies potential reasons for low accuracy and suggests improvements.  

---

## How to Use These Notebooks  
1. Open a Jupyter Notebook environment (JupyterLab, VS Code, or Google Colab).  
2. Navigate to the `ipynb` folder and open the relevant notebook.  
3. Run the cells sequentially to execute the code and visualize results.  

---

## Next Steps  
- If model performance is unsatisfactory, consider fine-tuning the dataset, increasing training epochs, or trying alternative architectures like Faster R-CNN.  
- Modify `Step 4 File` to experiment with different YOLOv8 variants (`yolov8x` for better accuracy).  
