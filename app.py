# Import necessary libraries
import streamlit as st  # Streamlit for creating the web interface
from ultralytics import YOLO  # YOLO object detection model from the ultralytics library
import cv2  # OpenCV for image processing and drawing bounding boxes
from PIL import Image  # PIL for image manipulation
import numpy as np  # NumPy for handling image arrays

# Streamlit page configuration
st.set_page_config(
    page_title="Tiny Object Detection - Buildings",  # Set the page title
    page_icon="🏢",  # Set the page icon (building emoji)
    layout="wide",  # Set layout to wide for more space
    initial_sidebar_state="expanded"  # Start with sidebar expanded
)

# Custom CSS for styling the page
st.markdown("""
    <style>
        body {
            margin-top: -10px;  # Adjust top margin
        }
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #808080;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 1.8em;
            color: #808080;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        .stTab {
            font-size: 1.4em;
            font-weight: bold;
            color: #2980B9;
        }
        .section-content{
            text-align: center;
        }
        .intro-title {
            font-size: 2.5em;
            color: #00ce39;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #017721;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            font-weight: bold;
        }
        .separator {
            height: 2px;
            background-color: #BDC3C7;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .prediction-text-good {
            font-size: 2em;
            font-weight: bold;
            color: #2980B9;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        .prediction-text-bad {
            font-size: 2em;
            font-weight: bold;
            color: #2980B9;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the Streamlit app
st.markdown('<div class="main-title">🏢 Tiny Object Detection - Buildings with YOLOv8</div>', unsafe_allow_html=True)

# Create tabs for the app
tab1, tab2 = st.tabs(["📖 About & Methodology", "📷 Upload & Detect"])

# Tab 1: About & Methodology content
with tab1:
    # Section: About Me
    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring AI/Data Scientist passionate about 
            <span class="highlight">🖼️ Computer Vision</span>, <span class="highlight">Natural Language Processing</span> and <span class="highlight">🧠 Deep Learning</span>. 
            Currently pursuing my Bachelor’s in Computer Science, I have hands-on experience in developing 
            AI models for real-world applications, particularly in object detection and NLP. 🚀
        </div>
    """, unsafe_allow_html=True)

    # Section: Project Overview
    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project focuses on developing an <span class="highlight">AI-powered tiny object detection model</span> 
            using <span class="highlight">YOLO (You Only Look Once)</span>. 
            The goal is to detect small objects (buildings) in aerial images with high accuracy, leveraging deep learning techniques.
        </div>
    """, unsafe_allow_html=True)

    # Section: Dataset Information
    st.markdown('<div class="section-title">📊 Dataset Information</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">Annotation Format:</span> The dataset follows the <b>YOLO annotation format</b>, containing bounding box labels for tiny objects.</li>
                <li><span class="highlight">Dataset Size:</span> Training Set: 184 images, Testing Set: 33 images, Validation Set: 24 images.</li>
                <li>The dataset is used to train the model to accurately detect tiny buildings in complex backgrounds.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Section: Steps Performed
    st.markdown('<div class="section-title">🔬 Steps Performed</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li>🛠 <b>Data Preprocessing:</b> Converted YOLO annotations txt format for training. Removed images without building labels and balanced dataset.</li>
                <li>📑 <b>Model Training:</b> Used <b>YOLOv8</b> for initial training and planned to experiment with YOLO for better accuracy.</li>
                <li>📈 <b>Evaluation:</b> Assessed model performance using <b>mAP (Mean Average Precision)</b> and adjusted hyperparameters for optimization.</li>
                <li>🌐 <b>Interactive UI:</b> Developed a <b>Streamlit</b> application for real-time object detection visualization.</li>
                <li>🚀 <b>Deployment:</b> Designed and deployed a user-friendly UI for testing object detection results.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Section: Technologies & Tools
    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Frameworks & Libraries:</span> TensorFlow, OpenCV, Albumentations, Streamlit.</li>
                <li><span class="highlight">⚙️ Model Architectures:</span> YOLOv8, Faster R-CNN with Feature Pyramid Networks (FPN).</li>
                <li><span class="highlight">📊 Metrics:</span> mAP (Mean Average Precision) for evaluating detection performance.</li>
                <li><span class="highlight">🌐 Deployment:</span> Streamlit for building an interactive object detection visualization tool.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Tab 2: Upload & Detect content
with tab2:
    st.markdown('<div class="section-title">🏠 Tiny Object (Building) Detection with YOLOv8</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

    # Path to the trained model
    MODEL_PATH = "./Model/best.pt"
    
    # Load the model
    model = YOLO(MODEL_PATH)

    if uploaded_file:
        # Read and convert the uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Run YOLOv8 inference for object detection
        st.write("🧐 Detecting buildings... Please wait.")
        results = model(image)

        # Draw bounding boxes on the image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                confidence = box.conf[0].item()  # Get the confidence score of the detection
                class_id = int(box.cls[0])  # Get the class ID (should be 0 for building)

                # Draw rectangle around detected building and display confidence score
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Building: {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with detected buildings
        # Display the image in medium size
        st.image(image, caption="Detected Buildings", width=700)

# Footer with credit and link to portfolio
st.markdown("""
    <div class="footer">Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app" target="_blank">Muhammad Umer Khan</a>. Powered by YOLOv8 & Streamlit 🌐 .</div>
""", unsafe_allow_html=True)