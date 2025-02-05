import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Tiny Object Detection - Buildings",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            margin-top: -10px;
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


# Title
st.markdown('<div class="main-title">🏢 Tiny Object Detection - Buildings</div>', unsafe_allow_html=True)

# Tab layout
tab1, tab2 = st.tabs(["📖 About & Methodology", "📷 Upload & Detect"])

# About & Methodology

with tab1:
    # About Me
    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring AI/Data Scientist passionate about 
            <span class="highlight">🖼️ Computer Vision</span>, <span class="highlight">Natural Language Processing</span> and <span class="highlight">🧠 Deep Learning</span>. 
            Currently pursuing my Bachelor’s in Computer Science, I have hands-on experience in developing 
            AI models for real-world applications, particularly in object detection and NLP. 🚀
        </div>
    """, unsafe_allow_html=True)

    # Project Overview
    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project focuses on developing an <span class="highlight">AI-powered tiny object detection model</span> 
            using <span class="highlight">YOLO (You Only Look Once)</span>. 
            The goal is to detect small objects (buildings) in aerial images with high accuracy, leveraging deep learning techniques.
        </div>
    """, unsafe_allow_html=True)

    # Dataset Information
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

    # Steps Performed
    st.markdown('<div class="section-title">🔬 Steps Performed</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li>🛠 <b>Data Preprocessing:</b> Converted YOLO annotations to Pascal VOC format for training. Removed images without building labels and balanced dataset.</li>
                <li>📑 <b>Model Training:</b> Used <b>YOLOv8</b> for initial training and planned to experiment with YOLO for better accuracy.</li>
                <li>📈 <b>Evaluation:</b> Assessed model performance using <b>mAP (Mean Average Precision)</b> and adjusted hyperparameters for optimization.</li>
                <li>🌐 <b>Interactive UI:</b> Developed a <b>Streamlit</b> application for real-time object detection visualization.</li>
                <li>🚀 <b>Deployment:</b> Designed and deployed a user-friendly UI for testing object detection results.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Technologies & Tools
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

# Second Tab: Resume Parsing

# Upload & Detect
with tab2:
    st.markdown('<div class="section-title">📷 Upload an Image for Detection</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption="Uploaded Image", width=350)

    #     # Load YOLO model
    #     model = YOLO("yolov8l.pt")  # Change to your trained model
    #     img_array = np.array(image)

    #     # Run detection
    #     results = model(img_array)

    #     # Display detection results
    #     st.markdown('<div class="section-title">🏢 Detection Results</div>', unsafe_allow_html=True)
    #     st.image(results[0].plot(), caption="Detection Output", width=600)

# Footer
st.markdown("""
    <div class="footer">Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app" target="_blank">Muhammad Umer Khan</a>. Powered by YOLOv8 & Streamlit.</div>
""", unsafe_allow_html=True)
