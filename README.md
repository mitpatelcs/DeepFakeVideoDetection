Deepfake Video Detection System

A deep learning–based web application that detects whether a video is Real or Fake (Deepfake) using computer vision and temporal sequence modeling.

Overview

This project uses a face-focused video analysis pipeline to identify manipulated videos. The system extracts faces from video frames, generates deep visual features using transfer learning, and analyzes temporal patterns using a GRU-based sequence model.

Features

* Upload video files through a web interface
* Detect manipulated (deepfake) videos
* Display prediction confidence scores
* Face-based video analysis using YuNet
* Real-time inference through a Flask application
* Interactive and user-friendly interface

Tech Stack

* Python
* TensorFlow
* OpenCV
* Flask
* Computer Vision
* Transfer Learning
* GRU (Gated Recurrent Unit)

Model Architecture

1. Video Frame Extraction
2. Face Detection using YuNet
3. Face Cropping and Preprocessing
4. Feature Extraction using InceptionV3
5. Temporal Sequence Modeling using GRU
6. Deepfake Classification (Real/Fake)

Dataset

The model was trained and evaluated on a deepfake video dataset containing real and manipulated videos.

Performance

Metric	Score
ROC-AUC	0.85
Recall	0.95
F1-Score	0.92

Project Structure

DeepFakeVideoDetection/
│
├── app.py
├── requirements.txt
├── Procfile
│
├── model/
│   ├── deepfake_video_model_v2.h5
│   └── face_detection_yunet.onnx
│
├── templates/
│   └── index.html
│
├── static/
│   ├── spinner.gif
│   └── spinner1.gif
│
└── notebook/
    └── training.ipynb

Installation

Clone Repository

git clone https://github.com/mitpatelcs/DeepFakeVideoDetection.git
cd DeepFakeVideoDetection

Create Virtual Environment

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Run Locally

python app.py

Open your browser and navigate to:

http://127.0.0.1:5000

Usage

1. Open the application in your browser.
2. Upload a video file containing a clearly visible face.
3. Wait for processing to complete.
4. View the prediction result and confidence score.

Note: The model performs best on videos with clear, frontal human faces. Results may vary for low-quality, heavily compressed, or partially occluded videos.

Future Improvements

* Support larger video files
* Improve deployment scalability
* Optimize inference speed
* Explore transformer-based video architectures