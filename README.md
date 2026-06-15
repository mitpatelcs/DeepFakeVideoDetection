DeepFake Video Detection

A simple web application that detects whether a video is REAL or a deepfake (FAKE). Users can upload a video through the browser and receive a prediction along with a confidence score.

The core idea is that deepfake artifacts primarily appear on the face, so the system detects and crops faces from video frames before classification.

⸻

How It Works

Video → Extract 20 Frames → Face Detection (YuNet) → InceptionV3 Features → GRU Model → Prediction

1. Extract 20 evenly spaced frames from the video.
2. Detect and crop the largest face in each frame.
3. Extract a 2048-dimensional feature vector from each face using InceptionV3.
4. Process the sequence of feature vectors using a GRU-based temporal model.
5. Generate a probability score and classify the video as REAL or FAKE.

⸻

Project Structure

DeepFakeVideoDetection/
├── model/
│   ├── deepfake_video_model_v2.h5
│   └── face_detection_yunet.onnx
├── notebook/
│   └── training.ipynb
├── static/
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
├── Procfile
└── README.md

⸻

Dataset

The model was trained and evaluated on a deepfake video dataset containing both real and manipulated videos.

⸻

Model Architecture

Stage	Details
Face Detection	YuNet (OpenCV DNN)
Feature Extraction	InceptionV3 (ImageNet pretrained)
Temporal Modeling	2-layer GRU Network
Output	Probability of the video being FAKE
Training Objective	Binary Classification

⸻

Performance

Metric	Score
ROC-AUC	0.85
Recall	0.95
F1-Score	0.92

⸻

Why Face Cropping?

The original approach used full video frames as input, where facial regions occupied only a small portion of the image. By focusing directly on cropped face regions, the model captures manipulation artifacts more effectively, improving classification performance from approximately 0.59 ROC-AUC to 0.85 ROC-AUC.

⸻

Installation

Requires Python 3.10.

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

⸻

Run Locally

python app.py

Open:

http://127.0.0.1:5000

⸻

Usage

1. Open the application in your browser.
2. Upload a video containing a clearly visible face.
3. Wait for processing to complete.
4. View the prediction result and confidence score.

Note: The model performs best on videos with clear, frontal human faces. Results may vary for low-quality, heavily compressed, or partially occluded videos.

⸻

Sample Output

{
  "result": "FAKE",
  "confidence": 0.99,
  "probability_fake": 0.991,
  "probability_real": 0.009
}

⸻

Future Improvements

* Support larger video files
* Improve deployment scalability
* Optimize inference speed
* Explore transformer-based video architectures
