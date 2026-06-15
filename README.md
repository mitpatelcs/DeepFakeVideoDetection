# 🎭 Deepfake Video Detection System

A deep learning–based web application that detects whether a video is **Real** or **Fake (Deepfake)** using computer vision and temporal sequence modeling.

---

## 📌 Overview

This project uses a face-focused video analysis pipeline to identify manipulated videos. The system extracts faces from video frames, generates deep visual features using transfer learning, and analyzes temporal patterns using a GRU-based sequence model.

---

## ✨ Features

- 📤 Upload video files through a clean web interface
- 🔍 Detect manipulated (deepfake) videos with high accuracy
- 📊 Display prediction confidence scores
- 👤 Face-based video analysis using **YuNet**
- ⚡ Real-time inference through a **Flask** application
- 🖥️ Interactive and user-friendly UI

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow |
| Computer Vision | OpenCV |
| Web Framework | Flask |
| Feature Extraction | Transfer Learning (InceptionV3) |
| Sequence Modeling | GRU (Gated Recurrent Unit) |

---

## 🧠 Model Architecture

```
Video Input
    │
    ▼
1. Video Frame Extraction
    │
    ▼
2. Face Detection using YuNet
    │
    ▼
3. Face Cropping & Preprocessing
    │
    ▼
4. Feature Extraction using InceptionV3
    │
    ▼
5. Temporal Sequence Modeling using GRU
    │
    ▼
6. Deepfake Classification → Real / Fake
```

---

## 📈 Performance

| Metric | Score |
|---|---|
| ROC-AUC | **0.85** |
| Recall | **0.95** |
| F1-Score | **0.92** |

---

## 📁 Project Structure

```
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
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mitpatelcs/DeepFakeVideoDetection.git
cd DeepFakeVideoDetection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 📖 Usage

1. 🌐 Open the application in your browser
2. 📂 Upload a video file using the upload button
3. ⏳ Wait for the processing to complete
4. ✅ View the **prediction result** and **confidence score**

---

## 🔮 Future Improvements

- [ ] Support for larger video files
- [ ] Improve deployment scalability
- [ ] Optimize inference speed
- [ ] Experiment with transformer-based video models

---

## 👤 Author

**Mit S Patel**

---

## 📄 License

This project is intended for educational and research purposes only.