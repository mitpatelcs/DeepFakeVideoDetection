# DeepFake Video Detection

A simple web application that detects whether a video is **REAL** or a **deepfake (FAKE)**. Users can upload a video through the browser and receive a prediction along with a confidence score.

The core idea is that deepfake artifacts primarily appear on the face, so the system detects and crops faces from video frames before classification.

---

## How It Works

```text
Video → Extract 20 Frames → Face Detection (YuNet) → InceptionV3 Features → GRU Model → Prediction
```

1. Extract 20 evenly spaced frames from the video.
2. Detect and crop the largest face in each frame.
3. Extract a 2048-dimensional feature vector from each face using InceptionV3.
4. Process the sequence of feature vectors using a GRU-based temporal model.
5. Generate a probability score and classify the video as REAL or FAKE.

---

## Project Structure

```text
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
```

---

## Model Architecture

| Stage | Details |
|---------|---------|
| Face Detection | YuNet (OpenCV DNN) |
| Feature Extraction | InceptionV3 (ImageNet Pretrained) |
| Temporal Modeling | 2-Layer GRU Network |
| Output | Probability of the video being FAKE |
| Training Objective | Binary Classification |

---

## Performance

| Metric | Score |
|---------|---------|
| ROC-AUC | 0.85 |
| Recall | 0.95 |
| F1-Score | 0.92 |

---

## Why Face Cropping?

Deepfake artifacts are usually concentrated around facial regions such as the eyes, mouth, skin texture, and blending boundaries. By focusing on cropped face regions instead of full video frames, the model captures these artifacts more effectively and improves classification performance.

---

## Installation

Requires Python 3.10.

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

**Mac/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Locally

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Usage

1. Launch the application.
2. Upload a video file.
3. Wait for processing to complete.
4. View the prediction result and confidence score.

---

## Sample Output

```json
{
  "result": "FAKE",
  "confidence": 0.99,
  "probability_fake": 0.991,
  "probability_real": 0.009
}
```

---

## Technologies Used

- Python
- TensorFlow
- OpenCV
- Flask
- InceptionV3
- GRU (Gated Recurrent Unit)
- Computer Vision
- Transfer Learning

---

## Future Improvements

- Support larger video files
- Improve deployment scalability
- Optimize inference speed
- Explore transformer-based video architectures
- Expand training data for better generalization

---

## Author

Mit S Patel