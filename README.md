# DeepFake Video Detection

A simple web app that detects whether a face video is **REAL** or a **deepfake (FAKE)**.
You upload a video in the browser, and a deep learning model returns the prediction
with a confidence score.

The key idea: deepfake artifacts live on the **face**, so the app crops the face out of
each frame before classifying — this is what makes the model accurate.

---

## How it works

```
Video ──> sample 20 frames ──> crop the face (YuNet) ──> InceptionV3 features ──> GRU model ──> P(FAKE)
```

1. Sample 20 evenly-spaced frames across the whole clip.
2. Detect and crop the largest face in each frame (with a small margin).
3. Turn each face into a 2048-d feature vector using InceptionV3 (ImageNet).
4. A small GRU network reads the 20-frame sequence and outputs P(FAKE).
5. `P(FAKE) >= 0.5` → FAKE, otherwise REAL. Confidence is the probability of the predicted class.

---

## Project structure

```
DeepFakeVideoDetection/
├── dataset/                 # DFDC sample videos + metadata.json (labels)
├── model/
│   ├── deepfake_video_model_v2.h5     # trained deepfake classifier (face-based)
│   └── face_detection_yunet.onnx      # YuNet face detector
├── notebook/
│   └── training.ipynb       # how the model was trained + evaluated
├── static/                  # spinner gifs
├── templates/
│   └── index.html           # upload page + result display
├── uploads/                 # temp folder for uploaded videos (auto-cleaned)
├── app.py                   # the whole app: preprocessing + inference + web server
├── README.md
├── requirements.txt
└── Procfile                 # for cloud deployment (gunicorn)
```

---

## Dataset

[DFDC – Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge) sample set.

- Videos are 10 seconds, 30 fps, 1080p, one or more faces per clip.
- Labels come from `dataset/train_sample_videos/metadata.json` (`REAL` / `FAKE`).
- On-disk distribution: **74 REAL / 320 FAKE** (imbalanced, ~4.3:1).

---

## Model architecture

| Stage | Details |
|-------|---------|
| Face detector | OpenCV **YuNet** (ONNX), largest face per frame + 30% margin |
| Feature extractor | **InceptionV3** (ImageNet, frozen), global average pooling → 2048-d per frame |
| Sequence model | GRU(64) → GRU(32) → Dropout(0.4) → Dense(32, relu) → Dropout(0.4) → Dense(1, sigmoid) |
| Output | Single sigmoid = probability the video is FAKE |
| Loss / training | binary cross-entropy, Adam, class weights, early stopping on validation AUC |

---

## Face detection pipeline (why it improved performance)

The original model fed **whole 1080p frames** to InceptionV3. On a full frame the face is
tiny, so the deepfake artifacts get averaged away — the features barely separated real from
fake (cross-validated **AUC ≈ 0.59**, almost random).

Switching to **face-cropped** frames focuses the model on exactly where manipulation happens
(blending edges, skin texture, eyes/mouth). On the same videos this raised separability to
**AUC ≈ 0.85** — the single biggest improvement in the project.

---

## Installation

Requires Python 3.10.

```shell
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to run

```shell
python app.py
```

Open <http://127.0.0.1:5000/> in your browser, upload a video, and click **Detect Deepfake**.

---

## Sample output

The `/predict` endpoint returns JSON:

```json
{
  "result": "FAKE",
  "confidence": 0.99,
  "probability_fake": 0.991,
  "probability_real": 0.009
}
```

Predictions on held-out test videos (unseen during training):

| video | truth | P(FAKE) | prediction | confidence |
|-------|-------|---------|------------|------------|
| degpbqvcay.mp4 | FAKE | 0.991 | **FAKE** ✓ | 99% |
| eprybmbpba.mp4 | FAKE | 0.997 | **FAKE** ✓ | 100% |
| bxzakyopjf.mp4 | REAL | 0.085 | **REAL** ✓ | 92% |
| dxbqjxrhin.mp4 | REAL | 0.088 | **REAL** ✓ | 91% |

---

## Results

Evaluated on a held-out test set of 99 videos (face model vs the original full-frame model):

| Metric | Old (full-frame) | **New (face-based)** |
|--------|------------------|----------------------|
| ROC-AUC | 0.649 | **0.847** |
| Recall (FAKE caught) | 0.312 | **0.950** |
| F1 score (FAKE) | 0.472 | **0.921** |
| Balanced accuracy | 0.630 | **0.738** |
| Fakes wrongly called REAL | 55 / 80 | **4 / 80** |

**Headline metrics:** ROC-AUC **0.847**, Recall **0.950**, F1 **0.921**.

> Note: the dataset is small and has few REAL videos, so the model occasionally flags a
> real video as fake. More REAL training data would improve this. See `notebook/training.ipynb`.
