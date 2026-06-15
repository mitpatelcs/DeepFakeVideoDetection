from flask import Flask, request, jsonify, render_template
import os
import ssl

from wcwidth import width
ssl._create_default_https_context = ssl._create_unverified_context  # allow ImageNet weights download

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Ensure the uploads directory exists (Gunicorn does not run the __main__ block)
os.makedirs("uploads", exist_ok=True)

IMG_SIZE = 224              # frame size fed to InceptionV3
NUM_FEATURES = 2048        # InceptionV3 output features per frame
SEQ_LEN = 20               # frames sampled per video
DECISION_THRESHOLD = 0.5   
MODEL_PATH = "model/deepfake_video_model_v2.h5"
YUNET_PATH = "model/face_detection_yunet.onnx"

# Build InceptionV3 feature extractor
def build_feature_extractor():
    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    outputs = base_model(preprocess_input(inputs))
    feature_extractor = Model(inputs, outputs, name="feature_extractor")
    
    return feature_extractor

# Load trained deepfake model
model = load_model(MODEL_PATH)
feature_extractor = build_feature_extractor()

# Create YuNet face detector
def create_face_detector():
    return cv2.FaceDetectorYN_create(YUNET_PATH, "", (320, 320), score_threshold=0.6)

# Crop the center square of a frame
def crop_center_square(frame):
    height, width = frame.shape[:2]
    square_size = min(height, width)

    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2

    return frame[start_y:start_y + square_size,start_x:start_x + square_size]

# Crop the largest detected face (with margin);
# if no face detected, crop the center square
def crop_face(frame, detector, margin=0.30):

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    # Use center crop if no face is found
    if faces is None or len(faces) == 0:
        face_crop = crop_center_square(frame)
    else:
        # Find largest face
        largest_face = faces[0]
        for face in faces:
            current_area = face[2] * face[3]
            largest_area = largest_face[2] * largest_face[3]
            if current_area > largest_area:
                largest_face = face

        x, y, face_width, face_height = largest_face[:4]

        center_x = x + face_width / 2
        center_y = y + face_height / 2
        crop_size = max(face_width, face_height)
        crop_size = crop_size * (1 + margin)
        
        start_x = int(max(0, center_x - crop_size / 2))
        start_y = int(max(0, center_y - crop_size / 2))
        end_x = int(min(width, center_x + crop_size / 2))
        end_y = int(min(height, center_y + crop_size / 2))

        face_crop = frame[start_y:end_y,start_x:end_x]

        # Use center crop if face crop fails
        if face_crop.size == 0:
            face_crop = crop_center_square(frame)

    # Resize face image
    face_crop = cv2.resize(face_crop,(IMG_SIZE, IMG_SIZE))
    
    # Convert BGR to RGB
    face_crop = cv2.cvtColor(face_crop,cv2.COLOR_BGR2RGB)
    return face_crop

# Extract face frames from video
def extract_face_frames(video_path, detector):

    cap = cv2.VideoCapture(video_path)
    total_frames = int( cap.get(cv2.CAP_PROP_FRAME_COUNT))
       

    if total_frames > 0:
        frame_step = max(1,total_frames // SEQ_LEN)
    else:
        frame_step = 1

    frames = []
    frame_index = 0

    while len(frames) < SEQ_LEN:
        success, frame = cap.read()
        if not success:
            break

        # Select evenly spaced frames
        if frame_index % frame_step == 0:

            face_frame = crop_face(frame,detector)
            frames.append(face_frame)

        frame_index += 1

    cap.release()
    return np.array(frames)

# Convert video into model input
def video_to_model_input(video_path):

    detector = create_face_detector()
    frames = extract_face_frames(video_path,detector)

    valid_frames = min(SEQ_LEN,len(frames))

    features = np.zeros((1, SEQ_LEN, NUM_FEATURES),dtype="float32")

    mask = np.zeros((1, SEQ_LEN), dtype="bool")

    if valid_frames > 0:
        extracted_features = feature_extractor.predict(frames[:valid_frames],verbose=0)

        features[0, :valid_frames] = extracted_features
        mask[0, :valid_frames] = True

    return features, mask, valid_frames

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)

    try:
        feats, mask, n = video_to_model_input(video_path)
        if n == 0:
            return jsonify({'error': 'Could not read any frames from the video.'}), 400

        p_fake = float(model.predict([feats, mask], verbose=0)[0][0])   # P(FAKE)
        p_real = 1.0 - p_fake
        result = 'FAKE' if p_fake >= DECISION_THRESHOLD else 'REAL'
        confidence = p_fake if result == 'FAKE' else p_real     # confidence in predicted label
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)   # delete the uploaded file

    return jsonify({
        'result': result,
        'confidence': confidence,
        'probability_fake': p_fake,
        'probability_real': p_real,
    })

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)