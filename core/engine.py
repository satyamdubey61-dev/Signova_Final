import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import math
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

# Initialize models
detector = HandDetector(staticMode=True, maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)

labels = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    labels.append(parts[1])
                else:
                    labels.append(line)
if not labels:
    labels = ["Hello", "Thank You", "Yes"]

IMG_SIZE = 300
OFFSET = 20

def preprocess_and_predict(img_array):
    """Run hand detection + classification on a BGR image. Returns (label, confidence) or (None, 0.0)."""
    if img_array is None or not hasattr(img_array, "shape") or len(img_array.shape) != 3:
        return None, 0.0

    hands, _ = detector.findHands(img_array)
    if not hands:
        return None, 0.0

    hand = hands[0]
    bbox = hand.get("bbox")
    if not bbox or len(bbox) != 4:
        return None, 0.0
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return None, 0.0

    img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

    y1 = max(0, y - OFFSET)
    y2 = min(img_array.shape[0], y + h + OFFSET)
    x1 = max(0, x - OFFSET)
    x2 = min(img_array.shape[1], x + w + OFFSET)

    img_crop = img_array[y1:y2, x1:x2]
    if img_crop.size == 0:
        return None, 0.0

    aspect_ratio = h / float(w)
    if aspect_ratio > 1:
        k = IMG_SIZE / h
        w_cal = math.ceil(k * w)
        img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
        w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
        img_white[:, w_gap : w_cal + w_gap] = img_resize
    else:
        k = IMG_SIZE / w
        h_cal = math.ceil(k * h)
        img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
        h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
        img_white[h_gap : h_cal + h_gap, :] = img_resize

    prediction, index = classifier.getPrediction(img_white, draw=False)
    if index is None or index < 0 or index >= len(labels):
        return None, 0.0
    confidence = float(prediction[index]) * 100
    return labels[index], confidence
