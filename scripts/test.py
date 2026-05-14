import os
import sys
import subprocess

# Auto-restart with the correct embedded Python environment if the user runs this with the buggy system Python
embedded_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python310_embed", "python.exe")
if os.path.exists(embedded_python) and sys.executable != embedded_python:
    print(f"[INFO] Auto-restarting with the correct Python environment: {embedded_python}")
    sys.exit(subprocess.call([embedded_python, *sys.argv]))

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "converted_keras", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "converted_keras", "labels.txt")

classifier = Classifier(MODEL_PATH, LABELS_PATH)

offset = 20
imgSize = 300

labels = ["Hello", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        print("Camera not detected!")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Safe crop coordinates
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

            # Display
            cv2.putText(imgOutput, label, (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 255, 0), 2)
            cv2.rectangle(imgOutput, (x, y),
                          (x + w, y + h), (0, 255, 0), 3)

            cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("ImageCrop", imgCrop)

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

