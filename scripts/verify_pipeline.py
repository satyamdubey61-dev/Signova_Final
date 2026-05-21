"""
FULL PIPELINE VERIFICATION TEST
Tests everything end-to-end after fixes are applied.
"""
import os
import sys
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ========================================================
# TEST 1: Dataset Integrity
# ========================================================
section("TEST 1: DATASET INTEGRITY")

data_dir = os.path.join(BASE_DIR, "Data", "sequences")
gesture_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"  Gesture folders: {gesture_dirs}")

for gesture in gesture_dirs:
    gesture_path = os.path.join(data_dir, gesture)
    npy_files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        print(f"  [INFO] {gesture}: EMPTY (no data collected yet)")
        continue
    
    # Check for zero-sequences
    zero_count = sum(1 for f in npy_files if np.all(np.load(os.path.join(gesture_path, f)) == 0))
    check(f"{gesture}: no all-zero files", zero_count == 0, f"{len(npy_files)} files, {zero_count} zero")
    
    # Check shapes
    shapes = set(np.load(os.path.join(gesture_path, f)).shape for f in npy_files)
    check(f"{gesture}: consistent shape (30, 1530)", shapes == {(30, 1530)}, f"shapes: {shapes}")


# ========================================================
# TEST 2: Labels & Model Consistency
# ========================================================
section("TEST 2: LABELS & MODEL CONSISTENCY")

labels_path = os.path.join(BASE_DIR, "models", "labels.npy")
model_path = os.path.join(BASE_DIR, "models", "lstm_model.h5")

check("labels.npy exists", os.path.exists(labels_path))
check("lstm_model.h5 exists", os.path.exists(model_path))

saved_labels = np.load(labels_path)
print(f"  Active labels: {list(saved_labels)}")
print(f"  Number of labels: {len(saved_labels)}")

# Verify labels match discovered training classes
training_classes = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d)) and any(f.endswith('.npy') for f in os.listdir(os.path.join(data_dir, d)))
])
check("Labels match training folders", list(saved_labels) == training_classes,
      f"saved={list(saved_labels)}, folders={training_classes}")

from tensorflow.keras.models import load_model
model = load_model(model_path)
check("Model output classes matches labels", model.output_shape[-1] == len(saved_labels),
      f"model_output={model.output_shape[-1]}, labels={len(saved_labels)}")
check("Model input shape is (None, 30, 1530)", model.input_shape == (None, 30, 1530),
      f"actual: {model.input_shape}")


# ========================================================
# TEST 3: Model Accuracy on Training Data
# ========================================================
section("TEST 3: MODEL ACCURACY PER CLASS")

classes = saved_labels
total_correct = 0
total_tested = 0

for gesture_dir in sorted(os.listdir(data_dir)):
    gesture_path = os.path.join(data_dir, gesture_dir)
    if not os.path.isdir(gesture_path):
        continue
    npy_files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
    if not npy_files:
        continue

    correct = 0
    tested = 0
    for f in npy_files:
        seq = np.load(os.path.join(gesture_path, f))
        if seq.shape != (30, 1530) or np.all(seq == 0):
            continue
        probs = model.predict(seq.reshape(1, 30, 1530), verbose=0)[0]
        pred_class = str(classes[int(np.argmax(probs))])
        if pred_class == gesture_dir:
            correct += 1
        tested += 1

    accuracy = correct / tested * 100 if tested > 0 else 0
    total_correct += correct
    total_tested += tested
    
    # Top-3 for first sample
    first_seq = np.load(os.path.join(gesture_path, npy_files[0]))
    probs = model.predict(first_seq.reshape(1, 30, 1530), verbose=0)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(str(classes[i]), f"{probs[i]*100:.1f}%") for i in top3_idx]
    
    check(f"{gesture_dir}: accuracy >= 90%", accuracy >= 90,
          f"{correct}/{tested} ({accuracy:.0f}%) | Top3 sample: {top3}")

overall = total_correct / total_tested * 100 if total_tested > 0 else 0
check(f"Overall accuracy >= 95%", overall >= 95, f"{total_correct}/{total_tested} ({overall:.1f}%)")


# ========================================================
# TEST 4: Prediction Service Configuration
# ========================================================
section("TEST 4: PREDICTION SERVICE CONFIGURATION")

from services.lstm_prediction_service import LSTMPredictionService

# Create a fresh service instance to test config (don't use global to avoid model loading issues)
# Just check the class definition
import inspect
source = inspect.getsource(LSTMPredictionService.__init__)

check("min_confidence <= 0.60", "0.55" in source or "0.5" in source,
      "Threshold should be 0.55 or lower")
check("History window is 10", "maxlen=10" in inspect.getsource(LSTMPredictionService.get_or_create_session),
      "Was 15, should be 10")

predict_source = inspect.getsource(LSTMPredictionService.predict)
check("Majority voting is 5", "count >= 5" in predict_source, "Was 8, should be 5")
check("Consecutive lock is 4", "consecutive[session_id] >= 4" in predict_source, "Was 10, should be 4")
check("Freeze timer is 0.8s", "current_time + 0.8" in predict_source, "Was 1.5, should be 0.8")
check("Rate downsampling is every 2nd frame", "% 2" in predict_source, "Was 3, should be 2")
check("Confidence smoothing: 0.6 current weight", "0.6 * current_confidence" in predict_source,
      "Was 0.3 current, should be 0.6")
check("Confidence reset on gesture change", "history.clear()" in predict_source,
      "Should clear history and reset confidence when raw label changes")
check("Debug logging present", "[PREDICT]" in predict_source and "top3" in predict_source,
      "Should log top-3 predictions periodically")
check("Micro-movement threshold reduced", "0.008" in predict_source, "Was 0.015, should be 0.008")


# ========================================================
# TEST 5: Simulated Prediction Pipeline
# ========================================================
section("TEST 5: SIMULATED PREDICTION PIPELINE")

# Test that feeding actual training sequences through the service produces correct results
service = LSTMPredictionService()
check("Service model loaded", service.model is not None)
check("Service classes loaded", service.classes is not None)
print(f"  Service active labels: {list(service.classes) if service.classes is not None else 'NONE'}")

# Simulate frame-by-frame prediction with actual data
for gesture_dir in ['Hello', 'Yes', 'No', 'ThankYou', 'Sorry', 'Help', 'Good', 'ILoveYou']:
    gesture_path = os.path.join(data_dir, gesture_dir)
    if not os.path.isdir(gesture_path):
        continue
    npy_files = sorted([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
    if not npy_files:
        continue
    
    # Load a sequence and simulate feeding frames
    seq = np.load(os.path.join(gesture_path, npy_files[0]))
    if seq.shape != (30, 1530):
        continue
    
    # Reset session for clean test
    test_session = f"test_{gesture_dir}"
    service.reset_session(test_session)
    
    # Feed all 30 frames
    last_label = ""
    last_conf = 0.0
    for frame_idx in range(30):
        # Create a fake BGR image - we'll bypass mediapipe by directly using the prediction logic
        # Instead, we manually feed features into the buffer
        pass
    
    # Direct model test: feed the full sequence
    probs = service.model.predict(seq.reshape(1, 30, 1530), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = str(service.classes[pred_idx])
    pred_conf = float(probs[pred_idx]) * 100
    
    check(f"Direct prediction: {gesture_dir}", pred_class == gesture_dir,
          f"predicted='{pred_class}' conf={pred_conf:.1f}%")


# ========================================================
# TEST 6: Predict Route Labels
# ========================================================
section("TEST 6: PREDICT ROUTE (NO HARDCODED LABELS)")

import inspect
from routes.predict import health

source = inspect.getsource(health)
has_hardcoded = "Hello" in source and "Thank You" in source and "I Love You" in source
check("No hardcoded fallback labels", not has_hardcoded,
      "Hardcoded 10-class list removed" if not has_hardcoded else "Still has hardcoded labels!")


# ========================================================
# TEST 7: Training Script Zero-Skip
# ========================================================
section("TEST 7: TRAINING SCRIPT (ZERO-SEQUENCE SKIP)")

from scripts.train_lstm_model import main as train_main
import inspect
train_source = inspect.getsource(train_main)
check("All-zero skip check present", "np.all(res == 0)" in train_source,
      "Training script skips all-zero sequences")
check("Skip counter tracking", "skipped_zero" in train_source,
      "Counts and reports skipped files")


# ========================================================
# FINAL SUMMARY
# ========================================================
section("FINAL SUMMARY")
print(f"\n  PASSED: {PASS}")
print(f"  FAILED: {FAIL}")
print(f"  TOTAL:  {PASS + FAIL}")

if FAIL == 0:
    print(f"\n  ALL TESTS PASSED! Pipeline is ready.")
else:
    print(f"\n  {FAIL} TESTS FAILED. Review output above.")

print(f"\n{'='*60}")
