"""Quick test: feed actual training data to the model and see prediction results."""
import os, sys, numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

model = load_model(os.path.join(BASE_DIR, "models", "lstm_model.h5"))
classes = np.load(os.path.join(BASE_DIR, "models", "labels.npy"))
data_dir = os.path.join(BASE_DIR, "Data", "sequences")

print(f"Classes: {list(classes)}")
print(f"Model output classes: {model.output_shape[-1]}")
print()

correct_total = 0
wrong_total = 0

for gesture_dir in sorted(os.listdir(data_dir)):
    gesture_path = os.path.join(data_dir, gesture_dir)
    if not os.path.isdir(gesture_path):
        continue
    npy_files = sorted([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
    if not npy_files:
        print(f"  {gesture_dir}: EMPTY - no data")
        continue

    correct = 0
    wrong = 0
    wrong_as = {}

    for f in npy_files:
        seq = np.load(os.path.join(gesture_path, f))
        if seq.shape != (30, 1530):
            continue
        probs = model.predict(seq.reshape(1, 30, 1530), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = str(classes[pred_idx])
        if pred_class == gesture_dir:
            correct += 1
        else:
            wrong += 1
            wrong_as[pred_class] = wrong_as.get(pred_class, 0) + 1

    accuracy = correct / (correct + wrong) * 100 if (correct + wrong) > 0 else 0
    correct_total += correct
    wrong_total += wrong
    
    status = "OK" if accuracy > 80 else "POOR" if accuracy > 50 else "FAIL"
    print(f"  {gesture_dir}: {correct}/{correct+wrong} correct ({accuracy:.0f}%) [{status}]", end="")
    if wrong_as:
        print(f" | Misclassified as: {wrong_as}", end="")
    print()

    # Show top3 for first sample
    seq = np.load(os.path.join(gesture_path, npy_files[0]))
    probs = model.predict(seq.reshape(1, 30, 1530), verbose=0)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(str(classes[i]), f"{probs[i]*100:.1f}%") for i in top3_idx]
    print(f"    Sample seq_0 top3: {top3}")

overall = correct_total / (correct_total + wrong_total) * 100 if (correct_total + wrong_total) > 0 else 0
print(f"\nOVERALL: {correct_total}/{correct_total+wrong_total} ({overall:.1f}%)")
