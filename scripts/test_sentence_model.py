"""
test_sentence_model.py v2 — Evaluate the trained sentence LSTM model.

OPTIMIZED:
  - Updated SEQUENCE_LENGTH = 30, FEATURE_SIZE = 195
  - Shows per-class confidence histogram
  - Live webcam test uses gesture-end detection
  - Displays top-3 predictions live during webcam test
  - Reports inference latency per sample

DO NOT confuse with test_model_accuracy.py (word/alphabet mode).
"""
import os
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.logger import logger


SEQUENCE_LENGTH = 30    # optimized (was 50)
FEATURE_SIZE    = 195   # optimized (was 225)

# Gesture-end detection thresholds (must match sentence_prediction_service.py)
VELOCITY_STILL_THRESHOLD   = 0.006
MIN_FRAMES_FOR_EARLY       = 15
VELOCITY_WINDOW            = 5


def load_dataset(data_dir: str, classes: list) -> tuple:
    sequences, labels = [], []
    label_map = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files  = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
        loaded = 0
        for fname in files:
            try:
                arr = np.load(os.path.join(cls_dir, fname))
                # Accept both formats during transition
                if arr.ndim == 2 and arr.shape[1] == FEATURE_SIZE:
                    if arr.shape[0] >= SEQUENCE_LENGTH:
                        arr = arr[:SEQUENCE_LENGTH]
                    else:
                        continue  # too short
                else:
                    continue
                if np.all(arr == 0):
                    continue
                sequences.append(arr)
                labels.append(label_map[cls])
                loaded += 1
            except Exception:
                pass
        print(f"  '{cls}': {loaded}/{len(files)} sequences loaded")

    return np.array(sequences, dtype=np.float32), np.array(labels)


def main() -> None:
    models_dir = os.path.join(BASE_DIR, "models")
    data_dir   = os.path.join(BASE_DIR, "Data", "sentence_sequences")

    model_path  = os.path.join(models_dir, "sentence_lstm_model.h5")
    labels_path = os.path.join(models_dir, "sentence_labels.npy")

    print("\n" + "=" * 64)
    print("   SIGNOVA SENTENCE MODEL TEST v2                        ")
    print("=" * 64)

    if not os.path.exists(model_path):
        print(f"[ERROR] sentence_lstm_model.h5 not found at: {model_path}")
        print("Run scripts/train_sentence_model.py first.")
        return
    if not os.path.exists(labels_path):
        print(f"[ERROR] sentence_labels.npy not found at: {labels_path}")
        return

    from tensorflow.keras.models import load_model  # type: ignore
    print(f"\nLoading model: {model_path}")
    model   = load_model(model_path)
    classes = list(np.load(labels_path))
    print(f"Classes ({len(classes)}): {classes}")
    model.summary()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading sequences from: {data_dir}")
    X, y_int = load_dataset(data_dir, classes)

    if len(X) == 0:
        print("[ERROR] No data found. Run collect_sentence_sequences.py first.")
        return

    from tensorflow.keras.utils import to_categorical  # type: ignore
    y = to_categorical(y_int, num_classes=len(classes)).astype(np.float32)

    _, X_test, _, y_test, _, y_test_int = train_test_split(
        X, y, y_int, test_size=0.20, random_state=42, stratify=y_int
    )
    print(f"\nTest set: {X_test.shape}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nRunning inference on test set...")
    t0    = time.time()
    y_pred = model.predict(X_test, verbose=1)
    elapsed = time.time() - t0
    n_test  = len(X_test)
    print(
        f"Inference: {elapsed:.2f}s total | "
        f"{elapsed / n_test * 1000:.1f}ms/sample | "
        f"{n_test / elapsed:.1f} samples/sec"
    )

    y_pred_cls = np.argmax(y_pred, axis=1).tolist()
    y_true_cls = y_test_int.tolist()

    print("\n" + "=" * 64)
    print("CONFUSION MATRIX:")
    print("=" * 64)
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    print(cm)

    print("\n" + "=" * 64)
    print("CLASSIFICATION REPORT:")
    print("=" * 64)
    print(classification_report(y_true_cls, y_pred_cls, target_names=classes))

    # Per-class accuracy
    print("PER-CLASS ACCURACY:")
    for i, cls in enumerate(classes):
        cls_mask = [j for j, t in enumerate(y_true_cls) if t == i]
        if cls_mask:
            correct = sum(1 for j in cls_mask if y_pred_cls[j] == i)
            acc     = correct / len(cls_mask) * 100
            print(f"  {cls:<20s} : {acc:.1f}%  ({correct}/{len(cls_mask)})")

    overall = sum(1 for p, t in zip(y_pred_cls, y_true_cls) if p == t) / len(y_true_cls) * 100
    print(f"\nOverall Accuracy : {overall:.2f}%")

    max_probs = np.max(y_pred, axis=1)
    print(f"Avg Confidence   : {np.mean(max_probs) * 100:.1f}%")
    print(f"Min Confidence   : {np.min(max_probs) * 100:.1f}%")
    print(f"Max Confidence   : {np.max(max_probs) * 100:.1f}%")

    # Confidence histogram per class
    print("\nPER-CLASS CONFIDENCE (avg):")
    for i, cls in enumerate(classes):
        cls_mask = [j for j, t in enumerate(y_true_cls) if t == i]
        if cls_mask:
            avg_conf = np.mean([max_probs[j] for j in cls_mask]) * 100
            print(f"  {cls:<20s} : {avg_conf:.1f}%")

    print("=" * 64)

    print("\nWould you like to run a live webcam test? (y/n)")
    answer = input("> ").strip().lower()
    if answer == "y":
        print("\nLaunching live webcam test. Press [Q] or ESC to exit.")
        run_live_test(model, classes)


def run_live_test(model, classes: list) -> None:
    """Live webcam verification with gesture-end detection and top-3 display."""
    import cv2
    from collections import deque, Counter
    from services.sentence_mediapipe_service import SentenceMediaPipeService

    mp_svc = SentenceMediaPipeService(model_complexity=0)
    cap    = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    buf:            deque = deque(maxlen=SEQUENCE_LENGTH)
    vote_window:    deque = deque(maxlen=5)
    still_count:    int   = 0
    early_triggered: bool = False

    cv2.namedWindow("Sentence Model Live Test v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sentence Model Live Test v2", 960, 540)

    print("Controls: [Q] or ESC to exit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        feats, results          = mp_svc.extract_features(frame)
        has_hands, velocity     = mp_svc.get_motion_state(results)
        mp_svc.draw_landmarks(frame, results)

        if has_hands:
            buf.append(feats)

        # Track stillness
        if velocity < VELOCITY_STILL_THRESHOLD:
            still_count += 1
        else:
            still_count     = 0
            early_triggered = False

        gesture_ended = (
            not early_triggered
            and len(buf) >= MIN_FRAMES_FOR_EARLY
            and still_count >= VELOCITY_WINDOW
        )
        if gesture_ended:
            early_triggered = True

        label    = "Collecting frames..."
        top3_str = ""
        conf_pct = 0.0

        buf_ready = len(buf) == SEQUENCE_LENGTH or gesture_ended

        if buf_ready and len(buf) >= MIN_FRAMES_FOR_EARLY:
            frames = list(buf)
            if len(frames) < SEQUENCE_LENGTH:
                last = frames[-1]
                while len(frames) < SEQUENCE_LENGTH:
                    frames.append(last)

            seq      = np.array([frames], dtype=np.float32)
            t0       = time.time()
            probs    = model.predict(seq, verbose=0)[0]
            inf_ms   = (time.time() - t0) * 1000

            idx      = int(np.argmax(probs))
            conf_pct = float(probs[idx]) * 100
            vote_window.append(classes[idx])

            ctr         = Counter(vote_window)
            top, cnt    = ctr.most_common(1)[0]
            threshold   = 3 if gesture_ended else 4

            if cnt >= threshold and conf_pct >= 80:
                label = f"{top}  ({conf_pct:.0f}%)"
            else:
                label = f"Analyzing...  ({conf_pct:.0f}%)"

            # Top-3 string
            top3_idx = np.argsort(probs)[::-1][:3]
            top3_str = "  |  ".join(
                f"{classes[i]}: {probs[i]*100:.0f}%"
                for i in top3_idx
            )

            if gesture_ended:
                label = f"⚡ {label} [EARLY]"

        # ── Draw overlay ──────────────────────────────────────────────────────
        overlay_h = 140 if top3_str else 80
        cv2.rectangle(frame, (0, h - overlay_h), (w, h), (8, 10, 20), -1)

        cv2.putText(frame, label, (15, h - overlay_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 211, 238), 2, cv2.LINE_AA)

        if top3_str:
            cv2.putText(frame, "Top-3:", (15, h - overlay_h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, top3_str, (15, h - overlay_h + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)

        stats = (
            f"Buf: {len(buf)}/{SEQUENCE_LENGTH}  "
            f"vel: {velocity:.4f}  "
            f"still: {still_count}  "
            f"early: {early_triggered}"
        )
        cv2.putText(frame, stats, (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 140), 1, cv2.LINE_AA)

        # Confidence bar
        if conf_pct > 0:
            bar_w = int((conf_pct / 100) * (w - 30))
            cv2.rectangle(frame, (15, h - overlay_h + 95), (15 + bar_w, h - overlay_h + 103),
                          (34, 211, 238), -1)

        cv2.imshow("Sentence Model Live Test v2", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
