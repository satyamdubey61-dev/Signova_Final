"""
collect_sentence_sequences.py v2 — SEPARATE data collection for sentence mode.

OPTIMIZED SETTINGS:
  - FRAMES_PER_SEQUENCE: 50 → 30  (matches new model sequence length)
  - SEQUENCES_PER_CLASS: 50 → 60  (20% more data per class)
  - FEATURE_SIZE: 225 → 195       (upper-body-only pose)
  - Added: motion quality check — skips saving if <8 frames have hand detected
  - Added: live motion velocity bar during recording
  - Added: auto-cleanup of corrupted sequences on startup
  - Added: faster 2s countdown (was 3s × 0.8s = 2.4s per step)
  - Added: clear quality indicators on screen

DO NOT confuse with collect_sequences.py (word/alphabet mode).

Sentence Classes (4):
  1. HelloHowAreYou  →  "Hello How Are You"
  2. IAmFine         →  "I Am Fine"
  3. ThankYou        →  "Thank You"
  4. INeedHelp       →  "I Need Help"

Data saved to:
  Data/sentence_sequences/<ClassName>/seq_N.npy
  Shape of each file: (30, 195)
"""
import os
import sys
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.sentence_mediapipe_service import SentenceMediaPipeService, SENTENCE_FEATURE_SIZE
from utils.logger import logger


# ── Configuration ──────────────────────────────────────────────────────────────
SENTENCE_CLASSES = [
    "HelloHowAreYou",
    "IAmFine",
    "ThankYou",
    "INeedHelp",
]

DISPLAY_NAMES = {
    "HelloHowAreYou": "Hello How Are You",
    "IAmFine":        "I Am Fine",
    "ThankYou":       "Thank You",
    "INeedHelp":      "I Need Help",
}

SEQUENCES_PER_CLASS  = 60   # was 50 — 20% more data
FRAMES_PER_SEQUENCE  = 30   # was 50 — matches new model
MIN_HAND_FRAMES      = 8    # minimum frames with hand detected for valid sequence
DATA_DIR = os.path.join(BASE_DIR, "Data", "sentence_sequences")


# ── Drawing Helpers ────────────────────────────────────────────────────────────

def draw_header(
    frame: np.ndarray,
    display_name: str,
    class_idx: int,
    total_classes: int,
    seq_idx: int,
    total_seqs: int,
    status: str,
    status_color: tuple,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 110), (8, 10, 20), -1)

    cv2.putText(
        frame,
        f"SIGNOVA SENTENCE COLLECTOR  [{class_idx + 1}/{total_classes}]",
        (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (34, 211, 238), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f'Sentence: "{display_name}"',
        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Seq {seq_idx + 1}/{total_seqs}  |  {status}",
        (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Frames/seq: {FRAMES_PER_SEQUENCE}  |  Feature size: {SENTENCE_FEATURE_SIZE}",
        (15, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 120), 1, cv2.LINE_AA,
    )
    return frame


def draw_recording_bar(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    hand_count: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_y = h - 30

    # Progress bar background
    cv2.rectangle(frame, (0, bar_y), (w, h), (8, 10, 20), -1)
    bar_w = int((frame_idx / total_frames) * w)

    # Color based on quality
    bar_color = (74, 222, 128) if hand_count >= MIN_HAND_FRAMES else (255, 165, 0)
    cv2.rectangle(frame, (0, bar_y), (bar_w, bar_y + 10), bar_color, -1)

    # Labels
    cv2.putText(
        frame,
        f"Frame {frame_idx}/{total_frames}   Hands detected: {hand_count}",
        (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA,
    )
    quality_txt = "GOOD QUALITY" if hand_count >= MIN_HAND_FRAMES else "SHOW YOUR HANDS!"
    quality_col = (74, 222, 128) if hand_count >= MIN_HAND_FRAMES else (255, 80, 80)
    cv2.putText(
        frame,
        quality_txt,
        (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, quality_col, 2, cv2.LINE_AA,
    )
    return frame


def draw_velocity_meter(
    frame: np.ndarray,
    velocity: float,
    mp_svc: SentenceMediaPipeService,
) -> np.ndarray:
    """Draw a small motion velocity bar in top-right corner."""
    h, w = frame.shape[:2]
    bar_x, bar_y, bar_h, bar_max_w = w - 160, 120, 14, 140

    cv2.putText(frame, "MOTION:", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 180), 1, cv2.LINE_AA)

    # Clamp velocity to display range
    v_norm = min(velocity / 0.05, 1.0)
    vel_w  = int(v_norm * bar_max_w)
    vel_col = (
        (74, 222, 128) if v_norm > 0.3
        else (255, 165, 0) if v_norm > 0.05
        else (100, 100, 120)
    )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (30, 30, 40), -1)
    if vel_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + vel_w, bar_y + bar_h), vel_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (60, 60, 80), 1)
    return frame


def draw_countdown(frame: np.ndarray, count: int) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(
        frame, str(count),
        (w // 2 - 35, h // 2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (239, 68, 68), 8, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "GET READY — SIGN THE SENTENCE",
        (w // 2 - 220, h // 2 + 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (239, 68, 68), 2, cv2.LINE_AA,
    )
    return frame


def auto_cleanup_bad_sequences(data_dir: str) -> None:
    """Remove all-zero and wrong-shape sequences on startup."""
    removed = 0
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir) or cls.startswith("_"):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(cls_dir, fname)
            try:
                arr = np.load(fpath)
                if np.all(arr == 0) or arr.ndim != 2:
                    os.remove(fpath)
                    removed += 1
                    print(f"  [AUTO-CLEAN] Removed corrupted: {cls}/{fname}")
            except Exception:
                os.remove(fpath)
                removed += 1
                print(f"  [AUTO-CLEAN] Removed unreadable: {cls}/{fname}")
    if removed:
        print(f"[AUTO-CLEAN] Removed {removed} corrupted sequences.\n")


# ── Main Collection Loop ───────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in SENTENCE_CLASSES:
        os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

    print("\n" + "=" * 62)
    print("   SIGNOVA SENTENCE SEQUENCE COLLECTOR v2 (OPTIMIZED)   ")
    print("=" * 62)
    print(f"Data directory   : {DATA_DIR}")
    print(f"Classes          : {SENTENCE_CLASSES}")
    print(f"Sequences/class  : {SEQUENCES_PER_CLASS}")
    print(f"Frames/sequence  : {FRAMES_PER_SEQUENCE}  ← OPTIMIZED (was 50)")
    print(f"Feature size     : {SENTENCE_FEATURE_SIZE}  ← OPTIMIZED (was 225)")
    print(f"Min hand frames  : {MIN_HAND_FRAMES} (quality gate)")
    print("-" * 62)
    print("Controls:")
    print("  [S] → Start collecting this class")
    print("  [N] → Skip to next class")
    print("  [Q] / [ESC] → Quit")
    print("=" * 62)

    # Auto-cleanup corrupted files before starting
    print("\n[AUTO-CLEAN] Checking for corrupted sequences...")
    auto_cleanup_bad_sequences(DATA_DIR)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS,          30)

    window_name = "Signova Sentence Collector v2"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 580)

    mp_service = SentenceMediaPipeService(
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
    )

    current_class_idx = 0

    while current_class_idx < len(SENTENCE_CLASSES):
        cls     = SENTENCE_CLASSES[current_class_idx]
        display = DISPLAY_NAMES[cls]
        cls_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)

        existing  = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
        start_seq = len(existing)

        if start_seq >= SEQUENCES_PER_CLASS:
            print(f"[SKIP] '{display}' already has {start_seq}/{SEQUENCES_PER_CLASS} sequences.")
            current_class_idx += 1
            continue

        print(f"\n[CLASS {current_class_idx + 1}/{len(SENTENCE_CLASSES)}] '{display}'")
        print(f"  Already collected: {start_seq}/{SEQUENCES_PER_CLASS}")
        print("  Press [S] to start collecting, [N] to skip, [Q] to quit.")

        collecting = False
        seq_idx    = start_seq

        while seq_idx < SEQUENCES_PER_CLASS:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            feats, results = mp_service.extract_features(frame)
            _, velocity    = mp_service.get_motion_state(results)
            mp_service.draw_landmarks(frame, results)

            draw_header(
                frame, display, current_class_idx, len(SENTENCE_CLASSES),
                seq_idx, SEQUENCES_PER_CLASS,
                "Press [S] to start" if not collecting else "READY",
                (34, 211, 238),
            )
            draw_velocity_meter(frame, velocity, mp_service)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                print("\n[QUIT] Collection stopped by user.")
                return
            if key == ord("n"):
                print(f"[SKIP] Skipping '{display}'")
                break
            if key == ord("s"):
                collecting = True

            if not collecting:
                continue

            # ── Collect sequences ────────────────────────────────────────────
            for s in range(seq_idx, SEQUENCES_PER_CLASS):
                # Countdown — 2 steps × 0.8s = 1.6s (faster than before)
                for count in range(2, 0, -1):
                    deadline = time.time() + 0.8
                    while time.time() < deadline:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = cv2.flip(frame, 1)
                        _, results_cd = mp_service.extract_features(frame)
                        mp_service.draw_landmarks(frame, results_cd)
                        draw_header(
                            frame, display, current_class_idx, len(SENTENCE_CLASSES),
                            s, SEQUENCES_PER_CLASS, "STARTING...", (255, 165, 0),
                        )
                        draw_countdown(frame, count)
                        cv2.imshow(window_name, frame)
                        cv2.waitKey(1)

                # Record FRAMES_PER_SEQUENCE frames
                sequence:   list = []
                hand_frames: int = 0

                for f_idx in range(FRAMES_PER_SEQUENCE):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    feats, results = mp_service.extract_features(frame)
                    has_hands, velocity = mp_service.get_motion_state(results)
                    if has_hands:
                        hand_frames += 1

                    sequence.append(feats)
                    mp_service.draw_landmarks(frame, results)

                    draw_header(
                        frame, display, current_class_idx, len(SENTENCE_CLASSES),
                        s, SEQUENCES_PER_CLASS, "● RECORDING", (74, 222, 128),
                    )
                    draw_recording_bar(frame, f_idx + 1, FRAMES_PER_SEQUENCE, hand_frames)
                    draw_velocity_meter(frame, velocity, mp_service)

                    # REC indicator
                    cv2.circle(frame, (frame.shape[1] - 30, 130), 12, (74, 222, 128), -1)
                    cv2.putText(frame, "REC", (frame.shape[1] - 55, 134),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (74, 222, 128), 1, cv2.LINE_AA)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)

                if len(sequence) == FRAMES_PER_SEQUENCE:
                    arr = np.array(sequence, dtype=np.float32)

                    # Quality gate
                    if hand_frames < MIN_HAND_FRAMES:
                        print(
                            f"  ⚠  Seq {s + 1}: only {hand_frames} hand frames "
                            f"(need ≥{MIN_HAND_FRAMES}) — SKIPPED (re-recording)"
                        )
                        # Pause to let user re-position
                        time.sleep(0.5)
                        continue  # don't increment seq_idx — retry same slot

                    save_path = os.path.join(cls_dir, f"seq_{s}.npy")
                    np.save(save_path, arr)
                    print(
                        f"  ✅ Saved seq {s + 1}/{SEQUENCES_PER_CLASS}  "
                        f"shape={arr.shape}  hands={hand_frames}/{FRAMES_PER_SEQUENCE}"
                    )
                else:
                    print(f"  [WARN] Incomplete seq {s} ({len(sequence)} frames) — skipping.")

                seq_idx = s + 1
                time.sleep(0.25)  # brief pause between sequences

            print(f"\n[DONE] Completed '{display}'!")
            break

        current_class_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 62)
    print("   SENTENCE COLLECTION COMPLETE!")
    print(f"   Data saved to: {DATA_DIR}")
    print(f"   Next step: python scripts/clean_sentence_data.py --move")
    print(f"   Then:      python scripts/train_sentence_model.py")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
