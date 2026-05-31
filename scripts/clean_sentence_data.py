"""
clean_sentence_data.py — Standalone data cleaning script for sentence sequences.

DO NOT confuse with anything in the word/alphabet pipeline.

What this script does:
  1. Scans ALL .npy files under Data/sentence_sequences/<ClassName>/
  2. Removes corrupted sequences:
     - All-zero arrays (no landmarks detected during recording)
     - Arrays with NaN or Inf values
     - Wrong shape (not matching EXPECTED_SHAPE)
     - Low-motion sequences (>80% of frames are identical/near-zero hands)
  3. Reports full per-class statistics before and after cleaning
  4. Does NOT delete files by default — run with --delete to actually remove them
     (default: moves bad files to Data/sentence_sequences/_quarantine/)

Usage:
    python scripts/clean_sentence_data.py           # dry run — inspect only
    python scripts/clean_sentence_data.py --delete  # permanently delete bad files
    python scripts/clean_sentence_data.py --move    # move bad files to _quarantine/
"""
import argparse
import os
import shutil
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ── Expected shape matches optimized collection settings ─────────────────────
# If you are still using old 50-frame data, set EXPECTED_SEQUENCE_LENGTH = 50.
# After recollection with new settings, this should be 30.
EXPECTED_SEQUENCE_LENGTH_V1 = 50   # old format
EXPECTED_SEQUENCE_LENGTH_V2 = 30   # new optimized format
EXPECTED_FEATURE_SIZE_V1    = 225  # old feature size
EXPECTED_FEATURE_SIZE_V2    = 195  # new upper-body pose feature size

# Low-motion threshold: if >80% of frames have hand features that are all-zero
LOW_MOTION_ZERO_RATIO = 0.80

QUARANTINE_DIR = os.path.join(BASE_DIR, "Data", "sentence_sequences", "_quarantine")


def classify_sequence(arr: np.ndarray) -> str:
    """
    Return a classification for a loaded .npy array:
      'ok'         — looks valid
      'all_zero'   — entire array is zeros
      'nan_inf'    — contains NaN or Inf
      'wrong_shape'— shape doesn't match any expected format
      'low_motion' — too many zero-hand frames (noisy/incomplete recording)
    """
    # Wrong shape check
    valid_shapes = [
        (EXPECTED_SEQUENCE_LENGTH_V1, EXPECTED_FEATURE_SIZE_V1),
        (EXPECTED_SEQUENCE_LENGTH_V2, EXPECTED_FEATURE_SIZE_V2),
        # Also allow any sequence length with valid feature sizes during transition
        (EXPECTED_SEQUENCE_LENGTH_V1, EXPECTED_FEATURE_SIZE_V2),
        (EXPECTED_SEQUENCE_LENGTH_V2, EXPECTED_FEATURE_SIZE_V1),
    ]
    if arr.ndim != 2 or arr.shape not in valid_shapes:
        return "wrong_shape"

    # All-zero check
    if np.all(arr == 0):
        return "all_zero"

    # NaN/Inf check
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return "nan_inf"

    # Low-motion check: examine the first 63 features (left hand)
    # A frame has no hands if both [0:63] and [63:126] are zero
    n_frames = arr.shape[0]
    zero_hand_frames = 0
    for frame in arr:
        left_hand  = frame[0:63]
        right_hand = frame[63:126]
        if np.all(left_hand == 0) and np.all(right_hand == 0):
            zero_hand_frames += 1

    if zero_hand_frames / n_frames > LOW_MOTION_ZERO_RATIO:
        return "low_motion"

    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean corrupted sentence sequences."
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Permanently delete bad files (cannot be undone)."
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move bad files to _quarantine/ directory (default behaviour)."
    )
    args = parser.parse_args()

    data_dir = os.path.join(BASE_DIR, "Data", "sentence_sequences")

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Sentence sequences directory not found: {data_dir}")
        print("Run scripts/collect_sentence_sequences.py first.")
        return

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("_")
    ])

    if not classes:
        print("[INFO] No class directories found.")
        return

    # Decide action
    action = "inspect"  # default: dry run
    if args.delete:
        action = "delete"
    elif args.move:
        action = "move"

    if action == "move":
        os.makedirs(QUARANTINE_DIR, exist_ok=True)

    print("\n" + "=" * 62)
    print("   SIGNOVA SENTENCE DATA CLEANER")
    print("=" * 62)
    print(f"Data dir : {data_dir}")
    print(f"Classes  : {classes}")
    print(f"Action   : {action.upper()}")
    if action == "inspect":
        print("  (Dry run — no files will be modified. Use --delete or --move to act.)")
    print("=" * 62)

    total_scanned = 0
    total_bad     = 0
    total_removed = 0

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files   = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])

        cls_ok    = 0
        cls_bad   = 0
        bad_files = []

        print(f"\n[CLASS] '{cls}'  ({len(files)} files)")

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                arr = np.load(fpath)
                status = classify_sequence(arr)
            except Exception as exc:
                status = f"load_error({exc})"

            total_scanned += 1

            if status == "ok":
                cls_ok += 1
            else:
                cls_bad   += 1
                total_bad += 1
                bad_files.append((fname, fpath, status))
                print(f"  [BAD] {fname}  [{status}]  shape={getattr(arr, 'shape', 'N/A')}")

        if not bad_files:
            print(f"  [OK] All {cls_ok} sequences are clean.")
        else:
            print(f"\n  Summary: {cls_ok} OK | {cls_bad} BAD")

            if action == "delete":
                for _, fpath, _ in bad_files:
                    os.remove(fpath)
                    total_removed += 1
                print(f"  [DELETED] {cls_bad} files removed.")
            elif action == "move":
                cls_quarantine = os.path.join(QUARANTINE_DIR, cls)
                os.makedirs(cls_quarantine, exist_ok=True)
                for fname, fpath, status in bad_files:
                    dest = os.path.join(cls_quarantine, f"{status}__{fname}")
                    shutil.move(fpath, dest)
                    total_removed += 1
                print(f"  [MOVED] {cls_bad} files -> {cls_quarantine}")

    print("\n" + "=" * 62)
    print("CLEANING COMPLETE")
    print(f"  Scanned : {total_scanned} files")
    print(f"  Bad     : {total_bad} files")
    print(f"  {'Removed' if action != 'inspect' else 'Would remove'}: {total_bad} files")
    if action == "inspect":
        print("\n  Run with --move or --delete to clean the data.")
    print("=" * 62)

    if total_bad == 0:
        print("\n[OK] Dataset is clean. Proceed to training.")
    else:
        print(f"\n[WARN] {total_bad} bad sequences found.")
        if action == "inspect":
            print("   Re-run with --move to quarantine them.")


if __name__ == "__main__":
    main()
