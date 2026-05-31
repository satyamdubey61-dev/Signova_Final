"""
Signova Post-Collection Data Validation Script
===============================================
Verifies the integrity of collected gesture sequence data:
  - Each class has the expected number of sequences
  - Each .npy file has shape (30, 1530)
  - No all-zero (corrupted) sequences
  - No NaN values in landmark arrays

Usage:
    python scripts/validate_collection.py
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


def validate(expected_sequences=50, expected_shape=(30, 1530)):
    data_dir = os.path.join(BASE_DIR, "Data", "sequences")

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return False

    # Discover all class directories
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not classes:
        print("ERROR: No class directories found.")
        return False

    print("\n=======================================================")
    print("     SIGNOVA DATA VALIDATION REPORT                   ")
    print("=======================================================")
    print(f"Data Dir: {data_dir}")
    print(f"Expected sequences per class: {expected_sequences}")
    print(f"Expected shape per sequence: {expected_shape}")
    print(f"Discovered {len(classes)} classes: {classes}\n")

    all_passed = True
    total_sequences = 0
    total_errors = 0

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        npy_files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
        num_files = len(npy_files)
        total_sequences += num_files

        errors = []

        # Check sequence count
        if num_files != expected_sequences:
            errors.append(f"Expected {expected_sequences} sequences, found {num_files}")

        # Validate each file
        corrupted = 0
        wrong_shape = 0
        has_nan = 0
        all_zero = 0

        for npy_file in npy_files:
            filepath = os.path.join(cls_dir, npy_file)
            try:
                data = np.load(filepath)

                if data.shape != expected_shape:
                    wrong_shape += 1

                if np.all(data == 0):
                    all_zero += 1

                if np.isnan(data).any():
                    has_nan += 1

            except Exception as e:
                corrupted += 1

        if wrong_shape > 0:
            errors.append(f"{wrong_shape} files with incorrect shape")
        if all_zero > 0:
            errors.append(f"{all_zero} all-zero (empty) sequences")
        if has_nan > 0:
            errors.append(f"{has_nan} files with NaN values")
        if corrupted > 0:
            errors.append(f"{corrupted} corrupted/unreadable files")

        total_errors += len(errors)

        # Report
        if errors:
            all_passed = False
            status = "FAIL"
            error_detail = " | ".join(errors)
        else:
            status = "PASS"
            error_detail = ""

        icon = "OK" if status == "PASS" else "XX"
        print(f"  {icon} [{status}] {cls:15s} -- {num_files:3d} sequences", end="")
        if error_detail:
            print(f"  WARNING: {error_detail}")
        else:
            print()

    print(f"\n-------------------------------------------------------")
    print(f"  Total classes:    {len(classes)}")
    print(f"  Total sequences:  {total_sequences}")
    print(f"  Total errors:     {total_errors}")
    print(f"  Overall status:   {'ALL PASSED' if all_passed else 'ISSUES FOUND'}")
    print(f"=======================================================\n")

    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
