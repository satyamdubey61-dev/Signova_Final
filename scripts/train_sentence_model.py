"""
train_sentence_model.py v2 — SEPARATE training script for sentence-level LSTM.

OPTIMIZED vs v1:
  - SEQUENCE_LENGTH: 50 → 30  (faster inference, lower latency)
  - FEATURE_SIZE: 225 → 195   (upper-body-only pose)
  - Model architecture: smaller + faster (64-64-32 LSTM with BatchNorm)
  - Stronger data augmentation: noise, time-shift, scale variation
  - Confusion matrix saved as PNG
  - Saves both sentence_lstm_model.h5 and best_sentence_model.h5

DO NOT confuse with train_lstm_model.py (word/alphabet mode).

Architecture (v2):
  Input (30, 195)
  → LSTM(64, return_sequences=True) + BatchNormalization
  → Dropout(0.3)
  → LSTM(64, return_sequences=True) + BatchNormalization
  → Dropout(0.3)
  → LSTM(32, return_sequences=False) + BatchNormalization
  → Dropout(0.5)
  → Dense(64, relu) + BatchNormalization
  → Dense(N_CLASSES, softmax)

Reads from:   Data/sentence_sequences/
Saves to:     models/sentence_lstm_model.h5
              models/best_sentence_model.h5
              models/sentence_labels.npy
              models/sentence_labels.txt
              models/confusion_matrix.png
"""
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.logger import logger


# ── Configuration ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 30     # must match collect_sentence_sequences.py v2
FEATURE_SIZE     = 195    # 63 left + 63 right + 69 upper-pose (23 × 3)
MAX_EPOCHS       = 200
BATCH_SIZE       = 32     # was 16 — better gradient estimates
VALIDATION_SPLIT = 0.20


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment_sequences(sequences: list, labels: list) -> tuple:
    """Apply multiple augmentation strategies for better generalization."""
    aug_seqs, aug_lbls = [], []

    for seq, lbl in zip(sequences, labels):
        arr = np.array(seq, dtype=np.float32)

        # 1. Gaussian noise
        noise = np.random.normal(0, 0.012, arr.shape).astype(np.float32)
        aug_seqs.append(arr + noise)
        aug_lbls.append(lbl)

        # 2. Temporal shift (shift sequence by 1–3 frames, pad with first/last)
        shift = np.random.randint(1, 4)
        shifted = np.concatenate([arr[shift:], np.tile(arr[-1:], (shift, 1))], axis=0)
        aug_seqs.append(shifted.astype(np.float32))
        aug_lbls.append(lbl)

        # 3. Scale variation (random scale 0.85..1.15)
        scale = np.random.uniform(0.88, 1.12)
        aug_seqs.append((arr * scale).astype(np.float32))
        aug_lbls.append(lbl)

        # 4. Joint dropout simulation (randomly zero out some features per frame)
        dropped = arr.copy()
        for fi in range(len(dropped)):
            if np.random.random() < 0.15:   # 15% chance to zero one hand segment
                hand_to_drop = np.random.choice([0, 1])  # 0=left, 1=right
                start = hand_to_drop * 63
                dropped[fi, start:start + 63] = 0.0
        aug_seqs.append(dropped.astype(np.float32))
        aug_lbls.append(lbl)

    return aug_seqs, aug_lbls


def main() -> None:
    data_dir   = os.path.join(BASE_DIR, "Data", "sentence_sequences")
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        logger.error(
            f"Sentence sequence directory not found: {data_dir}\n"
            "Run scripts/collect_sentence_sequences.py first."
        )
        return

    # ── Discover classes ─────────────────────────────────────────────────────
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith("_")
        and any(f.endswith(".npy") for f in os.listdir(os.path.join(data_dir, d)))
    ])

    if not classes:
        logger.error("No class directories with .npy files found.")
        return

    print("\n" + "=" * 64)
    print("   SIGNOVA SENTENCE MODEL TRAINING v2 (OPTIMIZED PIPELINE)   ")
    print("=" * 64)
    print(f"Data dir       : {data_dir}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Sequence length : {SEQUENCE_LENGTH}  (was 50)")
    print(f"Feature size    : {FEATURE_SIZE}  (was 225)")

    label_map = {label: idx for idx, label in enumerate(classes)}

    # ── Load sequences ────────────────────────────────────────────────────────
    sequences, labels = [], []
    skipped = {"zero": 0, "shape": 0, "error": 0, "nan": 0}

    for label in classes:
        class_path = os.path.join(data_dir, label)
        files      = [f for f in os.listdir(class_path) if f.endswith(".npy")]
        loaded     = 0

        for fname in files:
            fpath = os.path.join(class_path, fname)
            try:
                arr = np.load(fpath)
            except Exception as exc:
                logger.error(f"Error loading {fname}: {exc}")
                skipped["error"] += 1
                continue

            # Shape check — accept both old (50, 225) and new (30, 195)
            if arr.shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
                pass  # correct new format
            elif arr.ndim == 2 and arr.shape[1] == FEATURE_SIZE and arr.shape[0] >= SEQUENCE_LENGTH:
                # Trim to SEQUENCE_LENGTH
                arr = arr[:SEQUENCE_LENGTH]
            elif arr.ndim == 2 and arr.shape[1] != FEATURE_SIZE:
                logger.warning(
                    f"Skipping {fname}: feature size {arr.shape[1]} ≠ {FEATURE_SIZE} "
                    "(collected with old settings — recollect data)"
                )
                skipped["shape"] += 1
                continue
            else:
                logger.warning(f"Wrong shape {arr.shape} for {fname} — skipping")
                skipped["shape"] += 1
                continue

            if np.all(arr == 0):
                skipped["zero"] += 1
                continue

            if np.isnan(arr).any() or np.isinf(arr).any():
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
                skipped["nan"] += 1   # count but still use after fixing

            sequences.append(arr)
            labels.append(label_map[label])
            loaded += 1

        print(f"  '{label}': {loaded}/{len(files)} sequences loaded")

    if not sequences:
        logger.error("No valid sequences loaded. Aborting training.")
        return

    print(f"\nData summary:")
    print(f"  Total valid  : {len(sequences)}")
    print(f"  Skipped zero : {skipped['zero']}")
    print(f"  Skipped shape: {skipped['shape']}")
    print(f"  Skipped error: {skipped['error']}")
    print(f"  Fixed NaN    : {skipped['nan']}")

    # ── Augmentation ─────────────────────────────────────────────────────────
    original_count = len(sequences)
    aug_seqs, aug_lbls = augment_sequences(sequences, labels)
    sequences.extend(aug_seqs)
    labels.extend(aug_lbls)
    print(
        f"\n[AUGMENTATION] Original: {original_count} | "
        f"Added: {len(aug_seqs)} | Total: {len(sequences)}"
    )

    X     = np.array(sequences, dtype=np.float32)   # (N, 30, 195)
    y_int = np.array(labels)

    from tensorflow.keras.utils import to_categorical  # type: ignore[import-untyped]
    y = to_categorical(y_int, num_classes=len(classes)).astype(np.float32)

    print(f"\nX shape : {X.shape}")
    print(f"y shape : {y.shape}")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, lbl_train, lbl_test = train_test_split(
        X, y, y_int, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_int
    )
    print(f"Train   : {X_train.shape}  Test : {X_test.shape}")

    # ── Class weights ─────────────────────────────────────────────────────────
    unique   = np.unique(lbl_train)
    cw_arr   = compute_class_weight("balanced", classes=unique, y=lbl_train)
    class_weights = {int(k): float(v) for k, v in zip(unique, cw_arr)}
    print(f"Class weights: {class_weights}")

    # ── Build model v2 ────────────────────────────────────────────────────────
    import tensorflow as tf
    from tensorflow.keras.models import Sequential                   # type: ignore
    from tensorflow.keras.layers import (                            # type: ignore
        LSTM, Dense, Dropout, BatchNormalization, Input
    )

    model = Sequential(name="sentence_lstm_v2")
    model.add(Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)))

    # LSTM block 1
    model.add(LSTM(64, return_sequences=True, name="lstm_1"))
    model.add(BatchNormalization(name="bn_1"))
    model.add(Dropout(0.30, name="drop_1"))

    # LSTM block 2
    model.add(LSTM(64, return_sequences=True, name="lstm_2"))
    model.add(BatchNormalization(name="bn_2"))
    model.add(Dropout(0.30, name="drop_2"))

    # LSTM block 3
    model.add(LSTM(32, return_sequences=False, name="lstm_3"))
    model.add(BatchNormalization(name="bn_3"))
    model.add(Dropout(0.50, name="drop_3"))

    # Dense head
    model.add(Dense(64, activation="relu", name="dense_1"))
    model.add(BatchNormalization(name="bn_4"))
    model.add(Dense(len(classes), activation="softmax", name="output"))

    print("\nModel summary:")
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    from tensorflow.keras.callbacks import (  # type: ignore
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )

    best_model_path  = os.path.join(models_dir, "best_sentence_model.h5")
    final_model_path = os.path.join(models_dir, "sentence_lstm_model.h5")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=25,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_categorical_accuracy",
            save_best_only=True, mode="max", verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-6, verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[TRAINING] Starting sentence LSTM v2 (max {MAX_EPOCHS} epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    model.save(final_model_path)
    print(f"\n[SAVED] sentence_lstm_model.h5 → {final_model_path}")
    print(f"[SAVED] best_sentence_model.h5  → {best_model_path}")

    # ── Save labels ───────────────────────────────────────────────────────────
    labels_npy = os.path.join(models_dir, "sentence_labels.npy")
    labels_txt = os.path.join(models_dir, "sentence_labels.txt")
    np.save(labels_npy, np.array(classes))
    with open(labels_txt, "w") as f:
        for i, cls in enumerate(classes):
            f.write(f"{i}: {cls}\n")
    print(f"[SAVED] sentence_labels.npy  → {labels_npy}")
    print(f"[SAVED] sentence_labels.txt  → {labels_txt}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("                EVALUATION REPORT                          ")
    print("=" * 64)

    y_pred     = model.predict(X_test, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1).tolist()
    y_true_cls = np.argmax(y_test, axis=1).tolist()

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls, target_names=classes))

    val_acc  = history.history.get("val_categorical_accuracy", [0])[-1]
    val_loss = history.history.get("val_loss", [0])[-1]
    print(f"Final Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"Final Validation Loss     : {val_loss:.4f}")

    # Save confusion matrix as PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Sentence Model Confusion Matrix\nVal Acc: {val_acc*100:.1f}%")
        fig.tight_layout()
        cm_path = os.path.join(models_dir, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        print(f"\n[SAVED] confusion_matrix.png → {cm_path}")
    except ImportError:
        print("\n[INFO] matplotlib not installed — skipping confusion matrix PNG.")

    print("=" * 64)
    print("\nNext step: python scripts/test_sentence_model.py")
    print("Then:      python app.py  →  http://127.0.0.1:5000/sentence-mode")


if __name__ == "__main__":
    main()
