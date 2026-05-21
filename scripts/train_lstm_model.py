import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.logger import logger

def main():
    data_dir = os.path.join(BASE_DIR, "Data", "sequences")
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        logger.error(f"Sequences directory does not exist at: {data_dir}. Please collect sequence data first.")
        return

    # 1. Dynamically discover classes that contain sequences
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and any(f.endswith('.npy') for f in os.listdir(os.path.join(data_dir, d)))
    ])
    if not classes:
        logger.error(f"No active gesture sequence folders containing data found under {data_dir}.")
        return

    print("\n=======================================================")
    print("           SIGNOVA LSTM MODEL TRAINING PIPELINE        ")
    print("=======================================================")
    print(f"Discovered {len(classes)} classes: {classes}")

    # Map classes to integer indices
    label_map = {label: num for num, label in enumerate(classes)}

    sequences, labels = [], []
    skipped_zero = 0
    skipped_shape = 0
    skipped_error = 0

    # 2. Load sequence files with safety NaN cleaning and validation
    for label in classes:
        class_path = os.path.join(data_dir, label)
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        loaded_count = 0
        
        for file in npy_files:
            try:
                res = np.load(os.path.join(class_path, file))
                # Safety checks for shapes
                if res.shape != (30, 1530):
                    logger.warning(f"Skipping file {file} in {label} due to incorrect shape: {res.shape}")
                    skipped_shape += 1
                    continue
                # Skip all-zero sequences (corrupted / no landmarks detected)
                if np.all(res == 0):
                    logger.warning(f"Skipping all-zero sequence {file} in {label}")
                    skipped_zero += 1
                    continue
                if np.isnan(res).any():
                    res = np.nan_to_num(res, nan=0.0)
                sequences.append(res)
                labels.append(label_map[label])
                loaded_count += 1
            except Exception as load_err:
                logger.error(f"Failed to load sequence file {file}: {load_err}")
                skipped_error += 1

        print(f"Loading '{label}': {loaded_count}/{len(npy_files)} sequences loaded")

    if not sequences:
        logger.error("No valid sequences loaded. Cannot train model.")
        return

    if skipped_zero > 0 or skipped_shape > 0 or skipped_error > 0:
        print(f"\n[DATA CLEANING] Skipped: {skipped_zero} all-zero, {skipped_shape} wrong-shape, {skipped_error} errors")

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    print(f"\nDataset loaded successfully!")
    print(f"X shape: {X.shape} (samples, sequence_length, features)")
    print(f"y shape: {y.shape} (samples, one_hot_encoded_classes)")

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels)

    print(f"Train split: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test split: X_test={X_test.shape}, y_test={y_test.shape}")

    # 4. Build Robust Bidirectional LSTM Neural Network Model with BatchNormalization
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=(30, 1530)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=False, activation='tanh')),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(len(classes), activation='softmax')
    ])

    print("\nModel Summary:")
    model.summary()

    # 5. Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    # 6. Setup Callbacks (Early Stopping, Model Checkpointing, and ReduceLROnPlateau)
    checkpoint_path = os.path.join(models_dir, "lstm_model.h5")
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=25, 
            restore_best_weights=True, 
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path, 
            monitor='val_categorical_accuracy', 
            save_best_only=True, 
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 7. Train Model
    print("\nStarting model training...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=120, 
        batch_size=16, 
        validation_data=(X_test, y_test), 
        callbacks=callbacks
    )

    print(f"\nModel training finished! Best model saved to: {checkpoint_path}")

    # 8. Save Class Labels
    labels_path = os.path.join(models_dir, "labels.npy")
    np.save(labels_path, np.array(classes))
    print(f"Saved labels list to: {labels_path}")

    # 9. Evaluate Model Performance
    print("\n=======================================================")
    print("                  EVALUATION REPORT                    ")
    print("=======================================================")

    # Predictions
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    # Confusion Matrix & Classification Report
    cm = confusion_matrix(ytrue, yhat)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(ytrue, yhat, target_names=classes))

    # Print final validation metrics
    val_loss = history.history.get('val_loss', [-1])[-1]
    val_acc = history.history.get('val_categorical_accuracy', [-1])[-1]
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
