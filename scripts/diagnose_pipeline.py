"""
Full diagnostic script for Signova LSTM pipeline.
Checks: dataset structure, label encoding, model architecture, prediction pipeline consistency.
"""
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def diagnose_dataset():
    section("STEP 1: DATASET STRUCTURE ANALYSIS")
    data_dir = os.path.join(BASE_DIR, "Data", "sequences")
    
    if not os.path.exists(data_dir):
        print(f"[FATAL] Data directory does not exist: {data_dir}")
        return False
    
    gesture_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Total gesture folders found: {len(gesture_dirs)}")
    print(f"Gesture folders: {gesture_dirs}")
    
    all_shapes = {}
    total_sequences = 0
    issues = []
    
    for gesture in gesture_dirs:
        gesture_path = os.path.join(data_dir, gesture)
        npy_files = sorted([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
        
        if len(npy_files) == 0:
            issues.append(f"[EMPTY] {gesture}: No .npy files found!")
            print(f"  {gesture}: 0 sequences [EMPTY!]")
            continue
        
        shapes = []
        nan_count = 0
        zero_count = 0
        
        for f in npy_files:
            try:
                data = np.load(os.path.join(gesture_path, f))
                shapes.append(data.shape)
                if np.isnan(data).any():
                    nan_count += 1
                if np.all(data == 0):
                    zero_count += 1
            except Exception as e:
                issues.append(f"[CORRUPT] {gesture}/{f}: {e}")
        
        unique_shapes = set(shapes)
        total_sequences += len(npy_files)
        all_shapes[gesture] = unique_shapes
        
        print(f"  {gesture}: {len(npy_files)} sequences | shapes: {unique_shapes} | NaN files: {nan_count} | All-zero files: {zero_count}")
        
        if len(unique_shapes) > 1:
            issues.append(f"[SHAPE MISMATCH] {gesture}: Multiple shapes found: {unique_shapes}")
    
    print(f"\nTotal sequences across all gestures: {total_sequences}")
    
    if issues:
        print(f"\n[ISSUES FOUND] ({len(issues)}):")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n[OK] No dataset issues found.")
    
    return len(issues) == 0

def diagnose_labels():
    section("STEP 2: LABEL ENCODING ANALYSIS")
    
    # Check what labels are saved
    labels_path = os.path.join(BASE_DIR, "models", "labels.npy")
    if not os.path.exists(labels_path):
        print(f"[FATAL] labels.npy not found at: {labels_path}")
        return
    
    saved_labels = np.load(labels_path)
    print(f"Saved labels (labels.npy): {list(saved_labels)}")
    print(f"Number of saved labels: {len(saved_labels)}")
    
    # Check what training would discover
    data_dir = os.path.join(BASE_DIR, "Data", "sequences")
    training_classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and any(f.endswith('.npy') for f in os.listdir(os.path.join(data_dir, d)))
    ])
    print(f"\nTraining-discovered classes (sorted folder names): {training_classes}")
    print(f"Number of training classes: {len(training_classes)}")
    
    # Check collection script class names
    collection_classes = ['Hello', 'Yes', 'No', 'Thank You', 'Please', 'Sorry', 'Help', 'Good', 'Bad', 'I Love You']
    collection_folder_names = [c.replace(" ", "") for c in collection_classes]
    print(f"\nCollection script classes: {collection_classes}")
    print(f"Collection folder names (spaces removed): {collection_folder_names}")
    
    # Compare
    print(f"\n--- COMPARISON ---")
    print(f"saved labels vs training classes MATCH: {list(saved_labels) == training_classes}")
    print(f"saved labels vs collection folders MATCH: {list(saved_labels) == collection_folder_names}")
    
    if list(saved_labels) != training_classes:
        print(f"\n[CRITICAL] LABEL MISMATCH DETECTED!")
        print(f"  Saved labels:    {list(saved_labels)}")
        print(f"  Training classes: {training_classes}")
        for i, (s, t) in enumerate(zip(saved_labels, training_classes)):
            if s != t:
                print(f"  Index {i}: saved='{s}' vs training='{t}'")

def diagnose_model():
    section("STEP 3: MODEL ARCHITECTURE ANALYSIS")
    
    model_path = os.path.join(BASE_DIR, "models", "lstm_model.h5")
    if not os.path.exists(model_path):
        print(f"[FATAL] Model not found at: {model_path}")
        return
    
    print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"Model last modified: {os.path.getmtime(model_path)}")
    
    import datetime
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
    print(f"Model last modified (readable): {mod_time}")
    
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        print(f"\nModel loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Number of output classes: {model.output_shape[-1]}")
        
        # Check if output classes match labels
        labels_path = os.path.join(BASE_DIR, "models", "labels.npy")
        if os.path.exists(labels_path):
            saved_labels = np.load(labels_path)
            if model.output_shape[-1] != len(saved_labels):
                print(f"\n[CRITICAL] MODEL OUTPUT ({model.output_shape[-1]}) != SAVED LABELS ({len(saved_labels)})")
            else:
                print(f"[OK] Model output classes ({model.output_shape[-1]}) matches saved labels ({len(saved_labels)})")
        
        model.summary()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

def diagnose_prediction_pipeline():
    section("STEP 4: PREDICTION PIPELINE ANALYSIS")
    
    # Check the prediction service configuration
    print("Checking prediction service configuration...")
    
    # Simulate what happens during prediction
    model_path = os.path.join(BASE_DIR, "models", "lstm_model.h5")
    labels_path = os.path.join(BASE_DIR, "models", "labels.npy")
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("[FATAL] Model or labels missing!")
        return
    
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    classes = np.load(labels_path)
    
    print(f"Loaded classes in prediction: {list(classes)}")
    print(f"Model expects input shape: {model.input_shape}")
    
    # Test with a random sequence
    print(f"\n--- Testing with random data ---")
    random_seq = np.random.randn(1, 30, 1530).astype(np.float32)
    probs = model.predict(random_seq, verbose=0)[0]
    print(f"Random input prediction probabilities: {probs}")
    print(f"Predicted class index: {np.argmax(probs)}")
    print(f"Predicted class: {classes[np.argmax(probs)]}")
    print(f"Max probability: {np.max(probs):.4f}")
    
    # Test with all zeros (simulating no hand)
    print(f"\n--- Testing with all-zeros data ---")
    zero_seq = np.zeros((1, 30, 1530), dtype=np.float32)
    probs_zero = model.predict(zero_seq, verbose=0)[0]
    print(f"All-zeros prediction probabilities: {probs_zero}")
    print(f"Predicted class: {classes[np.argmax(probs_zero)]}")
    print(f"Max probability: {np.max(probs_zero):.4f}")
    
    # Test with actual data from each class
    print(f"\n--- Testing with actual training data ---")
    data_dir = os.path.join(BASE_DIR, "Data", "sequences")
    for gesture_dir in sorted(os.listdir(data_dir)):
        gesture_path = os.path.join(data_dir, gesture_dir)
        if not os.path.isdir(gesture_path):
            continue
        
        npy_files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
        if not npy_files:
            continue
        
        # Load first sequence
        seq = np.load(os.path.join(gesture_path, npy_files[0]))
        if seq.shape != (30, 1530):
            print(f"  {gesture_dir}: SKIPPED (shape {seq.shape})")
            continue
        
        seq_input = seq.reshape(1, 30, 1530)
        probs = model.predict(seq_input, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(classes[i], f"{probs[i]*100:.1f}%") for i in top3_idx]
        
        correct = "✓" if pred_class == gesture_dir else "✗"
        print(f"  {gesture_dir}: Predicted={pred_class} ({probs[pred_idx]*100:.1f}%) {correct} | Top3: {top3}")

def diagnose_stability_config():
    section("STEP 5: STABILITY SYSTEM CONFIGURATION")
    
    print("Current prediction service stability settings:")
    print(f"  min_confidence: 0.85 (85%)")
    print(f"  sequence_length: 30")
    print(f"  History window: 15 predictions")
    print(f"  Majority threshold: 8/15 (53%)")
    print(f"  Consecutive lock requirement: 10 frames")
    print(f"  Freeze duration: 1.5 seconds")
    print(f"  Micro-movement threshold: 0.015")
    print(f"  Prediction rate: every 3rd frame")
    
    print(f"\n--- POTENTIAL ISSUES ---")
    print(f"  [WARN] min_confidence=0.85 is very high - may filter out valid predictions")
    print(f"  [WARN] Consecutive requirement=10 means ~30+ frames needed to show prediction")
    print(f"  [WARN] With rate downsampling (1/3), effective consecutive = 30+ raw frames")
    print(f"  [WARN] Majority 8/15 + consecutive 10 = extremely high bar for showing prediction")
    print(f"  [WARN] Micro-movement threshold may cause stale predictions to persist")

if __name__ == "__main__":
    diagnose_dataset()
    diagnose_labels()
    diagnose_model()
    diagnose_prediction_pipeline()
    diagnose_stability_config()
