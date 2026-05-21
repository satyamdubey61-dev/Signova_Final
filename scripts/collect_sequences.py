import os
import sys
import time
import cv2
import numpy as np

# Add base directory to path so we can import services
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from services.mediapipe_service import mediapipe_service
from utils.logger import logger

def main():
    # 10 Unique Target ISL Gestures
    classes = [
        'Hello', 'Yes', 'No', 'Thank You', 'Please', 
        'Sorry', 'Help', 'Good', 'Bad', 'I Love You'
    ]

    # Configure parameters
    sequences_per_class = 40  # 40 actions per gesture class (exceeds 30 minimum requirement)
    sequence_length = 30      # 30 frames per sequence

    # Ensure Data folder exists
    data_dir = os.path.join(BASE_DIR, "Data", "sequences")
    os.makedirs(data_dir, exist_ok=True)

    # Initialize Camera
    logger.info("Initializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("Could not open webcam using CAP_DSHOW. Trying default camera index...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam completely.")
            print("\nError: Webcam not found or busy.")
            return

    # Window settings
    cv2.namedWindow("Signova LSTM Sequence Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Signova LSTM Sequence Collector", 854, 480)

    print("\n=======================================================")
    print("       SIGNOVA LSTM SEQUENCE COLLECTION UTILITY       ")
    print("=======================================================")
    print(f"Data Dir: {data_dir}")
    print("Instructions:")
    print("  - For each gesture, we will record 40 action sequences.")
    print("  - Each action sequence consists of 30 frames.")
    print("  - A 2-second countdown will prepare you before each action.")
    print("  - Press 's' to start collecting the active class.")
    print("  - Press 'n' to skip to the next class.")
    print("  - Press 'q' or ESC to quit.")
    print("=======================================================\n")

    current_class_idx = 0

    while current_class_idx < len(classes):
        target_class = classes[current_class_idx]
        class_dir = os.path.join(data_dir, target_class.replace(" ", ""))
        os.makedirs(class_dir, exist_ok=True)

        print(f"\n---> Ready to collect sequences for: '{target_class}' ({current_class_idx + 1}/{len(classes)})")
        
        collecting = False
        seq_idx = 0

        while seq_idx < sequences_per_class:
            success, frame = cap.read()
            if not success:
                print("Failed to grab camera frame. Exiting.")
                break

            # Mirror frame for intuitive view
            frame = cv2.flip(frame, 1)
            h_img, w_img, _ = frame.shape

            # Extract features for visualization
            _, results = mediapipe_service.extract_landmarks(frame)
            mediapipe_service.draw_landmarks(frame, results)

            # Header info overlay
            cv2.rectangle(frame, (0, 0), (w_img, 65), (21, 16, 16), -1)
            cv2.putText(frame, f"GESTURE: {target_class} ({current_class_idx + 1}/{len(classes)})", 
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            status_text = f"Sequence {seq_idx + 1}/{sequences_per_class} | Press 's' to START, 'n' to skip, 'q' to quit"
            cv2.putText(frame, status_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 211, 238), 2, cv2.LINE_AA)

            cv2.imshow("Signova LSTM Sequence Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("Quitting sequence collector.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                print(f"Skipping class: '{target_class}'")
                break
            elif key == ord('s'):
                # Start collecting sequences
                collecting = True

            if collecting:
                # Loop through and record all 40 sequences
                for s in range(seq_idx, sequences_per_class):
                    # 1. 2-second prepare countdown
                    for countdown in range(3, 0, -1):
                        start_t = time.time()
                        while time.time() - start_t < 0.7:
                            success, frame = cap.read()
                            if not success:
                                break
                            frame = cv2.flip(frame, 1)
                            
                            # Draw overlays
                            _, results = mediapipe_service.extract_landmarks(frame)
                            mediapipe_service.draw_landmarks(frame, results)
                            
                            cv2.rectangle(frame, (0, 0), (w_img, 65), (21, 16, 16), -1)
                            cv2.putText(frame, f"CLASS: {target_class} | SEQ {s+1}/{sequences_per_class}", 
                                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, f"GET READY... STARTING IN {countdown}", 
                                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (239, 68, 68), 2, cv2.LINE_AA)
                            
                            # Massive visual countdown in center
                            cv2.putText(frame, str(countdown), (w_img//2 - 20, h_img//2 + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (239, 68, 68), 5, cv2.LINE_AA)
                            
                            cv2.imshow("Signova LSTM Sequence Collector", frame)
                            cv2.waitKey(1)

                    # 2. Record 30-frame sequence
                    sequence = []
                    f_count = 0
                    while f_count < sequence_length:
                        success, frame = cap.read()
                        if not success:
                            break
                        frame = cv2.flip(frame, 1)

                        # Extract landmarks
                        feats, results = mediapipe_service.extract_landmarks(frame)
                        sequence.append(feats)
                        f_count += 1

                        # Draw stylized landmarks
                        mediapipe_service.draw_landmarks(frame, results)

                        # Overlay recording status
                        cv2.rectangle(frame, (0, 0), (w_img, 65), (21, 16, 16), -1)
                        cv2.putText(frame, f"RECORDING SEQ {s+1}/{sequences_per_class}", 
                                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (74, 222, 128), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Frame {f_count}/{sequence_length}", 
                                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (74, 222, 128), 2, cv2.LINE_AA)
                        
                        # Big green dot indicator
                        cv2.circle(frame, (w_img - 30, 30), 12, (74, 222, 128), -1)
                        
                        cv2.imshow("Signova LSTM Sequence Collector", frame)
                        cv2.waitKey(1)

                    # Save sequence as .npy file
                    npy_path = os.path.join(class_dir, f"seq_{s}.npy")
                    np.save(npy_path, np.array(sequence))
                    print(f"Saved: {npy_path} (shape: {np.array(sequence).shape})")

                    # Brief pause between recordings to reset posture
                    time.sleep(0.4)

                # Done collecting all sequences for this class
                print(f"\nCompleted collection for: '{target_class}'!")
                break

        current_class_idx += 1

    print("\n=======================================================")
    print("   SEQUENCE COLLECTION COMPLETE! All classes captured.  ")
    print(f"   Stored under: {data_dir}")
    print("=======================================================\n")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
