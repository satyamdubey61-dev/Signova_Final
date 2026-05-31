import os
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from collections import deque, Counter
from tensorflow.keras.models import load_model  # type: ignore[import-untyped]
from services.mediapipe_service import mediapipe_service
from services.speech_service import speech_service
from utils.logger import logger


class LSTMPredictionService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        min_confidence: float = 0.75,
        sequence_length: int = 30,
    ) -> None:
        """Initializes the LSTM sequence prediction service with tuned stability filters."""
        base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path: str = model_path or os.path.join(base_dir, "models", "lstm_model.h5")
        self.labels_path: str = labels_path or os.path.join(base_dir, "models", "labels.npy")
        self.min_confidence: float = min_confidence
        self.sequence_length: int = sequence_length

        self.model: Any = None
        self.classes: Optional[NDArray[Any]] = None

        # State tracking per session (key = session_id or 'default')
        self.session_buffers: Dict[str, Deque[NDArray[Any]]] = {}
        self.session_history: Dict[str, Deque[str]] = {}
        self.session_stable_label: Dict[str, Optional[str]] = {}
        self.session_consecutive: Dict[str, int] = {}
        self.session_last_candidate: Dict[str, Optional[str]] = {}
        self.session_no_hand_count: Dict[str, int] = {}

        # Stability system (tuned for responsiveness)
        self.session_freeze_until: Dict[str, float] = {}
        self.session_frame_counter: Dict[str, int] = {}
        self.session_last_result: Dict[str, Tuple[str, float]] = {}
        self.session_prev_confidence: Dict[str, float] = {}
        self.session_prev_features: Dict[str, Optional[NDArray[Any]]] = {}
        self.session_prev_raw_label: Dict[str, Optional[str]] = {}

        # Configurable stabilization and latency parameters
        self.stable_frames_required: int = 2
        self.prediction_window: int = 5
        self.freeze_duration: float = 0.2

        # Speech Cooldown per session
        self.session_last_spoken: Dict[str, Optional[str]] = {}
        self.session_last_spoken_time: Dict[str, float] = {}
        self.speech_cooldown: float = 1.0

        # Debug log throttle (log every Nth inference to avoid spam)
        self.session_debug_counter: Dict[str, int] = {}
        self._debug_log_interval: int = 5  # log every 5th inference

        self.load_model()

    def load_model(self) -> bool:
        """Attempts to load the trained LSTM model and class labels from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.labels_path):
            logger.warning(f"LSTM model or labels not found. Paths:\nModel: {self.model_path}\nLabels: {self.labels_path}")
            return False

        try:
            self.model = load_model(self.model_path)
            self.classes = np.load(self.labels_path)
            logger.info(
                f"LSTM model loaded successfully from {self.model_path}. "
                f"Classes ({len(self.classes)}): {list(self.classes)}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load LSTM model or labels: {e}")
            return False

    def get_or_create_session(self, session_id: str) -> Tuple[Deque[NDArray[Any]], Deque[str]]:
        """Helper to fetch or instantiate session-specific state buffers."""
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = deque(maxlen=self.sequence_length)
            self.session_history[session_id] = deque(maxlen=self.prediction_window)
            self.session_stable_label[session_id] = None
            self.session_consecutive[session_id] = 0
            self.session_last_candidate[session_id] = None
            self.session_no_hand_count[session_id] = 0

            self.session_freeze_until[session_id] = 0.0
            self.session_frame_counter[session_id] = 0
            self.session_last_result[session_id] = ("ANALYZING...", 0.0)
            self.session_prev_confidence[session_id] = 0.0
            self.session_prev_features[session_id] = None
            self.session_prev_raw_label[session_id] = None

            self.session_last_spoken[session_id] = None
            self.session_last_spoken_time[session_id] = 0.0

            self.session_debug_counter[session_id] = 0

        return (
            self.session_buffers[session_id],
            self.session_history[session_id]
        )

    def _format_display(self, label: str, confidence: float) -> str:
        """Formats label + confidence for display output."""
        if confidence > 0 and label not in ("ANALYZING...", "LOW CONFIDENCE", "NO HAND DETECTED", "MODEL ERROR"):
            return f"{label} ({confidence:.0f}%)"
        return label

    def predict(self, img_bgr: NDArray[Any], session_id: str = "default") -> Tuple[str, float]:
        """
        Runs prediction on a BGR image frame with tuned real-time stability.
        Returns:
            - label: text to display (e.g. "HELLO (92%)", "ANALYZING...", "NO HAND DETECTED").
            - confidence: float confidence score (0 to 100).
        """
        if self.model is None:
            if not self.load_model():
                return "MODEL ERROR", 0.0

        # Fetch/Create Session state
        buffer: Deque[NDArray[Any]]
        history: Deque[str]
        buffer, history = self.get_or_create_session(session_id)
        current_time: float = time.time()

        # 1. Check Output Freeze Timer
        if current_time < self.session_freeze_until.get(session_id, 0.0):
            frozen_label, frozen_conf = self.session_last_result[session_id]
            return self._format_display(frozen_label, frozen_conf), frozen_conf

        # 2. Extract landmarks (1530 features)
        features: NDArray[Any]
        features, _ = mediapipe_service.extract_landmarks(img_bgr)

        # Check if BOTH hands are missing
        left_hand_missing: bool = bool(np.all(features[0:63] == 0))
        right_hand_missing: bool = bool(np.all(features[63:126] == 0))

        # 3. No Hand Detection Logic & Gradual Clearing
        if left_hand_missing and right_hand_missing:
            self.session_no_hand_count[session_id] += 1

            # Gradually clear sequence buffer frame-by-frame
            if len(buffer) > 0:
                buffer.popleft()

            # If hands remain absent for >= 8 frames, hard-reset prediction state
            if self.session_no_hand_count[session_id] >= 8:
                buffer.clear()
                history.clear()
                self.session_stable_label[session_id] = None
                self.session_consecutive[session_id] = 0
                self.session_last_candidate[session_id] = None
                self.session_prev_features[session_id] = None
                self.session_prev_confidence[session_id] = 0.0
                self.session_prev_raw_label[session_id] = None
                self.session_last_result[session_id] = ("NO HAND DETECTED", 0.0)
                return "NO HAND DETECTED", 0.0

            # Tolerating temporary hand loss: return last valid cached result
            cached_label, cached_conf = self.session_last_result[session_id]
            return self._format_display(cached_label, cached_conf), cached_conf
        else:
            self.session_no_hand_count[session_id] = 0

        # 4. Micro-movement filter (reduced threshold for better responsiveness)
        prev_feats: Optional[NDArray[Any]] = self.session_prev_features.get(session_id)
        if prev_feats is not None:
            diff: float = float(np.mean(np.abs(features[0:126] - prev_feats[0:126])))
            if diff < 0.008:
                # Still append to buffer so sequence stays current, just skip inference
                buffer.append(features)
                cached_label, cached_conf = self.session_last_result[session_id]
                return self._format_display(cached_label, cached_conf), cached_conf

        self.session_prev_features[session_id] = features

        # Append landmark frame to sequence buffer
        buffer.append(features)

        # Wait for full 30-frame sequence
        if len(buffer) < self.sequence_length:
            self.session_last_result[session_id] = ("ANALYZING...", 0.0)
            return "ANALYZING...", 0.0

        # 5. Prediction Rate Downsampling (every 2nd frame instead of 3rd)
        self.session_frame_counter[session_id] += 1
        if self.session_frame_counter[session_id] % 2 != 0:
            cached_label, cached_conf = self.session_last_result[session_id]
            return self._format_display(cached_label, cached_conf), cached_conf

        try:
            # Prepare shape (1, 30, 1530) for LSTM input
            seq_data: NDArray[Any] = np.array([list(buffer)])

            # Predict probabilities
            probabilities: NDArray[Any] = self.model.predict(seq_data, verbose=0)[0]
            max_idx: int = int(np.argmax(probabilities))
            raw_label: str = str(self.classes[max_idx]) if self.classes is not None else "UNKNOWN"
            current_confidence: float = float(probabilities[max_idx]) * 100.0

            # --- DEBUG LOGGING: Top-3 predictions ---
            self.session_debug_counter[session_id] += 1
            if self.session_debug_counter[session_id] % self._debug_log_interval == 0:
                top3_idx = np.argsort(probabilities)[-3:][::-1]
                top3 = [(str(self.classes[i]), f"{probabilities[i]*100:.1f}%") for i in top3_idx]
                logger.info(
                    f"[PREDICT] session='{session_id}' | "
                    f"top3={top3} | "
                    f"seq_len={len(buffer)} | "
                    f"raw_conf={current_confidence:.1f}%"
                )

            # --- FAST TRANSITION BYPASS FOR HIGH-CONFIDENCE PREDICTIONS (>85%) ---
            if current_confidence > 85.0 and raw_label not in ("ANALYZING...", "LOW CONFIDENCE", "NO HAND DETECTED", "MODEL ERROR"):
                if self.session_stable_label.get(session_id) != raw_label:
                    logger.info(
                        f"[FAST-LOCK] session='{session_id}' | "
                        f"gesture='{raw_label}' conf={current_confidence:.1f}% | "
                        f"bypassing stabilization due to high confidence (>85%)"
                    )
                    # Trigger async Speech output
                    if self.should_speak(session_id, raw_label):
                        speech_service.speak(raw_label)

                self.session_stable_label[session_id] = raw_label
                self.session_last_result[session_id] = (raw_label, current_confidence)
                self.session_consecutive[session_id] = self.stable_frames_required
                self.session_last_candidate[session_id] = raw_label
                
                # Fill history queue with raw_label to maintain stability
                history.clear()
                for _ in range(self.prediction_window):
                    history.append(raw_label)
                
                # Set a short freeze to prevent high-speed duplicate triggers but keep feel immediate
                self.session_freeze_until[session_id] = current_time + self.freeze_duration
                
                return f"{raw_label} ({current_confidence:.0f}%)", current_confidence

            # 6. Confidence Smoothing (responsive: 60% current, 40% previous)
            prev_conf: float = self.session_prev_confidence.get(session_id, 0.0)
            prev_raw_label: Optional[str] = self.session_prev_raw_label.get(session_id)

            # Reset smoothing when predicted label changes (prevents stale drag)
            if prev_raw_label is not None and raw_label != prev_raw_label:
                smoothed_confidence: float = current_confidence
                # Also clear old history to allow fast gesture switching
                history.clear()
                self.session_consecutive[session_id] = 0
                self.session_last_candidate[session_id] = None
                logger.info(
                    f"[SWITCH] session='{session_id}' | "
                    f"gesture changed: '{prev_raw_label}' -> '{raw_label}' | "
                    f"confidence reset, history cleared"
                )
            elif prev_conf == 0.0:
                smoothed_confidence = current_confidence
            else:
                smoothed_confidence = 0.4 * prev_conf + 0.6 * current_confidence

            self.session_prev_confidence[session_id] = smoothed_confidence
            self.session_prev_raw_label[session_id] = raw_label

            # 7. Confidence Floor Filtering
            if smoothed_confidence < (self.min_confidence * 100.0):
                # Log rejection reason periodically
                if self.session_debug_counter[session_id] % self._debug_log_interval == 0:
                    logger.info(
                        f"[REJECTED] session='{session_id}' | "
                        f"label='{raw_label}' conf={smoothed_confidence:.1f}% < threshold={self.min_confidence*100:.0f}% | "
                        f"reason='LOW_CONFIDENCE'"
                    )
                # Maintain the last stable prediction if available
                last_stable = self.session_stable_label.get(session_id)
                if last_stable:
                    self.session_last_result[session_id] = (last_stable, smoothed_confidence)
                    return f"{last_stable} ({smoothed_confidence:.0f}%)", smoothed_confidence

                self.session_last_result[session_id] = ("LOW CONFIDENCE", smoothed_confidence)
                return "LOW CONFIDENCE", smoothed_confidence

            # 8. Rolling Prediction Smoothing & Majority Voting (3/5 window)
            history.append(raw_label)
            counter: Counter[str] = Counter(history)
            smoothed_label: str
            count: int
            smoothed_label, count = counter.most_common(1)[0]

            # Candidate requires majority (at least 3 out of 5 predictions)
            candidate: Optional[str] = smoothed_label if count >= 3 else None

            if candidate is None:
                if self.session_debug_counter[session_id] % self._debug_log_interval == 0:
                    logger.info(
                        f"[REJECTED] session='{session_id}' | "
                        f"label='{raw_label}' | history={dict(counter)} | "
                        f"reason='NO_MAJORITY ({count}/{len(history)})'"
                    )
                # Keep last stable prediction visible during transition
                last_stable = self.session_stable_label.get(session_id)
                if last_stable:
                    self.session_last_result[session_id] = (last_stable, smoothed_confidence)
                    return f"{last_stable} ({smoothed_confidence:.0f}%)", smoothed_confidence

                self.session_last_result[session_id] = ("ANALYZING...", smoothed_confidence)
                return "ANALYZING...", smoothed_confidence

            # 9. Stable Gesture Locking
            if candidate == self.session_last_candidate[session_id]:
                self.session_consecutive[session_id] += 1
            else:
                self.session_consecutive[session_id] = 1
                self.session_last_candidate[session_id] = candidate

            # Accept prediction after stable_frames_required consecutive matches
            if self.session_consecutive[session_id] >= self.stable_frames_required:
                if self.session_stable_label[session_id] != candidate:
                    logger.info(
                        f"[LOCKED] session='{session_id}' | "
                        f"gesture='{candidate}' conf={smoothed_confidence:.1f}% | "
                        f"consecutive={self.session_consecutive[session_id]}"
                    )
                    # Trigger async Speech output
                    if self.should_speak(session_id, candidate):
                        speech_service.speak(candidate)

                    # Freeze output for self.freeze_duration seconds (0.2 seconds)
                    self.session_freeze_until[session_id] = current_time + self.freeze_duration

                self.session_stable_label[session_id] = candidate
                self.session_last_result[session_id] = (candidate, smoothed_confidence)
                return f"{candidate} ({smoothed_confidence:.0f}%)", smoothed_confidence

            # Before stable matches reached, show last stable prediction if available
            if self.session_debug_counter[session_id] % self._debug_log_interval == 0:
                logger.info(
                    f"[PENDING] session='{session_id}' | "
                    f"candidate='{candidate}' conf={smoothed_confidence:.1f}% | "
                    f"consecutive={self.session_consecutive[session_id]}/{self.stable_frames_required} | "
                    f"reason='BUILDING_STABILITY'"
                )
            
            last_stable = self.session_stable_label.get(session_id)
            if last_stable:
                self.session_last_result[session_id] = (last_stable, smoothed_confidence)
                return f"{last_stable} ({smoothed_confidence:.0f}%)", smoothed_confidence

            self.session_last_result[session_id] = ("ANALYZING...", smoothed_confidence)
            return "ANALYZING...", smoothed_confidence

        except Exception as e:
            logger.error(f"Error during LSTM sequence prediction: {e}")
            return "ANALYZING...", 0.0

    def should_speak(self, session_id: str, label: str) -> bool:
        """Determines speech queue logic and prevents double voice spam."""
        if not label or label in ("ANALYZING...", "LOW CONFIDENCE", "NO HAND DETECTED"):
            return False

        current_time: float = time.time()
        last_spoken: Optional[str] = self.session_last_spoken.get(session_id)

        if label != last_spoken:
            self.session_last_spoken[session_id] = label
            self.session_last_spoken_time[session_id] = current_time
            return True

        return False

    def reset_session(self, session_id: str) -> None:
        """Resets the rolling sequence buffer and states for a session."""
        if session_id in self.session_buffers:
            self.session_buffers[session_id].clear()
            self.session_history[session_id].clear()
            self.session_stable_label[session_id] = None
            self.session_consecutive[session_id] = 0
            self.session_last_candidate[session_id] = None
            self.session_no_hand_count[session_id] = 0

            self.session_freeze_until[session_id] = 0.0
            self.session_frame_counter[session_id] = 0
            self.session_last_result[session_id] = ("ANALYZING...", 0.0)
            self.session_prev_confidence[session_id] = 0.0
            self.session_prev_features[session_id] = None
            self.session_prev_raw_label[session_id] = None

            self.session_last_spoken[session_id] = None
            self.session_last_spoken_time[session_id] = 0.0

            self.session_debug_counter[session_id] = 0
            logger.info(f"Session '{session_id}' state reset successfully.")


# Instantiate global prediction service instance
lstm_prediction_service: LSTMPredictionService = LSTMPredictionService()
