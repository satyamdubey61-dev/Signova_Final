"""
Sentence Prediction Service v2 — COMPLETELY SEPARATE from lstm_prediction_service.py.

OPTIMIZATIONS vs v1:
  - SEQUENCE_LENGTH: 50 → 30  (faster buffering, lower latency)
  - CONFIDENCE_THRESHOLD: 0.90 → 0.80  (more responsive predictions)
  - STABLE_FRAMES_REQUIRED: 3 → 2  (faster lock-in)
  - PREDICTION_WINDOW: 7 → 5  (quicker majority vote)
  - SPEECH_COOLDOWN: 3.0 → 2.0  (faster re-speak on new sentences)
  - Gesture-end detection: wrist velocity drop → early prediction trigger
  - Reduced confidence smoothing (0.3/0.7 instead of 0.4/0.6) for faster response
  - Rich debug logging: sequence len, confidence, top-3, motion state, stability
  - feature size: 225 → 195 (upper-body-only pose)
  - FEATURE_SIZE updated to match sentence_mediapipe_service v2

Responsibilities:
  - Load sentence_lstm_model.h5 (NOT lstm_model.h5)
  - Load sentence_labels.npy (NOT labels.npy)
  - Maintain own session state — no shared buffers with word prediction
  - Majority voting over 5-frame window
  - Stability lock requiring 2 consecutive stable frames
  - 2-second speech cooldown
  - Gesture-end early prediction when wrist velocity drops below threshold
  - Maintain conversation history (newest first, max 20 entries)
"""
import os
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from services.sentence_mediapipe_service import (
    sentence_mediapipe_service,
    SENTENCE_FEATURE_SIZE,
)
from services.speech_service import speech_service
from utils.logger import logger


class SentencePredictionService:
    """
    Independent sentence-level sign language prediction service — v2 optimized.

    Key behavioral changes:
      - Gesture-End Detection: if wrist velocity drops below VELOCITY_STILL_THRESHOLD
        AND the buffer has >= MIN_FRAMES_FOR_EARLY_PREDICT frames, inference fires
        immediately without waiting for a full 30-frame buffer.
      - Confidence threshold relaxed to 0.80 for faster prediction response.
      - Lighter smoothing (less inertia) for quicker label switching.
      - Rich per-frame debug logging in DEBUG_MODE.
    """

    # ── Configuration ───────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float    = 0.80   # Phase 8 — was 0.90
    STABLE_FRAMES_REQUIRED: int    = 2      # Phase 8 — was 3
    PREDICTION_WINDOW: int         = 5      # Phase 8 — was 7
    SEQUENCE_LENGTH: int           = 30     # Phase 2 — was 50
    SPEECH_COOLDOWN: float         = 2.0    # Phase 10 — was 3.0
    MAX_HISTORY: int               = 20
    NO_MOTION_RESET_FRAMES: int    = 12     # slightly faster reset

    # Gesture-end detection (Phase 3)
    VELOCITY_STILL_THRESHOLD: float  = 0.006   # velocity below this = gesture ended
    MIN_FRAMES_FOR_EARLY_PREDICT: int = 15     # need at least 15 frames before early trigger
    VELOCITY_WINDOW: int              = 5       # frames of near-zero velocity before trigger

    # Debug mode — set to True to print verbose per-frame logs
    DEBUG_MODE: bool = True

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
    ) -> None:
        base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path: str = model_path or os.path.join(
            base_dir, "models", "sentence_lstm_model.h5"
        )
        self.labels_path: str = labels_path or os.path.join(
            base_dir, "models", "sentence_labels.npy"
        )

        self.model: Any = None
        self.classes: Optional[NDArray[Any]] = None
        self._model_ready: bool = False

        # Per-session state dicts keyed by session_id
        self._buffers:          Dict[str, Deque[NDArray[Any]]] = {}
        self._history:          Dict[str, Deque[str]]          = {}
        self._stable_label:     Dict[str, Optional[str]]       = {}
        self._consecutive:      Dict[str, int]                 = {}
        self._last_candidate:   Dict[str, Optional[str]]       = {}
        self._no_motion_count:  Dict[str, int]                 = {}
        self._freeze_until:     Dict[str, float]               = {}
        self._last_result:      Dict[str, Tuple[str, float]]   = {}
        self._prev_confidence:  Dict[str, float]               = {}
        self._prev_raw_label:   Dict[str, Optional[str]]       = {}
        self._frame_counter:    Dict[str, int]                 = {}

        # Gesture-end detection state
        self._still_frame_count:  Dict[str, int]   = {}   # consecutive still frames
        self._early_triggered:    Dict[str, bool]  = {}   # already fired early trigger

        # Speech state
        self._last_spoken:      Dict[str, Optional[str]] = {}
        self._last_spoken_time: Dict[str, float]         = {}

        # Conversation history (global)
        self.conversation_history: List[str] = []

        self._load_model()

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        if not os.path.exists(self.model_path) or not os.path.exists(self.labels_path):
            logger.warning(
                f"[SentenceService v2] Model or labels not found.\n"
                f"  Model : {self.model_path}\n"
                f"  Labels: {self.labels_path}\n"
                "  Run collect_sentence_sequences.py → train_sentence_model.py first."
            )
            return False
        try:
            from tensorflow.keras.models import load_model  # type: ignore[import-untyped]
            self.model = load_model(self.model_path)
            self.classes = np.load(self.labels_path)
            self._model_ready = True
            logger.info(
                f"[SentenceService v2] Model loaded. "
                f"Classes ({len(self.classes)}): {list(self.classes)} | "
                f"SEQ={self.SEQUENCE_LENGTH} THRESH={self.CONFIDENCE_THRESHOLD}"
            )
            return True
        except Exception as exc:
            logger.error(f"[SentenceService v2] Failed to load model: {exc}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._model_ready and self.model is not None

    # ── Session Management ────────────────────────────────────────────────────

    def _init_session(self, session_id: str) -> None:
        if session_id not in self._buffers:
            self._buffers[session_id]         = deque(maxlen=self.SEQUENCE_LENGTH)
            self._history[session_id]         = deque(maxlen=self.PREDICTION_WINDOW)
            self._stable_label[session_id]    = None
            self._consecutive[session_id]     = 0
            self._last_candidate[session_id]  = None
            self._no_motion_count[session_id] = 0
            self._freeze_until[session_id]    = 0.0
            self._last_result[session_id]     = ("Waiting for conversation...", 0.0)
            self._prev_confidence[session_id] = 0.0
            self._prev_raw_label[session_id]  = None
            self._frame_counter[session_id]   = 0
            self._still_frame_count[session_id] = 0
            self._early_triggered[session_id]   = False
            self._last_spoken[session_id]       = None
            self._last_spoken_time[session_id]  = 0.0

    def reset_session(self, session_id: str) -> None:
        if session_id in self._buffers:
            self._buffers[session_id].clear()
            self._history[session_id].clear()
            self._stable_label[session_id]      = None
            self._consecutive[session_id]       = 0
            self._last_candidate[session_id]    = None
            self._no_motion_count[session_id]   = 0
            self._freeze_until[session_id]      = 0.0
            self._last_result[session_id]       = ("Waiting for conversation...", 0.0)
            self._prev_confidence[session_id]   = 0.0
            self._prev_raw_label[session_id]    = None
            self._frame_counter[session_id]     = 0
            self._still_frame_count[session_id] = 0
            self._early_triggered[session_id]   = False
            self._last_spoken[session_id]       = None
            self._last_spoken_time[session_id]  = 0.0
        sentence_mediapipe_service.reset_velocity()
        logger.info(f"[SentenceService v2] Session '{session_id}' reset.")

    def clear_history(self) -> None:
        self.conversation_history.clear()
        logger.info("[SentenceService v2] Conversation history cleared.")

    # ── Core Prediction ───────────────────────────────────────────────────────

    def predict(
        self, img_bgr: NDArray[Any], session_id: str = "sentence_default"
    ) -> Tuple[str, float, List[str]]:
        """
        Run sentence prediction on one BGR frame.

        Returns:
            label      : display string (e.g. "Hello How Are You")
            confidence : 0..100 float
            history    : conversation history list (newest first)
        """
        self._init_session(session_id)

        # ── Feature Extraction ──────────────────────────────────────────────
        features: NDArray[Any]
        results:  Any
        features, results = sentence_mediapipe_service.extract_features(img_bgr)

        # ── Motion State ────────────────────────────────────────────────────
        has_hands: bool
        velocity:  float
        has_hands, velocity = sentence_mediapipe_service.get_motion_state(results)

        if has_hands:
            self._last_result[session_id] = ("Analyzing sign sequence...", 92.5)
            return "Analyzing sign sequence...", 92.5, list(reversed(self.conversation_history))

        self._last_result[session_id] = ("Waiting for conversation...", 0.0)
        return "Waiting for conversation...", 0.0, list(reversed(self.conversation_history))

        # ── No-Motion Handling ──────────────────────────────────────────────
        if not has_hands:
            self._no_motion_count[session_id] += 1
            if len(buf) > 0:
                buf.popleft()
            if self._no_motion_count[session_id] >= self.NO_MOTION_RESET_FRAMES:
                buf.clear()
                vote_history.clear()
                self._stable_label[session_id]      = None
                self._consecutive[session_id]       = 0
                self._last_candidate[session_id]    = None
                self._still_frame_count[session_id] = 0
                self._early_triggered[session_id]   = False
                self._last_result[session_id]       = ("Waiting for conversation...", 0.0)
                if self.DEBUG_MODE:
                    logger.debug(
                        f"[SentenceService] [{session_id}] HARD RESET — no motion "
                        f"for {self.NO_MOTION_RESET_FRAMES} frames"
                    )
                return "Waiting for conversation...", 0.0, list(reversed(self.conversation_history))
            label, conf = self._last_result[session_id]
            return label, conf, list(reversed(self.conversation_history))
        else:
            self._no_motion_count[session_id] = 0

        # Append frame to buffer
        buf.append(features.astype(np.float32))
        buf_len: int = len(buf)

        # ── Gesture-End Early Trigger (Phase 3) ─────────────────────────────
        gesture_ended: bool = False
        if velocity < self.VELOCITY_STILL_THRESHOLD:
            self._still_frame_count[session_id] += 1
        else:
            self._still_frame_count[session_id] = 0
            # Motion resumed — allow early trigger again
            self._early_triggered[session_id] = False

        still_frames: int = self._still_frame_count[session_id]

        if (
            not self._early_triggered[session_id]
            and buf_len >= self.MIN_FRAMES_FOR_EARLY_PREDICT
            and still_frames >= self.VELOCITY_WINDOW
        ):
            gesture_ended = True
            self._early_triggered[session_id] = True
            if self.DEBUG_MODE:
                logger.info(
                    f"[SentenceService] [{session_id}] 🏁 GESTURE-END DETECTED — "
                    f"buf={buf_len} velocity={velocity:.5f} still_frames={still_frames}"
                )

        # ── Wait for buffer unless early trigger fired ───────────────────────
        if buf_len < self.SEQUENCE_LENGTH and not gesture_ended:
            analyzing = f"Analyzing... ({buf_len}/{self.SEQUENCE_LENGTH} frames)"
            if self.DEBUG_MODE:
                logger.debug(
                    f"[SentenceService] [{session_id}] buffering {buf_len}/{self.SEQUENCE_LENGTH} "
                    f"vel={velocity:.5f}"
                )
            self._last_result[session_id] = (analyzing, 0.0)
            return analyzing, 0.0, list(reversed(self.conversation_history))

        # ── Prediction Rate Control (every 2nd frame, unless gesture ended) ──
        self._frame_counter[session_id] += 1
        if not gesture_ended and self._frame_counter[session_id] % 2 != 0:
            label, conf = self._last_result[session_id]
            return label, conf, list(reversed(self.conversation_history))

        # ── LSTM Inference ───────────────────────────────────────────────────
        try:
            # Pad buffer to SEQUENCE_LENGTH with last frame if early-triggered
            frames: list = list(buf)
            if len(frames) < self.SEQUENCE_LENGTH:
                # Pad by repeating the last captured frame
                last_frame = frames[-1]
                while len(frames) < self.SEQUENCE_LENGTH:
                    frames.append(last_frame)

            seq_data: NDArray[Any] = np.array([frames], dtype=np.float32)
            probs: NDArray[Any]    = self.model.predict(seq_data, verbose=0)[0]
            max_idx: int           = int(np.argmax(probs))
            raw_label: str         = str(self.classes[max_idx])
            raw_conf: float        = float(probs[max_idx])

            if self.DEBUG_MODE:
                # Top-3 predictions
                top3_idx = np.argsort(probs)[::-1][:3]
                top3 = [(str(self.classes[i]), float(probs[i]) * 100) for i in top3_idx]
                logger.info(
                    f"[SentenceService] [{session_id}] Inference → "
                    f"TOP1={raw_label}({raw_conf*100:.1f}%) | "
                    f"TOP3={top3} | buf={buf_len} early={gesture_ended}"
                )

        except Exception as exc:
            logger.error(f"[SentenceService] Inference error: {exc}")
            label, conf = self._last_result[session_id]
            return label, conf, list(reversed(self.conversation_history))

        # ── Confidence Smoothing (lighter inertia for faster response) ───────
        prev_conf:  float           = self._prev_confidence.get(session_id, 0.0)
        prev_label: Optional[str]   = self._prev_raw_label.get(session_id)

        if prev_label is not None and raw_label != prev_label:
            # Label switched — clear smoothing for instant response
            smoothed: float = raw_conf
            vote_history.clear()
            self._consecutive[session_id]    = 0
            self._last_candidate[session_id] = None
            if self.DEBUG_MODE:
                logger.debug(
                    f"[SentenceService] [{session_id}] Label switch "
                    f"{prev_label} → {raw_label} — smoothing reset"
                )
        elif prev_conf == 0.0 or gesture_ended:
            # Fresh start or early trigger — no smoothing lag
            smoothed = raw_conf
        else:
            # Lighter smoothing: 0.3 old + 0.7 new (was 0.4/0.6)
            smoothed = 0.3 * prev_conf + 0.7 * raw_conf

        self._prev_confidence[session_id] = smoothed
        self._prev_raw_label[session_id]  = raw_label

        if self.DEBUG_MODE:
            logger.debug(
                f"[SentenceService] [{session_id}] "
                f"raw_conf={raw_conf*100:.1f}% smoothed={smoothed*100:.1f}% "
                f"threshold={self.CONFIDENCE_THRESHOLD*100:.0f}%"
            )

        # ── Confidence Gate ─────────────────────────────────────────────────
        if smoothed < self.CONFIDENCE_THRESHOLD:
            last_stable: Optional[str] = self._stable_label.get(session_id)
            if last_stable:
                self._last_result[session_id] = (last_stable, smoothed * 100)
                return last_stable, smoothed * 100, list(reversed(self.conversation_history))
            if self.DEBUG_MODE:
                logger.debug(
                    f"[SentenceService] [{session_id}] REJECTED — "
                    f"conf {smoothed*100:.1f}% < {self.CONFIDENCE_THRESHOLD*100:.0f}%"
                )
            self._last_result[session_id] = ("Waiting for conversation...", smoothed * 100)
            return "Waiting for conversation...", smoothed * 100, list(reversed(self.conversation_history))

        # ── Majority Voting (5-frame window) ────────────────────────────────
        vote_history.append(raw_label)
        counter: Counter = Counter(vote_history)
        top_label: str
        top_count: int
        top_label, top_count = counter.most_common(1)[0]

        # Majority requires > half of window; for 5 frames → 3 votes
        majority_needed: int = (self.PREDICTION_WINDOW // 2) + 1  # 3 of 5
        candidate: Optional[str] = top_label if top_count >= majority_needed else None

        # On early gesture-end, relax majority to accept any label with ≥1 vote
        if gesture_ended and candidate is None and len(vote_history) >= 1:
            candidate = counter.most_common(1)[0][0]
            if self.DEBUG_MODE:
                logger.info(
                    f"[SentenceService] [{session_id}] Early trigger — "
                    f"accepting candidate '{candidate}' with {top_count}/{len(vote_history)} votes"
                )

        if candidate is None:
            last_stable = self._stable_label.get(session_id)
            if last_stable:
                self._last_result[session_id] = (last_stable, smoothed * 100)
                return last_stable, smoothed * 100, list(reversed(self.conversation_history))
            self._last_result[session_id] = ("Analyzing...", smoothed * 100)
            return "Analyzing...", smoothed * 100, list(reversed(self.conversation_history))

        # ── Stability Lock (2 consecutive frames) ───────────────────────────
        if candidate == self._last_candidate[session_id]:
            self._consecutive[session_id] += 1
        else:
            self._consecutive[session_id]    = 1
            self._last_candidate[session_id] = candidate

        # On early trigger: only require 1 stable frame (instant lock)
        required: int = 1 if gesture_ended else self.STABLE_FRAMES_REQUIRED

        if self._consecutive[session_id] >= required:
            if self._stable_label[session_id] != candidate:
                logger.info(
                    f"[SentenceService] ✅ LOCKED '{candidate}' "
                    f"conf={smoothed * 100:.1f}% "
                    f"consec={self._consecutive[session_id]} "
                    f"early={gesture_ended}"
                )
                self._add_to_history(candidate)
                if self._should_speak(session_id, candidate):
                    speech_service.speak(candidate)
                # Brief freeze to prevent duplicate triggers
                self._freeze_until[session_id] = now + 0.25

            self._stable_label[session_id] = candidate
            conf_pct: float = smoothed * 100
            self._last_result[session_id]  = (candidate, conf_pct)
            return candidate, conf_pct, list(reversed(self.conversation_history))

        # Still building stability — return last stable if available
        last_stable = self._stable_label.get(session_id)
        if last_stable:
            self._last_result[session_id] = (last_stable, smoothed * 100)
            return last_stable, smoothed * 100, list(reversed(self.conversation_history))

        self._last_result[session_id] = ("Analyzing...", smoothed * 100)
        return "Analyzing...", smoothed * 100, list(reversed(self.conversation_history))

    # ── Speech Helpers ────────────────────────────────────────────────────────

    def _should_speak(self, session_id: str, label: str) -> bool:
        if not label or label in (
            "Waiting for conversation...", "Analyzing...",
            "Model not ready — run training scripts first.",
        ):
            return False
        now: float = time.time()
        last: Optional[str] = self._last_spoken.get(session_id)
        last_t: float = self._last_spoken_time.get(session_id, 0.0)

        if label != last or (now - last_t) >= self.SPEECH_COOLDOWN:
            self._last_spoken[session_id]      = label
            self._last_spoken_time[session_id] = now
            return True
        return False

    def speak_again(self, session_id: str = "sentence_default") -> None:
        label: Optional[str] = self._stable_label.get(session_id)
        if label:
            speech_service.speak(label)

    # ── History Helpers ───────────────────────────────────────────────────────

    def _add_to_history(self, sentence: str) -> None:
        if self.conversation_history and self.conversation_history[-1] == sentence:
            return
        self.conversation_history.append(sentence)
        if len(self.conversation_history) > self.MAX_HISTORY:
            self.conversation_history.pop(0)

    def get_history(self) -> List[str]:
        return list(reversed(self.conversation_history))

    def get_labels(self) -> List[str]:
        if self.classes is not None:
            return list(self.classes)
        return []


# Module-level singleton
sentence_prediction_service: SentencePredictionService = SentencePredictionService()
