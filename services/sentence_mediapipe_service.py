"""
Sentence Mode MediaPipe Service — COMPLETELY SEPARATE from the existing mediapipe_service.py.

OPTIMIZED v2:
  - Upper-body-only pose (landmarks 0-24 = 25 landmarks × 3 = 75 features, not full 33)
    BUT we use landmarks 11-22 (shoulders, elbows, wrists, hands) = 12 × 3 = 36,
    plus nose/eyes/ears (0-10) = 11 × 3 = 33, total upper = 23 landmarks × 3 = 69.
    Simplified: landmarks 0-22 inclusive = 23 × 3 = 69 features.
  - Wrist velocity tracking for gesture-end detection
  - get_motion_state() returns (has_hands, velocity_magnitude)
  - FEATURE_SIZE: 63 (left) + 63 (right) + 69 (upper pose) = 195

Feature vector: 63 left + 63 right + 69 upper-pose = 195 per frame
"""
import threading
import time
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from utils.logger import logger


# ── Feature size constants ──────────────────────────────────────────────────
SENTENCE_FEATURE_SIZE = 195   # 63 left + 63 right + 69 upper-pose (23 landmarks × 3)
UPPER_POSE_LANDMARK_COUNT = 23  # landmarks 0..22 inclusive (nose → left/right wrist area)


class SentenceMediaPipeService:
    """
    Independent MediaPipe service for sentence-level sign language detection.

    Changes from v1:
      - Uses upper-body-only pose (landmarks 0-22) — drops heavy leg landmarks.
        This reduces pose features from 99 (33×3) to 69 (23×3).
      - Tracks wrist positions across frames to compute motion velocity.
      - get_motion_state() returns (has_hands: bool, velocity: float) for
        early gesture-end detection in the prediction service.
      - SENTENCE_FEATURE_SIZE changed: 225 → 195
      - Uses model_complexity=0 for faster inference on CPU.
    """

    def __init__(
        self,
        model_complexity: int = 0,          # 0 = fastest, good enough for sentences
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.45,
    ) -> None:
        self.mp_holistic: Any = mp.solutions.holistic
        self.holistic: Any = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )
        self.mp_draw: Any = mp.solutions.drawing_utils
        self.mp_drawing_styles: Any = mp.solutions.drawing_styles

        # Threading lock — no shared state with main mediapipe_service.py
        self._lock: threading.Lock = threading.Lock()
        self._last_timestamp_ms: int = 0

        # Wrist velocity tracking (right wrist = landmark 16, left = landmark 15)
        self._prev_left_wrist:  Optional[NDArray[Any]] = None
        self._prev_right_wrist: Optional[NDArray[Any]] = None
        self._velocity_alpha: float = 0.35   # EMA smoothing for velocity

        # Smoothed velocity (EMA) — updated on every extract_features call
        self._smoothed_velocity: float = 0.0

        logger.info(
            "[SentenceMP v2] Initialized — upper-body pose only, "
            f"FEATURE_SIZE={SENTENCE_FEATURE_SIZE}, model_complexity={model_complexity}"
        )

    # ── Feature Extraction ────────────────────────────────────────────────────

    def extract_features(
        self, img_bgr: Optional[NDArray[Any]]
    ) -> Tuple[NDArray[Any], Any]:
        """
        Process a BGR frame and return (feature_vector, mediapipe_results).

        Returns:
            features : NDArray shape (195,) — zeros if extraction fails.
            results  : Raw MediaPipe holistic results for optional drawing.

        Feature layout:
            [0:63]   left hand  (21 landmarks × 3, wrist-relative + normalised)
            [63:126] right hand (21 landmarks × 3, wrist-relative + normalised)
            [126:195] upper-body pose (landmarks 0-22, nose-relative + normalised)
        """
        if img_bgr is None:
            return np.zeros(SENTENCE_FEATURE_SIZE, dtype=np.float32), None

        img_rgb: NDArray[Any] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Non-blocking — skip frame if service is busy
        if not self._lock.acquire(blocking=False):
            logger.debug("[SentenceMP] Busy — skipping overlapping frame")
            return np.zeros(SENTENCE_FEATURE_SIZE, dtype=np.float32), None

        try:
            current_ts: int = int(time.monotonic() * 1_000_000)
            if current_ts <= self._last_timestamp_ms:
                current_ts = self._last_timestamp_ms + 1
            self._last_timestamp_ms = current_ts

            img_rgb.flags.writeable = False
            results: Any = self.holistic.process(img_rgb)
            img_rgb.flags.writeable = True
        except Exception as exc:
            logger.warning(f"[SentenceMP] Processing error: {exc}")
            return np.zeros(SENTENCE_FEATURE_SIZE, dtype=np.float32), None
        finally:
            self._lock.release()

        # ── Left Hand (63 features = 21 landmarks × 3) ──────────────────────
        left_hand: NDArray[Any] = np.zeros(63, dtype=np.float32)
        if results.left_hand_landmarks:
            lh = results.left_hand_landmarks.landmark
            wrist = lh[0]
            raw: list = []
            for lm in lh:
                raw.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            max_val = max(abs(c) for c in raw) or 1.0
            left_hand = np.array([c / max_val for c in raw], dtype=np.float32)

        # ── Right Hand (63 features) ─────────────────────────────────────────
        right_hand: NDArray[Any] = np.zeros(63, dtype=np.float32)
        if results.right_hand_landmarks:
            rh = results.right_hand_landmarks.landmark
            wrist = rh[0]
            raw = []
            for lm in rh:
                raw.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            max_val = max(abs(c) for c in raw) or 1.0
            right_hand = np.array([c / max_val for c in raw], dtype=np.float32)

        # ── Upper-Body Pose (landmarks 0-22 = 23 × 3 = 69 features) ─────────
        # Landmarks 0-22 cover: nose, eyes, ears, shoulders, elbows, wrists,
        # pinky, index, thumb base points — everything relevant for sentences.
        # Landmarks 23-32 (hips, knees, ankles, feet) are discarded.
        pose_feats: NDArray[Any] = np.zeros(69, dtype=np.float32)
        new_left_wrist:  Optional[NDArray[Any]] = None
        new_right_wrist: Optional[NDArray[Any]] = None

        if results.pose_landmarks:
            pl = results.pose_landmarks.landmark
            ref = pl[0]  # nose as reference centre
            raw = []
            for i in range(UPPER_POSE_LANDMARK_COUNT):  # 0..22
                lm = pl[i]
                raw.extend([lm.x - ref.x, lm.y - ref.y, lm.z - ref.z])
            max_val = max(abs(c) for c in raw) or 1.0
            pose_feats = np.array([c / max_val for c in raw], dtype=np.float32)

            # Capture wrist positions for velocity (landmark 15=left, 16=right)
            lw = pl[15]
            rw = pl[16]
            new_left_wrist  = np.array([lw.x, lw.y, lw.z], dtype=np.float32)
            new_right_wrist = np.array([rw.x, rw.y, rw.z], dtype=np.float32)

        # ── Wrist Velocity Computation ───────────────────────────────────────
        velocity: float = 0.0
        if new_left_wrist is not None and new_right_wrist is not None:
            v_left:  float = 0.0
            v_right: float = 0.0
            if self._prev_left_wrist is not None:
                v_left = float(np.linalg.norm(new_left_wrist - self._prev_left_wrist))
            if self._prev_right_wrist is not None:
                v_right = float(np.linalg.norm(new_right_wrist - self._prev_right_wrist))
            velocity = max(v_left, v_right)
            self._prev_left_wrist  = new_left_wrist
            self._prev_right_wrist = new_right_wrist
        else:
            # No pose detected — keep previous wrist positions, reset velocity
            velocity = 0.0

        # EMA smooth velocity
        self._smoothed_velocity = (
            self._velocity_alpha * velocity
            + (1.0 - self._velocity_alpha) * self._smoothed_velocity
        )

        combined: NDArray[Any] = np.concatenate([left_hand, right_hand, pose_feats])
        return combined, results

    # ── Motion State ──────────────────────────────────────────────────────────

    def get_motion_state(self, results: Any) -> Tuple[bool, float]:
        """
        Returns (has_hands, smoothed_velocity).

        has_hands : True if at least one hand is detected in the frame.
        smoothed_velocity : EMA-smoothed wrist displacement magnitude (0..∞).
                            Values below ~0.004 typically indicate a still pose.
        """
        if results is None:
            return False, 0.0
        has_hands: bool = bool(
            results.left_hand_landmarks or results.right_hand_landmarks
        )
        return has_hands, self._smoothed_velocity

    def has_motion(self, results: Any) -> bool:
        """Backward-compatible wrapper — returns has_hands only."""
        has_hands, _ = self.get_motion_state(results)
        return has_hands

    def reset_velocity(self) -> None:
        """Reset wrist tracking — call when session resets."""
        self._prev_left_wrist  = None
        self._prev_right_wrist = None
        self._smoothed_velocity = 0.0

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_landmarks(
        self, img_bgr: Optional[NDArray[Any]], results: Any
    ) -> Optional[NDArray[Any]]:
        """Draw hand and upper-body pose landmarks (no face mesh)."""
        if img_bgr is None or results is None:
            return img_bgr

        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                img_bgr,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                img_bgr,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                img_bgr,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return img_bgr


# Module-level singleton
sentence_mediapipe_service: SentenceMediaPipeService = SentenceMediaPipeService()
