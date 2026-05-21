from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from utils.logger import logger


class MediaPipeService:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        """Initializes the MediaPipe Holistic solution for both hands and face mesh detection."""
        self.mp_holistic: Any = mp.solutions.holistic
        self.holistic: Any = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw: Any = mp.solutions.drawing_utils
        self.mp_drawing_styles: Any = mp.solutions.drawing_styles
        logger.info("MediaPipe Holistic service initialized successfully.")

    def extract_landmarks(self, img_bgr: Optional[NDArray[Any]]) -> Tuple[NDArray[Any], Any]:
        """
        Processes a BGR image and extracts left hand, right hand, and face landmarks.
        Returns:
            - combined_features: a flat numpy array of 1530 floats (63 left hand, 63 right hand, 1404 face).
            - results: raw holistic results object containing landmarks for drawing.
        """
        if img_bgr is None:
            # Return zeros if image is invalid
            return np.zeros(1530), None

        h_img: int
        w_img: int
        h_img, w_img, _ = img_bgr.shape
        img_rgb: NDArray[Any] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Disable writing on image to improve speed
        img_rgb.flags.writeable = False
        results: Any = self.holistic.process(img_rgb)
        img_rgb.flags.writeable = True

        # 1. Extract and Normalize Left Hand (63 features)
        left_hand_features: NDArray[Any] = np.zeros(63)
        if results.left_hand_landmarks:
            lh: Any = results.left_hand_landmarks.landmark
            wrist: Any = lh[0]
            ref_x: float = wrist.x
            ref_y: float = wrist.y
            ref_z: float = wrist.z

            raw_coords: list[float] = []
            for lm in lh:
                raw_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])

            # Scale-invariance
            max_val: float = max(abs(c) for c in raw_coords)
            if max_val > 0:
                left_hand_features = np.array([c / max_val for c in raw_coords])
            else:
                left_hand_features = np.array(raw_coords)

        # 2. Extract and Normalize Right Hand (63 features)
        right_hand_features: NDArray[Any] = np.zeros(63)
        if results.right_hand_landmarks:
            rh: Any = results.right_hand_landmarks.landmark
            wrist = rh[0]
            ref_x = wrist.x
            ref_y = wrist.y
            ref_z = wrist.z

            raw_coords = []
            for lm in rh:
                raw_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])

            # Scale-invariance
            max_val = max(abs(c) for c in raw_coords)
            if max_val > 0:
                right_hand_features = np.array([c / max_val for c in raw_coords])
            else:
                right_hand_features = np.array(raw_coords)

        # 3. Extract and Normalize Face (1404 features - 468 landmarks)
        face_features: NDArray[Any] = np.zeros(1404)
        if results.face_landmarks:
            fl: Any = results.face_landmarks.landmark
            # Use landmark 1 (nose tip) as face reference center
            ref_lm: Any = fl[1]
            ref_x = ref_lm.x
            ref_y = ref_lm.y
            ref_z = ref_lm.z

            raw_coords = []
            for lm in fl:
                raw_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])

            # Scale-invariance
            max_val = max(abs(c) for c in raw_coords)
            if max_val > 0:
                face_features = np.array([c / max_val for c in raw_coords])
            else:
                face_features = np.array(raw_coords)

        # Combine into flat 1530 feature vector
        combined_features: NDArray[Any] = np.concatenate([left_hand_features, right_hand_features, face_features])
        return combined_features, results

    def draw_landmarks(self, img_bgr: Optional[NDArray[Any]], results: Any) -> Optional[NDArray[Any]]:
        """Draws stylized holistic landmarks (hands, face mesh contours) on the image."""
        if img_bgr is None or results is None:
            return img_bgr

        # Draw face mesh contours
        if results.face_landmarks:
            self.mp_draw.draw_landmarks(
                img_bgr,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Draw left hand skeletal connections
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                img_bgr,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )

        # Draw right hand skeletal connections
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                img_bgr,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )

        return img_bgr


# Instantiate a global service instance
mediapipe_service: MediaPipeService = MediaPipeService()
