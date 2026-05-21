import base64
from typing import Optional

import numpy as np
import cv2


def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """Decodes a base64 image string into a numpy array suitable for cv2.
    Returns None if decoding fails."""
    try:
        if isinstance(b64_string, str) and b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[-1]
        raw: bytes = base64.b64decode(b64_string)
        nparr: np.ndarray = np.frombuffer(raw, np.uint8)
        img: Optional[np.ndarray] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None
