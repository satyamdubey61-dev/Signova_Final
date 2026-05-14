import base64
import numpy as np
import cv2

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decodes a base64 image string into a numpy array suitable for cv2."""
    if isinstance(b64_string, str) and b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[-1]
    raw = base64.b64decode(b64_string)
    nparr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
