from typing import Any, Dict, List, Optional, Tuple, Union

from flask import Blueprint, Response, request, jsonify, session
from services.lstm_prediction_service import lstm_prediction_service
from utils.image_utils import decode_base64_image
from utils.logger import logger

import numpy as np

predict_bp: Blueprint = Blueprint('predict', __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict() -> Tuple[Response, int] | Response:
    img_data: Optional[str] = None

    if request.is_json:
        data: Dict[str, Any] = request.get_json() or {}
        img_data = data.get("image")
    elif request.files:
        file = request.files.get("image")
        if file is not None:
            import base64
            img_data = base64.b64encode(file.read()).decode("utf-8")

    if not img_data:
        logger.warning("Prediction requested without image data.")
        return jsonify({"label": None, "error": "No image provided. Send JSON { \"image\": \"<base64>\" } or form-data 'image'."}), 400

    img: Optional[np.ndarray] = decode_base64_image(img_data)
    if img is None:
        logger.warning("Failed to decode image data.")
        return jsonify({"label": None, "error": "Invalid image data."}), 400

    # Retrieve unique session_id to isolate user sequence buffers
    session_id: str = str(session.get("user_email", "guest"))

    try:
        label: Optional[str]
        confidence: float
        label, confidence = lstm_prediction_service.predict(img, session_id=session_id)
    except Exception as e:
        logger.error(f"MediaPipe+LSTM pipeline prediction failed: {e}")
        label, confidence = None, 0.0

    return jsonify({"label": label, "confidence": confidence})


@predict_bp.route("/health")
def health() -> Response:
    # Retrieve labels dynamically from the LSTM model — only return trained labels
    labels: Optional[List[str]] = None
    if lstm_prediction_service.classes is not None:
        labels = list(lstm_prediction_service.classes)

    if not labels:
        labels = []

    return jsonify({"status": "ok", "labels": labels})
