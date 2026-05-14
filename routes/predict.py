from flask import Blueprint, request, jsonify
from services.prediction_service import PredictionService
from core.engine import labels
from utils.image_utils import decode_base64_image
from utils.logger import logger

predict_bp = Blueprint('predict', __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    img_data = None

    if request.is_json:
        data = request.get_json()
        img_data = data.get("image")
    elif request.files:
        file = request.files.get("image")
        if file:
            import base64
            img_data = base64.b64encode(file.read()).decode("utf-8")

    if not img_data:
        logger.warning("Prediction requested without image data.")
        return jsonify({"label": None, "error": "No image provided. Send JSON { \"image\": \"<base64>\" } or form-data 'image'."}), 400

    img = decode_base64_image(img_data)
    if img is None:
        logger.warning("Failed to decode image data.")
        return jsonify({"label": None, "error": "Invalid image data."}), 400

    label, confidence = PredictionService.predict(img)
    return jsonify({"label": label, "confidence": confidence})

@predict_bp.route("/health")
def health():
    return jsonify({"status": "ok", "labels": labels})
