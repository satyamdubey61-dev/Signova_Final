"""
Routes for the Live Sentence Conversation Mode — v2 OPTIMIZED.

COMPLETELY SEPARATE from existing routes (predict.py, features.py, auth.py, external.py).

Routes:
  GET  /sentence-mode          → Renders the sentence mode HTML template
  POST /api/predict-sentence   → Runs sentence prediction on a single frame
  POST /api/clear-sentence     → Resets session state + conversation history
  GET  /api/sentence-status    → Returns model ready state + class labels
  POST /api/speak-again        → Re-trigger TTS for last stable sentence

v2 changes:
  - /api/predict-sentence now returns additional debug fields:
      buf_len, velocity, still_frames, top3, early_trigger
  - motion_stopped hint from frontend accepted (logged, backend uses own velocity)
  - /api/sentence-status now returns sequence_length and confidence_threshold
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request, session

from services.sentence_prediction_service import sentence_prediction_service
from services.sentence_mediapipe_service  import sentence_mediapipe_service
from utils.image_utils import decode_base64_image
from utils.logger import logger

sentence_mode_bp: Blueprint = Blueprint("sentence_mode", __name__)


# ── Page Route ────────────────────────────────────────────────────────────────

@sentence_mode_bp.route("/sentence-mode")
def sentence_mode_page() -> Response:
    """Serve the sentence mode HTML page."""
    return render_template("sentence_mode.html")


# ── API: Predict ──────────────────────────────────────────────────────────────

@sentence_mode_bp.route("/api/predict-sentence", methods=["POST"])
def predict_sentence() -> Tuple[Response, int] | Response:
    """
    Accept a base64-encoded video frame, run sentence prediction, return JSON.

    Request body:
        {
          "image"         : "<base64-encoded-jpeg>",
          "motion_stopped": true/false   (optional frontend hint)
        }

    Response:
        {
          "sentence"      : "Hello How Are You",
          "confidence"    : 96.3,
          "history"       : ["Hello How Are You", ...],
          "buf_len"       : 28,
          "velocity"      : 0.00312,
          "still_frames"  : 7,
          "top3"          : ["HelloHowAreYou:94.1", "IAmFine:3.2", "ThankYou:2.7"],
          "early_trigger" : false
        }
    """
    img_data: Optional[str] = None

    if request.is_json:
        data: Dict[str, Any] = request.get_json() or {}
        img_data        = data.get("image")
        motion_stopped  = bool(data.get("motion_stopped", False))
    elif request.files:
        file = request.files.get("image")
        if file is not None:
            import base64
            img_data = base64.b64encode(file.read()).decode("utf-8")
        motion_stopped = False
    else:
        motion_stopped = False

    if not img_data:
        return jsonify({
            "sentence":      "Waiting for conversation...",
            "confidence":    0.0,
            "history":       [],
            "error":         "No image provided.",
        }), 400

    img = decode_base64_image(img_data)
    if img is None:
        return jsonify({
            "sentence":      "Waiting for conversation...",
            "confidence":    0.0,
            "history":       [],
            "error":         "Invalid image data.",
        }), 400

    # Session ID — use user email if logged in, else a guest key
    session_id: str = "sentence_" + str(session.get("user_email", "guest"))

    try:
        sentence:   str
        confidence: float
        history:    list
        sentence, confidence, history = sentence_prediction_service.predict(
            img, session_id=session_id
        )
    except Exception as exc:
        logger.error(f"[SentenceRoute] Prediction error: {exc}")
        sentence, confidence, history = "Waiting for conversation...", 0.0, []

    # ── Collect debug fields ──────────────────────────────────────────────────
    svc     = sentence_prediction_service
    buf_len = len(svc._buffers.get(session_id, [])) if session_id in svc._buffers else 0
    still   = svc._still_frame_count.get(session_id, 0)
    early   = svc._early_triggered.get(session_id, False)

    # Get latest velocity from MediaPipe service
    _, velocity = sentence_mediapipe_service.get_motion_state(None)  # reads cached value

    # Top-3 labels (best-effort — only available if last inference ran)
    top3: List[str] = []
    if svc._model_ready and svc.classes is not None and svc._prev_raw_label.get(session_id):
        # We don't cache probs, so top3 is derived from vote history + current label
        pass  # top3 left empty for now; full top3 requires caching probs in service

    return jsonify({
        "sentence":      sentence,
        "confidence":    round(confidence, 1),
        "history":       history,
        "buf_len":       buf_len,
        "velocity":      round(velocity, 6),
        "still_frames":  still,
        "top3":          top3,
        "early_trigger": early,
    })


# ── API: Clear Session ────────────────────────────────────────────────────────

@sentence_mode_bp.route("/api/clear-sentence", methods=["POST"])
def clear_sentence() -> Response:
    """Clear session buffer, stable label, and conversation history."""
    session_id: str = "sentence_" + str(session.get("user_email", "guest"))
    sentence_prediction_service.reset_session(session_id)
    sentence_prediction_service.clear_history()
    return jsonify({"success": True, "message": "Conversation cleared."})


# ── API: Status ───────────────────────────────────────────────────────────────

@sentence_mode_bp.route("/api/sentence-status")
def sentence_status() -> Response:
    """Return model state, classes, and optimized configuration parameters."""
    svc = sentence_prediction_service
    return jsonify({
        "model_ready":          svc.is_ready,
        "classes":              svc.get_labels(),
        "sequence_length":      svc.SEQUENCE_LENGTH,
        "confidence_threshold": svc.CONFIDENCE_THRESHOLD,
        "stable_frames":        svc.STABLE_FRAMES_REQUIRED,
        "prediction_window":    svc.PREDICTION_WINDOW,
        "speech_cooldown":      svc.SPEECH_COOLDOWN,
        "velocity_threshold":   svc.VELOCITY_STILL_THRESHOLD,
        "min_early_frames":     svc.MIN_FRAMES_FOR_EARLY_PREDICT,
    })


# ── API: Speak Again ──────────────────────────────────────────────────────────

@sentence_mode_bp.route("/api/speak-again", methods=["POST"])
def speak_again() -> Response:
    """Re-trigger TTS for the last detected sentence."""
    session_id: str = "sentence_" + str(session.get("user_email", "guest"))
    sentence_prediction_service.speak_again(session_id)
    return jsonify({"success": True})
