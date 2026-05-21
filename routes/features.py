"""Routes for text-to-sign and translation features."""
import os
import re
import random
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Response, request, jsonify, send_file

features_bp: Blueprint = Blueprint('features', __name__)

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "Data")

# --- Translation dictionary (offline, no external API needed) ---
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "hindi": {
        "hello": "नमस्ते", "thank you": "धन्यवाद", "yes": "हाँ",
        "no": "नहीं", "i love you": "मैं तुमसे प्यार करता हूँ",
        "please": "कृपया", "sorry": "माफ़ करें", "help": "मदद",
        "good": "अच्छा", "bad": "बुरा", "water": "पानी",
        "food": "खाना", "friend": "दोस्त", "family": "परिवार",
        "school": "विद्यालय", "home": "घर", "name": "नाम",
    },
    "marathi": {
        "hello": "नमस्कार", "thank you": "धन्यवाद", "yes": "हो",
        "no": "नाही", "i love you": "मी तुझ्यावर प्रेम करतो",
        "please": "कृपया", "sorry": "माफ करा", "help": "मदत",
        "good": "चांगले", "bad": "वाईट", "water": "पाणी",
        "food": "अन्न", "friend": "मित्र", "family": "कुटुंब",
        "school": "शाळा", "home": "घर", "name": "नाव",
    },
    "konkani": {
        "hello": "नमस्कार", "thank you": "देव बरें करूं", "yes": "हय",
        "no": "ना", "i love you": "हांव तुका मोगाचो",
        "please": "उपकार करून", "sorry": "माफ करात", "help": "कुमक",
        "good": "बरें", "bad": "वायट", "water": "उदक",
        "food": "जेवण", "friend": "इश्ट", "family": "कुटुंब",
        "school": "इस्कोल", "home": "घर", "name": "नांव",
    },
    "tamil": {
        "hello": "வணக்கம்", "thank you": "நன்றி", "yes": "ஆம்",
        "no": "இல்லை", "i love you": "நான் உன்னை காதலிக்கிறேன்",
        "please": "தயவுசெய்து", "sorry": "மன்னிக்கவும்", "help": "உதவி",
        "good": "நல்ல", "bad": "கெட்ட", "water": "தண்ணீர்",
        "food": "உணவு", "friend": "நண்பன்", "family": "குடும்பம்",
        "school": "பள்ளி", "home": "வீடு", "name": "பெயர்",
    },
}


def _find_sign_image(word: str) -> Optional[str]:
    """Find a sample image from Data/<Word>/ folder. Case-insensitive match."""
    if not os.path.isdir(DATA_DIR):
        return None

    # Build a map of lowercase folder name -> actual folder name
    folder_map: Dict[str, str] = {}
    for entry in os.listdir(DATA_DIR):
        full: str = os.path.join(DATA_DIR, entry)
        if os.path.isdir(full):
            normalized: str = re.sub(r'[^\w\s]', '', entry.lower())
            normalized = ' '.join(normalized.split())
            folder_map[normalized] = full

    target: str = re.sub(r'[^\w\s]', '', word.lower())
    target = ' '.join(target.split())
    folder: Optional[str] = folder_map.get(target)
    if not folder:
        return None

    images: List[str] = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        return None

    return os.path.join(folder, random.choice(images))


@features_bp.route("/text-to-sign", methods=["POST"])
def text_to_sign() -> Tuple[Response, int] | Response:
    data: Dict[str, Any] = request.get_json() or {}
    word: str = (data.get("text") or "").strip()
    if not word:
        return jsonify({"success": False, "message": "No text provided."}), 400

    img_path: Optional[str] = _find_sign_image(word)
    if not img_path:
        return jsonify({"success": False, "message": f"Word '{word}' not found in dataset."}), 404

    return send_file(img_path, mimetype="image/jpeg")


@features_bp.route("/translate", methods=["POST"])
def translate() -> Tuple[Response, int] | Response:
    data: Dict[str, Any] = request.get_json() or {}
    text: str = (data.get("text") or "").strip().lower()
    lang: str = (data.get("language") or "").strip().lower()

    if not text:
        return jsonify({"success": False, "message": "No text provided."}), 400

    if lang == "english":
        return jsonify({"success": True, "translated": text.title()})

    lang_dict: Optional[Dict[str, str]] = TRANSLATIONS.get(lang)
    if not lang_dict:
        return jsonify({"success": False, "message": f"Language '{lang}' not supported."}), 400

    translated: Optional[str] = lang_dict.get(text)
    if translated:
        return jsonify({"success": True, "translated": translated})

    return jsonify({"success": True, "translated": f"{text} ({lang} translation unavailable)"})
