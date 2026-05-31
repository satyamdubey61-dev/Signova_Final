"""Routes for text-to-sign and translation features."""
import os
import re
import random
import urllib.request
import urllib.parse
import json
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Response, request, jsonify, send_from_directory

features_bp: Blueprint = Blueprint('features', __name__)

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Expanded offline translation dictionary (covering all 11 target gestures) ---
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "hindi": {
        "hello": "नमस्ते", 
        "yes": "हाँ",
        "no": "नहीं", 
        "thank you": "धन्यवाद", 
        "thankyou": "धन्यवाद",
        "sorry": "माफ़ करें", 
        "help": "मदद",
        "i love you": "मैं तुमसे प्यार करता हूँ",
        "iloveyou": "मैं तुमसे प्यार करता हूँ",
        "a": "ए",
        "c": "सी",
        "v": "वी",
        "i": "आई"
    },
    "marathi": {
        "hello": "नमस्कार", 
        "yes": "हो",
        "no": "नाही", 
        "thank you": "धन्यवाद", 
        "thankyou": "धन्यवाद",
        "sorry": "माफ करा", 
        "help": "मदत",
        "i love you": "मी तुझ्यावर प्रेम करतो",
        "iloveyou": "मी तुझ्यावर प्रेम करतो",
        "a": "ए",
        "c": "सी",
        "v": "वी",
        "i": "आय"
    },
    "konkani": {
        "hello": "नमस्कार", 
        "yes": "हय",
        "no": "ना", 
        "thank you": "देव बरें करूं", 
        "thankyou": "देव बरें करूं",
        "sorry": "माफ करात", 
        "help": "कुमक",
        "i love you": "हांव तुका मोगाचो",
        "iloveyou": "हांव तुका मोगाचो",
        "a": "ए",
        "c": "सी",
        "v": "वी",
        "i": "आय"
    },
    "tamil": {
        "hello": "வணக்கம்", 
        "yes": "ஆம்",
        "no": "இல்லை", 
        "thank you": "நன்றி", 
        "thankyou": "நன்றி",
        "sorry": "மன்னிக்கவும்", 
        "help": "உதவி",
        "i love you": "நான் உன்னை காதலிக்கிறேன்",
        "iloveyou": "நான் உன்னை காதலிக்கிறேன்",
        "a": "ஏ",
        "c": "சி",
        "v": "வி",
        "i": "ஐ"
    }
}


def _normalize_name(name: str) -> str:
    """Normalize input strings by lowercasing, stripping whitespace and special chars."""
    clean = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
    return clean.strip()


def translate_via_mymemory(text: str, to_lang: str) -> Optional[str]:
    """Fetch translation from MyMemory API with a tight 2s timeout."""
    try:
        encoded_text = urllib.parse.quote(text)
        url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair=en|{to_lang}"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=2.0) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            if res_data.get("responseStatus") == 200:
                translated = res_data.get("responseData", {}).get("translatedText")
                if translated and translated.strip():
                    return translated.strip()
    except Exception as e:
        print(f"[Translation API Warning] MyMemory failed: {e}")
    return None


def _find_sign_asset(word: str) -> Optional[Tuple[str, str]]:
    """
    Search inside static/sign_assets/ for a file matching the word.
    Supports .gif, .png, .jpg, .mp4.
    Returns (relative_file_path, file_type) or None.
    """
    assets_dir = os.path.join(BASE_DIR, "static", "sign_assets")
    if not os.path.isdir(assets_dir):
        return None
        
    target = _normalize_name(word)
    if not target:
        return None
        
    supported_extensions = {
        '.gif': 'image',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.mp4': 'video',
        '.webm': 'video'
    }
    
    for filename in os.listdir(assets_dir):
        base, ext = os.path.splitext(filename)
        if ext.lower() in supported_extensions:
            if _normalize_name(base) == target:
                relative_path = f"sign_assets/{filename}"
                return relative_path, supported_extensions[ext.lower()]
                
    return None


@features_bp.route("/text-to-sign", methods=["POST"])
def text_to_sign() -> Tuple[Response, int] | Response:
    data: Dict[str, Any] = request.get_json() or {}
    word: str = (data.get("text") or "").strip()
    if not word:
        return jsonify({"success": False, "message": "No text provided."}), 400

    asset_info = _find_sign_asset(word)
    if not asset_info:
        return jsonify({"success": False, "message": "Animated sign not available"}), 404

    relative_path, asset_type = asset_info
    return jsonify({
        "success": True, 
        "url": f"/{relative_path}", 
        "type": asset_type
    })


from services.translation_service import TranslationService

@features_bp.route("/translate", methods=["POST"])
def translate() -> Tuple[Response, int] | Response:
    data: Dict[str, Any] = request.get_json() or {}
    text: str = (data.get("text") or "").strip()
    lang: str = (data.get("language") or "").strip().lower()

    if not text:
        return jsonify({"success": False, "message": "No text provided."}), 400

    # Translate using static dictionary TranslationService
    translated = TranslationService.translate(text, lang)

    return jsonify({"success": True, "translated": translated})

