from flask import Blueprint, render_template, request, jsonify, Response
import re
from typing import Dict, Any, List

ai_assistant_bp: Blueprint = Blueprint("ai_assistant", __name__)

# Map lowercase keywords to exact GIF and text instructions
CHAT_RESPONSES = {
    "hello": {
        "text": "To sign **'Hello'** in Indian Sign Language, raise your dominant hand to about forehead level, palm facing outward, and make a polite waving or saluting gesture from left to right. It is a warm, universal sign of respect.",
        "video": "/static/sign_assets/Hello.mp4",
        "title": "Hello"
    },
    "yes": {
        "text": "To sign **'Yes'**, form a soft fist with your dominant hand and tilt it forward and backward from the wrist, mimicking a nodding head. This gesture is clear and highly communicative.",
        "video": "/static/sign_assets/Yes.mp4",
        "title": "Yes"
    },
    "no": {
        "text": "To sign **'No'**, extend your index and middle fingers together, then snap them down quickly to tap against your thumb. This represents a crisp, firm negation.",
        "video": "/static/sign_assets/No.mp4",
        "title": "No"
    },
    "thank you": {
        "text": "To sign **'Thank You'**, bring the flat palm of your dominant hand to your lips, then move it gracefully forward and down toward the person you are thanking. It represents offering gratitude directly from the heart.",
        "video": "/static/sign_assets/ThankYou.mp4",
        "title": "Thank You"
    },
    "thankyou": {
        "text": "To sign **'Thank You'**, bring the flat palm of your dominant hand to your lips, then move it gracefully forward and down toward the person you are thanking. It represents offering gratitude directly from the heart.",
        "video": "/static/sign_assets/ThankYou.mp4",
        "title": "Thank You"
    },
    "sorry": {
        "text": "To sign **'Sorry'**, form a fist (specifically the 'A' shape hand) with your dominant hand and rub it in a gentle circular motion over your chest/heart. This is a sincere expression of regret.",
        "video": "/static/sign_assets/Sorry.mp4",
        "title": "Sorry"
    },
    "help": {
        "text": "To sign **'Help'**, place your closed dominant hand with thumb up (like a thumbs-up sign) on top of the open flat palm of your non-dominant hand, and lift both hands upward together. It represents physically raising or supporting someone.",
        "video": "/static/sign_assets/Help.mp4",
        "title": "Help"
    },
    "i love you": {
        "text": "To sign **'I Love You'** (often abbreviated as ILY), extend your thumb, index finger, and pinky finger of your dominant hand while keeping the middle and ring fingers folded flat against your palm. Raise your hand with palm facing the receiver.",
        "video": "/static/sign_assets/ILoveYou.mp4",
        "title": "I Love You"
    },
    "iloveyou": {
        "text": "To sign **'I Love You'** (often abbreviated as ILY), extend your thumb, index finger, and pinky finger of your dominant hand while keeping the middle and ring fingers folded flat against your palm. Raise your hand with palm facing the receiver.",
        "video": "/static/sign_assets/ILoveYou.mp4",
        "title": "I Love You"
    },
    "a": {
        "text": "For the alphabet **'A'**, form a closed fist with your dominant hand, keeping your thumb resting flat against the side of your index finger. This is the foundational letter of the ISL manual alphabet.",
        "video": "/static/sign_assets/A.mp4",
        "title": "Alphabet A"
    },
    "c": {
        "text": "For the alphabet **'C'**, curve your dominant hand into a semi-circle shape, resembling the letter 'C' itself. Keep your fingers slightly separated and palm facing sideways.",
        "video": "/static/sign_assets/C.mp4",
        "title": "Alphabet C"
    },
    "i": {
        "text": "For the alphabet **'I'**, raise only your pinky finger straight up into the air, while keeping your thumb folded over your other clenched fingers.",
        "video": "/static/sign_assets/I.mp4",
        "title": "Alphabet I"
    },
    "v": {
        "text": "For the alphabet **'V'**, raise your index and middle fingers in a spread 'V' shape, mimicking a peace sign or the literal letter shape, while folding your other fingers flat.",
        "video": "/static/sign_assets/V.mp4",
        "title": "Alphabet V"
    }
}

@ai_assistant_bp.route("/ai-assistant")
def ai_assistant_page() -> Response:
    """Serve the conversational AI Sign Mentor interface."""
    return render_template("ai_assistant.html")

@ai_assistant_bp.route("/api/ai-chat", methods=["POST"])
def ai_chat() -> Response:
    """
    Handle conversation messages. Intelligently scans user queries
    for ISL vocabulary terms and returns customized guide bubbles containing
    embedded animation links and rich descriptions.
    """
    data = request.get_json() or {}
    message = (data.get("message") or "").strip().lower()
    
    if not message:
        return jsonify({"success": False, "reply": "I'm listening! Please type a question."}), 400

    # Look for vocabulary matches
    found_key = None
    for key in CHAT_RESPONSES.keys():
        # Match whole word to avoid false positives (e.g. matching "a" in "hello")
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, message) or key == message:
            found_key = key
            break

    if found_key:
        resp = CHAT_RESPONSES[found_key]
        reply = (
            f"Hello! I am your AI Sign Mentor. I can certainly help you learn that! \n\n"
            f"{resp['text']}\n\n"
            f"I've attached the holographic visual reference for **'{resp['title']}'** below to guide your practice."
        )
        return jsonify({
            "success": True,
            "reply": reply,
            "embed": {
                "type": "video",
                "url": resp["video"],
                "title": resp["title"]
            }
        })
        
    # If no specific sign matched, check for generic greeting, tips or listing requests
    if any(greet in message for greet in ["hello", "hi", "hey", "greet"]):
        reply = (
            "Hello there! ⚡ I am your **Signova AI Assistant**, dedicated to helping you master Indian Sign Language (ISL).\n\n"
            "You can ask me how to perform specific signs like **'Hello'**, **'Sorry'**, **'Thank You'**, or alphabets like **'A'**, **'C'**, **'I'**, and **'V'**.\n\n"
            "What would you like to practice today?"
        )
    elif "tip" in message or "advice" in message or "how to learn" in message:
        reply = (
            "Here is my top **Sign Mentor Tip** for today:\n\n"
            "1. **Stabilize your hand posture**: Keep your wrist relaxed but steady when starting a sign.\n"
            "2. **Use our Gesture Practice module**: Open the Learn Hub, start your webcam, and let our real-time AI validate your wrist and finger shapes.\n"
            "3. **Practice sentences sequentially**: Don't just learn isolated words—try practicing full concepts like *'Hello How Are You'*!"
        )
    else:
        reply = (
            "I hear you! I'm designed specifically to teach you Indian Sign Language (ISL).\n\n"
            "Try asking me: \n"
            "- *'How do I sign Thank You?'*\n"
            "- *'Can you show me the gesture for Sorry?'*\n"
            "- *'Show me the alphabet V'* \n\n"
            "I'll explain the movement and show you a visual animation immediately!"
        )
        
    return jsonify({"success": True, "reply": reply, "embed": None})
