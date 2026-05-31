from flask import Blueprint, render_template, jsonify, Response
import random
from typing import Any, Dict, List

learn_bp: Blueprint = Blueprint("learn", __name__)

# Core vocabulary supported by our models and assets
LETTERS = ["A", "C", "I", "V"]
COMMON_WORDS = ["Hello", "Yes", "No", "ThankYou", "Sorry", "Help", "ILoveYou"]
ALL_VOCAB = LETTERS + COMMON_WORDS

# Display titles for user friendliness
VOCAB_DISPLAY = {
    "A": "Alphabet A",
    "C": "Alphabet C",
    "I": "Alphabet I",
    "V": "Alphabet V",
    "Hello": "Hello",
    "Yes": "Yes",
    "No": "No",
    "ThankYou": "Thank You",
    "Sorry": "Sorry",
    "Help": "Help",
    "ILoveYou": "I Love You"
}

@learn_bp.route("/learn")
def learn_page() -> Response:
    """Serve the interactive AI Learning Hub page."""
    return render_template("learn.html")

@learn_bp.route("/api/quiz-questions", methods=["GET"])
def quiz_questions() -> Response:
    """
    Generate a dynamic, randomized set of 5 multiple-choice quiz questions
    tailored to the real ISL signs available in the application.
    """
    questions: List[Dict[str, Any]] = []
    
    # 5 questions total
    for q_idx in range(5):
        # Choose a random correct answer from all vocabulary
        correct_item = random.choice(ALL_VOCAB)
        
        # Pick 3 unique wrong options
        options_pool = [item for item in ALL_VOCAB if item != correct_item]
        wrong_options = random.sample(options_pool, 3)
        
        # Merge and shuffle all options
        options = [correct_item] + wrong_options
        random.shuffle(options)
        
        # Format options as friendly names
        friendly_options = [VOCAB_DISPLAY.get(opt, opt) for opt in options]
        correct_index = options.index(correct_item)
        
        # 50% chance for visual sign-matching or word-matching question
        if random.choice([True, False]):
            q_type = "gif_to_word"
            question_text = "What word or letter does this Indian Sign Language gesture represent?"
            # Asset URL mapping
            video_filename = f"{correct_item}.mp4"
            asset_url = f"/static/sign_assets/{video_filename}"
        else:
            q_type = "word_to_gif"
            question_text = f"Which of the following gestures corresponds to '{VOCAB_DISPLAY.get(correct_item, correct_item)}'?"
            asset_url = None # Users will choose between options, or we display a static clue
            
        questions.append({
            "id": q_idx + 1,
            "type": q_type,
            "question": question_text,
            "correct_item": correct_item,
            "correct_display": VOCAB_DISPLAY.get(correct_item, correct_item),
            "options": friendly_options,
            "raw_options": options,
            "correct_index": correct_index,
            "asset_url": asset_url
        })
        
    return jsonify({"success": True, "questions": questions})
