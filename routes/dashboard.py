from flask import Blueprint, render_template, jsonify, Response
from typing import Any, Dict

dashboard_bp: Blueprint = Blueprint("dashboard", __name__)

@dashboard_bp.route("/dashboard")
def dashboard_page() -> Response:
    """Serve the progress tracking Dashboard."""
    return render_template("dashboard.html")

@dashboard_bp.route("/api/progress-summary", methods=["GET"])
def progress_summary() -> Response:
    """
    Expose dynamic summary indicators for the dashboard.
    Works hand-in-hand with browser localStorage to sync user achievements.
    """
    return jsonify({
        "success": True,
        "default_stats": {
            "streak": 3,
            "accuracy_avg": 92.5,
            "cards_learned": 4, # out of 11
            "quizzes_taken": 2,
            "recent_lessons": [
                {"title": "Greetings & Basics", "date": "Today", "status": "Completed"},
                {"title": "Alphabet Essentials", "date": "Yesterday", "status": "Completed"},
                {"title": "Emergency Signs", "date": "3 days ago", "status": "In Progress"}
            ]
        }
    })
