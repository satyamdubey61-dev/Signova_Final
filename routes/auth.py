from typing import Any, Dict, Tuple, Union

from flask import Blueprint, Response, request, jsonify, session
from services.auth_service import AuthService

auth_bp: Blueprint = Blueprint('auth', __name__)


@auth_bp.route("/signup", methods=["POST"])
def signup() -> Tuple[Response, int] | Response:
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data: Dict[str, Any] = request.get_json() or {}
    name: str = (data.get("name") or "").strip()
    email: str = (data.get("email") or "").strip().lower()
    password: str = str(data.get("password") or "")

    if not name or not email or not password:
        return jsonify({"success": False, "message": "Name, email and password are required."}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters long."}), 400

    success: bool
    message: str
    success, message = AuthService.create_user(name, email, password)
    return jsonify({"success": success, "message": message}), 201 if success else 400


@auth_bp.route("/login", methods=["POST"])
def login() -> Tuple[Response, int] | Response:
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data: Dict[str, Any] = request.get_json() or {}
    email: str = (data.get("email") or "").strip().lower()
    password: str = str(data.get("password") or "")

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400

    success: bool
    user_or_msg: Any
    success, user_or_msg = AuthService.authenticate_user(email, password)
    if not success:
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    # user_or_msg is a Dict at this point (success == True)
    user_dict: Dict[str, Any] = dict(user_or_msg)

    # Store user info in session
    session["user_id"] = user_dict["id"]
    session["user_name"] = user_dict["name"]
    session["user_email"] = user_dict["email"]

    return jsonify({"success": True, "message": "Login successful.", "user": user_dict})


@auth_bp.route("/logout", methods=["POST"])
def logout() -> Response:
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."})


@auth_bp.route("/check-session", methods=["GET"])
def check_session() -> Tuple[Response, int] | Response:
    if "user_id" in session:
        return jsonify({
            "success": True,
            "user": {
                "id": session["user_id"],
                "name": session["user_name"],
                "email": session["user_email"]
            }
        })
    return jsonify({"success": False, "message": "No active session."}), 401
