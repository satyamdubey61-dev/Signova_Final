from flask import Blueprint, request, jsonify
from services.auth_service import AuthService

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/signup", methods=["POST"])
def signup():
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"success": False, "message": "Name, email and password are required."}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters long."}), 400

    success, message = AuthService.create_user(name, email, password)
    return jsonify({"success": success, "message": message}), 201 if success else 400

@auth_bp.route("/login", methods=["POST"])
def login():
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400

    success, user_or_msg = AuthService.authenticate_user(email, password)
    if not success:
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    return jsonify({"success": True, "message": "Login successful.", "user": user_or_msg})
