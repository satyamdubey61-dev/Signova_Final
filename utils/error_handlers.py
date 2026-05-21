from typing import Any, Dict, Optional, Tuple

from flask import Flask, Response, jsonify
from utils.logger import logger


def register_error_handlers(app: Flask) -> None:
    """Registers standard HTTP error handlers on the Flask app instance."""

    @app.errorhandler(400)
    def bad_request(error: Exception) -> Tuple[Response, int]:
        return jsonify({"success": False, "error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error: Exception) -> Tuple[Response, int]:
        return jsonify({"success": False, "error": "Not Found", "message": "The requested URL was not found."}), 404

    @app.errorhandler(500)
    def internal_error(error: Exception) -> Tuple[Response, int]:
        logger.error(f"Internal Server Error: {error}")
        return jsonify({"success": False, "error": "Internal Server Error", "message": "An unexpected error occurred."}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e: Exception) -> Tuple[Response, int]:
        logger.exception("Unhandled exception occurred")
        return jsonify({"success": False, "error": "Server Error", "message": str(e)}), 500
