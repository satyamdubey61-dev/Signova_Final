from flask import jsonify
from utils.logger import logger

def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"success": False, "error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"success": False, "error": "Not Found", "message": "The requested URL was not found."}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal Server Error: {error}")
        return jsonify({"success": False, "error": "Internal Server Error", "message": "An unexpected error occurred."}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        logger.exception("Unhandled exception occurred")
        return jsonify({"success": False, "error": "Server Error", "message": str(e)}), 500
