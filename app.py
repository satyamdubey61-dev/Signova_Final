"""
Flask backend for Signova - Sign Language Detection.
Refactored to be modular and clean.
"""
import os
import sys
import subprocess

# Auto-restart with the correct embedded Python environment
embedded_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python310_embed", "python.exe")
if os.path.exists(embedded_python) and sys.executable != embedded_python:
    print(f"[INFO] Auto-restarting with the correct Python environment: {embedded_python}")
    sys.exit(subprocess.call([embedded_python, *sys.argv]))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv():
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and os.environ.get(key) in (None, ""):
                    os.environ[key] = value

_load_dotenv()

# pyrefly: ignore [missing-import]
from flask import Flask, send_from_directory
from flask_cors import CORS

from core.database import init_users_db
from routes.auth import auth_bp
from routes.predict import predict_bp
from routes.external import external_bp
from routes.features import features_bp
from utils.error_handlers import register_error_handlers
from utils.logger import logger

# Ensure DB exists
init_users_db()

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("SIGNIFYCONNECT_SECRET_KEY", "dev-secret-key")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Register Blueprints
app.register_blueprint(auth_bp, url_prefix="/api")
app.register_blueprint(predict_bp, url_prefix="/api")
app.register_blueprint(external_bp, url_prefix="/api")
app.register_blueprint(features_bp, url_prefix="/api")

# Register Error Handlers
register_error_handlers(app)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    os.environ.setdefault("FLASK_SKIP_DOTENV", "1")
    logger.info("Starting Signova backend. Open http://127.0.0.1:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=True)
