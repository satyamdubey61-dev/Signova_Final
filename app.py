"""
Flask backend for Signova - Sign Language Detection.
Refactored to be modular and clean.
"""
import os
import sys
import subprocess

# Auto-restart with the correct virtual environment Python interpreter if not already running in it
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    print(f"[INFO] Auto-restarting with the correct virtual environment: {venv_python}")
    sys.exit(subprocess.call([venv_python, *sys.argv]))

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv() -> None:
    env_path: str = os.path.join(BASE_DIR, ".env")
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
from flask import Flask, Response, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402

from core.database import init_users_db  # noqa: E402
from routes.auth import auth_bp  # noqa: E402
from routes.predict import predict_bp  # noqa: E402
from routes.external import external_bp  # noqa: E402
from routes.features import features_bp  # noqa: E402
from utils.error_handlers import register_error_handlers  # noqa: E402
from utils.logger import logger  # noqa: E402

# Ensure DB exists
init_users_db()

app: Flask = Flask(__name__, static_folder="static", static_url_path="")
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
def index() -> Response:
    static_dir: str = app.static_folder or "static"
    return send_from_directory(static_dir, "index.html")


if __name__ == "__main__":
    os.environ.setdefault("FLASK_SKIP_DOTENV", "1")
    logger.info("Starting Signova backend. Open http://127.0.0.1:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=True)
