import os
import logging
import hashlib
import urllib.request
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from backend.routes import main_bp
from backend.utils import verify_paths, get_device
from backend.config import OUTPUT_DIR, CHECKPOINT_DIR

# Download model weights on Render if not present
MODEL_URL = os.environ.get("MODEL_WEIGHTS_URL", "")  # Set this in Render env vars
MODEL_SHA256 = os.environ.get("MODEL_WEIGHTS_SHA256", "").strip().lower()
MODEL_DOWNLOAD_TIMEOUT = 120

logger = logging.getLogger(__name__)

def download_weights():
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists() and MODEL_URL:
        print(f"[STARTUP] Downloading model weights from {MODEL_URL}...")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(".download")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=MODEL_DOWNLOAD_TIMEOUT) as response:
                with open(temporary_path, "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)

            if MODEL_SHA256:
                digest = hashlib.sha256()
                with open(temporary_path, "rb") as downloaded:
                    for chunk in iter(lambda: downloaded.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != MODEL_SHA256:
                    raise ValueError("model weights checksum does not match MODEL_WEIGHTS_SHA256")

            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        print("[STARTUP] Model weights downloaded.")

def create_app():
    download_weights()  # Download model if running on Render
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_BYTES", 100 * 1024 * 1024))
    CORS(app, origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://niranjanprakash.github.io",
        os.environ.get("FRONTEND_URL", ""),   # Vercel URL — set in Render env vars
    ])
    
    # Initialize directory paths
    verify_paths()
    
    # Register blueprints
    app.register_blueprint(main_bp)

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"success": False, "error": "Video file is too large"}), 413

    @app.errorhandler(500)
    def internal_error(_error):
        logger.exception("Unhandled backend error")
        return jsonify({"success": False, "error": "Internal server error"}), 500

    # Serve output files (plots, confusion matrix, metrics) for the React frontend
    @app.route('/outputs/<path:filename>')
    def serve_outputs(filename):
        return send_from_directory(str(OUTPUT_DIR), filename)
    
    return app


# Expose the application for WSGI servers such as Gunicorn.
app = create_app()

if __name__ == "__main__":
    # Pre-check device info on startup
    get_device()
    app.run(host="0.0.0.0", port=5000, debug=True)
