import time
import os
import logging
import threading
from collections import defaultdict, deque
from uuid import uuid4
from pathlib import Path
import cv2
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from backend.config import CLASSES, NUM_CLASSES, UPLOAD_DIR, CONFIDENCE_THRESHOLD
from backend.utils import get_device

main_bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

MAX_VIDEO_DURATION_SECONDS = int(os.environ.get("MAX_VIDEO_DURATION_SECONDS", 60))
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("PREDICTION_RATE_LIMIT", 10))
_request_times = defaultdict(deque)
_rate_limit_lock = threading.Lock()


def _rate_limit_exceeded(client_id: str) -> bool:
    now = time.monotonic()
    with _rate_limit_lock:
        timestamps = _request_times[client_id]
        while timestamps and now - timestamps[0] >= RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            return True
        timestamps.append(now)
        return False


def _video_duration_seconds(filepath: Path) -> float:
    capture = cv2.VideoCapture(str(filepath))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            return 0.0
        return frame_count / fps
    finally:
        capture.release()

@main_bp.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "LightMamba-ASL: Efficient Video ASL Recognition",
        "status": "online",
        "supported_classes": CLASSES,
        "classes_count": NUM_CLASSES
    })

@main_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })

@main_bp.route("/api/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "classes": CLASSES
    })

@main_bp.route("/api/model/info", methods=["GET"])
def model_info():
    # We will import prediction service lazily to avoid circular imports during startup
    from backend.services.prediction_service import get_model_details
    details = get_model_details()
    return jsonify(details)

@main_bp.route("/api/predict/video", methods=["POST"])
def predict_video():
    if _rate_limit_exceeded(request.remote_addr or "unknown"):
        return jsonify({"success": False, "error": "Too many prediction requests"}), 429

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
        
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty file name"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".webm"):
        return jsonify({"success": False, "error": "Only .mp4 and .webm formats are supported"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"success": False, "error": "Invalid file name"}), 400

    filepath = (UPLOAD_DIR / f"{uuid4().hex}_{filename}").resolve()
    if not str(filepath).startswith(str(UPLOAD_DIR.resolve())):
        return jsonify({"success": False, "error": "Invalid file path"}), 400
    file.save(str(filepath))

    try:
        duration = _video_duration_seconds(filepath)
        if duration <= 0:
            return jsonify({"success": False, "error": "Uploaded file is not a readable video"}), 400
        if duration > MAX_VIDEO_DURATION_SECONDS:
            return jsonify({
                "success": False,
                "error": f"Video must be {MAX_VIDEO_DURATION_SECONDS} seconds or shorter",
            }), 400

        from backend.services.prediction_service import predict_video_file
        start_time = time.time()
        result = predict_video_file(filepath)
        processing_time_ms = (time.time() - start_time) * 1000

        # Clean up temporary file
        if filepath.exists():
            os.remove(filepath)

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "confidence": float(result["confidence"]),
            "uncertain": result["uncertain"],
            "top_predictions": result["top_predictions"],
            "processing_time_ms": float(round(processing_time_ms, 2))
        })
    except Exception:
        logger.exception("Video inference failed")
        return jsonify({"success": False, "error": "Inference failed"}), 500
    finally:
        if filepath.exists():
            filepath.unlink()
