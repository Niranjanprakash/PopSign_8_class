import io

from app import app
from backend.inference.confidence import get_ambiguity_warning


def client():
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_endpoint():
    response = client().get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_prediction_requires_video_file():
    response = client().post("/api/predict/video")

    assert response.status_code == 400
    assert response.get_json()["error"] == "No video file provided"


def test_prediction_rejects_unsupported_extension():
    response = client().post(
        "/api/predict/video",
        data={"video": (io.BytesIO(b"not a video"), "sample.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "formats are supported" in response.get_json()["error"]


def test_close_bird_airplane_scores_are_marked_ambiguous():
    warning = get_ambiguity_warning(
        [
            {"class_id": 1, "probability": 0.48},
            {"class_id": 2, "probability": 0.43},
        ],
        ["after", "airplane", "bird"],
    )

    assert warning is not None
    assert warning["labels"] == ["airplane", "bird"]
    assert warning["margin"] == 0.05


def test_clear_bird_airplane_scores_remain_decisive():
    warning = get_ambiguity_warning(
        [
            {"class_id": 1, "probability": 0.82},
            {"class_id": 2, "probability": 0.12},
        ],
        ["after", "airplane", "bird"],
    )

    assert warning is None
