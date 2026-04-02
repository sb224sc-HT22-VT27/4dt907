"""Tests for the /api/v1/squat/classify endpoint."""
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.api.v1.endpoints.squat import router as squat_router


def create_test_app():
    app = FastAPI()
    app.include_router(squat_router, prefix="/api/v1")
    return app


def _make_kp3(name, x, y, z):
    return {"name": name, "x": x, "y": y, "z": z}


def _make_kp2(name, x, y):
    return {"name": name, "x": x, "y": y}


def _deep_squat_payload():
    r = math.radians(80)
    kp3 = [
        _make_kp3("left_hip", 0, 1, 0),
        _make_kp3("left_knee", 0, 0, 0),
        _make_kp3("left_ankle", math.sin(r), -math.cos(r), 0),
        _make_kp3("right_hip", 0, 1, 0),
        _make_kp3("right_knee", 0, 0, 0),
        _make_kp3("right_ankle", math.sin(r), -math.cos(r), 0),
    ]
    kp2 = [_make_kp2(k["name"], k["x"], k["y"]) for k in kp3]
    return {"keypoints_3d": kp3, "keypoints_2d": kp2}


def test_squat_classify_status_code():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.squat.classify_squat",
        return_value=("Deep", 80.0, 80.0, 0.9),
    ):
        response = client.post("/api/v1/squat/classify", json=_deep_squat_payload())

    assert response.status_code == 200


def test_squat_classify_response_shape():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.squat.classify_squat",
        return_value=("Deep", 80.0, 82.0, 0.9),
    ):
        response = client.post("/api/v1/squat/classify", json=_deep_squat_payload())

    body = response.json()
    assert body["classification"] == "Deep"
    assert body["left_knee_angle"] == 80.0
    assert body["right_knee_angle"] == 82.0
    assert body["confidence"] == 0.9


def test_squat_classify_invalid_payload_status_code():
    app = create_test_app()
    client = TestClient(app)

    response = client.post("/api/v1/squat/classify", json={"wrong_key": []})
    assert response.status_code == 422


def test_squat_classify_service_error_returns_503():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.squat.classify_squat",
        side_effect=RuntimeError("model exploded"),
    ):
        response = client.post("/api/v1/squat/classify", json=_deep_squat_payload())

    assert response.status_code == 503


def test_squat_classify_no_confidence_allowed():
    """confidence field is optional — None should serialise as null."""
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.squat.classify_squat",
        return_value=("Invalid", 0.0, 0.0, None),
    ):
        response = client.post("/api/v1/squat/classify", json=_deep_squat_payload())

    assert response.status_code == 200
    assert response.json()["confidence"] is None
