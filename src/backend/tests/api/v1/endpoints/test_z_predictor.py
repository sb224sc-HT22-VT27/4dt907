from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.api.v1.endpoints.z_predictor import router as z_predictor_router


def create_test_app():
    app = FastAPI()
    app.include_router(z_predictor_router, prefix="/api/v1")
    return app


def test_z_predictor_champion_success_status_code():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.z_predictor.predict_one",
        return_value=(0.12, "models:/ZPredictor/3", "run_123"),
    ):
        response = client.post(
            "/api/v1/z-predictor/champion",
            json={"features": [1.0, 2.0]},
        )

    assert response.status_code == 200


def test_z_predictor_champion_success_response():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.z_predictor.predict_one",
        return_value=(0.12, "models:/ZPredictor/3", "run_123"),
    ):
        response = client.post(
            "/api/v1/z-predictor/champion",
            json={"features": [1.0, 2.0]},
        )

    assert response.json() == {
        "prediction": 0.12,
        "model_uri": "models:/ZPredictor/3",
        "run_id": "run_123",
    }


def test_z_predictor_latest_validation_error_status_code():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.z_predictor.predict_one",
        side_effect=ValueError("Model expects 2 features"),
    ):
        response = client.post(
            "/api/v1/z-predictor/latest",
            json={"features": [1.0]},
        )

    assert response.status_code == 422


def test_z_predictor_latest_service_failure_status_code():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.z_predictor.predict_one",
        side_effect=RuntimeError("MLflow unavailable"),
    ):
        response = client.post(
            "/api/v1/z-predictor/latest",
            json={"features": [1.0, 2.0]},
        )

    assert response.status_code == 503
