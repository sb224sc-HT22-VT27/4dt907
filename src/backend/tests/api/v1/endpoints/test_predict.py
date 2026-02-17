from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch
from app.api.v1.endpoints.predict import router as predict_router


def create_test_app():
    app = FastAPI()
    app.include_router(predict_router, prefix="/api/v1")
    return app


def test_predict_champion_success_status_code():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.predict.predict_one",
        return_value=(0.88, "models:/Champion/3", "run_123"),
    ):
        response = client.post(
            "/api/v1/predict/champion",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 200


def test_predict_champion_success_response():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.predict.predict_one",
        return_value=(0.88, "models:/Champion/3", "run_123"),
    ):
        response = client.post(
            "/api/v1/predict/champion",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert response.json() == {
        "prediction": 0.88,
        "model_uri": "models:/Champion/3",
        "run_id": "run_123",
    }


def test_predict_latest_validation_error_status_code():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.predict.predict_one",
        side_effect=ValueError("Model expects 3 features"),
    ):
        response = client.post(
            "/api/v1/predict/latest",
            json={"features": [1.0]},
        )

    assert response.status_code == 422


def test_predict_latest_validation_error_response():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.predict.predict_one",
        side_effect=ValueError("Model expects 3 features"),
    ):
        response = client.post(
            "/api/v1/predict/latest",
            json={"features": [1.0]},
        )

    assert "Model expects 3 features" in response.json()["detail"]


def test_predict_latest_service_failure_status_code():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.predict.predict_one",
        side_effect=RuntimeError("MLflow unavailable"),
    ):
        response = client.post(
            "/api/v1/predict/latest",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 503


def test_predict_latest_service_failure_response():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.predict.predict_one",
        side_effect=RuntimeError("MLflow unavailable"),
    ):
        response = client.post(
            "/api/v1/predict/latest",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert "RuntimeError" in response.json()["detail"]
