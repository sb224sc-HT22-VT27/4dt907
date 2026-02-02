from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock

from app.api.v1.endpoints.model_info import router as model_info_router


def create_test_app():
    app = FastAPI()
    app.include_router(model_info_router, prefix="/api/v1")
    return app


def test_model_info_latest():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.model_info.get_model",
        return_value=(MagicMock(), "models:/Latest/5"),
    ), patch(
        "app.api.v1.endpoints.model_info.expected_feature_count",
        return_value=4,
    ):
        response = client.get("/api/v1/model-info/latest")

    assert response.status_code == 200
    assert response.json() == {
        "variant": "latest",
        "model_uri": "models:/Latest/5",
        "expected_features": 4,
    }


def test_model_info_champion():
    app = create_test_app()
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.model_info.get_model",
        return_value=(MagicMock(), "models:/Champion/2"),
    ), patch(
        "app.api.v1.endpoints.model_info.expected_feature_count",
        return_value=3,
    ):
        response = client.get("/api/v1/model-info/champion")

    assert response.status_code == 200
    assert response.json() == {
        "variant": "champion",
        "model_uri": "models:/Champion/2",
        "expected_features": 3,
    }
