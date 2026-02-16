from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock
from app.api.v1.endpoints.model_info import router as model_info_router


def create_test_app():
    app = FastAPI()
    app.include_router(model_info_router, prefix="/api/v1")
    return app


def test_model_info_weakest_link_latest_status_code():
    app = create_test_app()
    client = TestClient(app)
    with patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.get_model",
        return_value=(MagicMock(), "models:/WeakestLink_Latest/3"),
    ), patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.expected_feature_count",
        return_value=5,
    ):
        response = client.get("/api/v1/model-info/weakest-link/latest")
        assert response.status_code == 200


def test_model_info_weakest_link_latest_response():
    app = create_test_app()
    client = TestClient(app)
    with patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.get_model",
        return_value=(MagicMock(), "models:/WeakestLink_Latest/3"),
    ), patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.expected_feature_count",
        return_value=5,
    ):
        response = client.get("/api/v1/model-info/weakest-link/latest")
        assert response.json() == {
            "variant": "latest",
            "model_uri": "models:/WeakestLink_Latest/3",
            "expected_features": 5,
        }


def test_model_info_weakest_link_champion_status_code():
    app = create_test_app()
    client = TestClient(app)
    with patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.get_model",
        return_value=(MagicMock(), "models:/WeakestLink_Champion/1"),
    ), patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.expected_feature_count",
        return_value=5,
    ):
        response = client.get("/api/v1/model-info/weakest-link/champion")
        assert response.status_code == 200


def test_model_info_weakest_link_champion_response():
    app = create_test_app()
    client = TestClient(app)
    with patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.get_model",
        return_value=(MagicMock(), "models:/WeakestLink_Champion/1"),
    ), patch(
        "app.api.v1.endpoints.model_info.weaklink_model_service.expected_feature_count",
        return_value=5,
    ):
        response = client.get("/api/v1/model-info/weakest-link/champion")
        assert response.json() == {
            "variant": "champion",
            "model_uri": "models:/WeakestLink_Champion/1",
            "expected_features": 5,
        }
