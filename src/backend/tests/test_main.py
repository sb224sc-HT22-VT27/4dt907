from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

response = client.get("/")
body = response.json()


def test_root_endpoint_status_code():
    assert response.status_code == 200


def test_root_endpoint_body_message():
    assert body["message"] == "Backend is running"


def test_root_endpoint_body_docs():
    assert body["docs"] == "/docs"


def test_root_endpoint_body_health():
    assert body["health"] == "/health"


def test_root_endpoint_body_predict_champion():
    assert body["predict_champion"] == "/api/v1/predict/champion"


def test_root_endpoint_body_predict_latest():
    assert body["predict_latest"] == "/api/v1/predict/latest"


def test_root_endpoint_body_v2_status():
    assert body["v2_status"] == "/api/v2/status"


def test_health_endpoint_status_code():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_response():
    response = client.get("/health")
    assert response.json()["status"] == "ok"
