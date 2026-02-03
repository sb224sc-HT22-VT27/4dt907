from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert body["message"] == "Backend is running"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"
    assert body["predict_champion"] == "/api/v1/predict/champion"
    assert body["predict_latest"] == "/api/v1/predict/latest"
    assert body["v2_status"] == "/api/v2/status"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
