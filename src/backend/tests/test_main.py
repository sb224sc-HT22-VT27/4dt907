"""Tests for backend API."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_hello_v1_endpoint():
    r = client.get("/api/v1/hello")
    assert r.status_code == 200
    data = r.json()
    assert data["api_version"] == "v1"
    assert data["message"] == "Hello World from 4dt907!"


def test_hello_v2_endpoint():
    r = client.get("/api/v2/hello")
    assert r.status_code == 200
    data = r.json()
    assert data["api_version"] == "v2"
    assert data["message"] == "Hello World from 4dt907!"
    assert "server_time_utc" in data
