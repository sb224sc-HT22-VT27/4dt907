"""Tests for backend API."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns hello world message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Hello World" in response.json()["message"]


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "backend"


def test_hello_name():
    """Test personalized hello endpoint."""
    name = "TestUser"
    response = client.get(f"/api/v1/hello/{name}")
    assert response.status_code == 200
    assert name in response.json()["message"]
    assert "Welcome to 4dt907" in response.json()["message"]
