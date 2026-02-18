def test_health_status_code(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_content(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
