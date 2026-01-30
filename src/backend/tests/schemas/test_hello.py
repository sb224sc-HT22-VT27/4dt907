from datetime import datetime, timezone
import pytest

from app.schemas.hello import HelloResponseV1, HelloResponseV2

def test_hello_response_v1_defaults():
    model = HelloResponseV1(message="hello")

    assert model.message == "hello"
    assert model.api_version == "v1"
    assert isinstance(model.api_version, str)

def test_hello_response_v2_success():
    now = datetime.now(timezone.utc)

    model = HelloResponseV2(
        message="hello",
        server_time_utc=now,
    )

    assert model.message == "hello"
    assert model.api_version == "v2"
    assert model.server_time_utc == now
