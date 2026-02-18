from datetime import datetime, timezone

from app.schemas.hello import HelloResponseV1, HelloResponseV2


def test_hello_response_v1_defaults_message():
    model = HelloResponseV1(message="hello")
    assert model.message == "hello"


def test_hello_response_v1_defaults_api_version():
    model = HelloResponseV1(message="hello")
    assert model.api_version == "v1"


def test_hello_response_v1_defaults_instance():
    model = HelloResponseV1(message="hello")
    assert isinstance(model.api_version, str)


def test_hello_response_v2_success_message():
    now = datetime.now(timezone.utc)

    model = HelloResponseV2(
        message="hello",
        server_time_utc=now,
    )

    assert model.message == "hello"


def test_hello_response_v2_success_api_version():
    now = datetime.now(timezone.utc)

    model = HelloResponseV2(
        message="hello",
        server_time_utc=now,
    )

    assert model.api_version == "v2"


def test_hello_response_v2_success_server_time():
    now = datetime.now(timezone.utc)

    model = HelloResponseV2(
        message="hello",
        server_time_utc=now,
    )

    assert model.server_time_utc == now
