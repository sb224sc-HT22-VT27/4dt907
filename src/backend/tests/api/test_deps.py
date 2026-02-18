from app.api.deps import get_request_source


def test_get_request_source_client_result():
    result = get_request_source()

    assert result == "api"


def test_get_request_source_client_instance_type():
    result = get_request_source()

    assert isinstance(result, str)
