from app.api.deps import get_request_source


def test_get_request_source_client():
    result = get_request_source()

    assert result == "api"
    assert isinstance(result, str)
