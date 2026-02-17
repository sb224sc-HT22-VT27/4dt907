import pytest

from app.services import weaklink_model_service
from unittest.mock import MagicMock

"""Test cases for weaklink_model_service module."""

"""_clean_uri tests"""


def test_clean_uri_variants_None():
    assert weaklink_model_service._clean_uri(None) is None


def test_clean_uri_variants_remove_quotation():
    assert weaklink_model_service._clean_uri('  "abc" ') == "abc"


def test_clean_uri_variants_remove_apostrophe():
    assert weaklink_model_service._clean_uri(" 'def' ") == "def"


"""_direct_uri_for_variant tests"""


def test_direct_uri_for_variant_uri_prod_champion(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", " prod_uri ")
    assert weaklink_model_service._direct_uri_for_variant("champion") == "prod_uri"


def test_direct_uri_for_variant_uri_prod_best(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", " prod_uri ")
    assert weaklink_model_service._direct_uri_for_variant("best") == "prod_uri"


def test_direct_uri_for_variant_uri_prod_prod(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", " prod_uri ")
    assert weaklink_model_service._direct_uri_for_variant("prod") == "prod_uri"


def test_direct_uri_for_variant_uri_prod_production(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", " prod_uri ")
    assert weaklink_model_service._direct_uri_for_variant("production") == "prod_uri"


def test_direct_uri_for_variant_uri_dev(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_DEV", " dev_uri ")
    assert weaklink_model_service._direct_uri_for_variant("dev") == "dev_uri"


def test_direct_uri_for_variant_uri_development(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_DEV", " dev_uri ")
    assert weaklink_model_service._direct_uri_for_variant("development") == "dev_uri"


def test_direct_uri_for_variant_uri_latest(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_DEV", " dev_uri ")
    assert weaklink_model_service._direct_uri_for_variant("latest") == "dev_uri"


def test_direct_uri_for_variant_uri_backup(monkeypatch):
    monkeypatch.setenv("WEAKLINK_MODEL_URI_BACKUP", " backup_uri ")
    assert weaklink_model_service._direct_uri_for_variant("backup") == "backup_uri"


def test_direct_uri_for_variant_uri_unknown(monkeypatch):
    assert weaklink_model_service._direct_uri_for_variant("unknown") is None


"""init_mlflow tests"""


def test_init_mlflow_missing_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI is not set"):
        weaklink_model_service._init_mlflow()


def test_init_mlflow_with_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    weaklink_model_service._init_mlflow()


"""_is_models_alias_uri tests"""


def test_is_models_alias_uri_valid_prod():
    assert weaklink_model_service._is_models_alias_uri("models:/MyModel@prod") is True


def test_is_models_alias_uri_valid_dev():
    assert (
        weaklink_model_service._is_models_alias_uri("models:/Project_Model@dev") is True
    )


def test_is_models_alias_uri_valid_latest():
    assert weaklink_model_service._is_models_alias_uri("models:/Model@latest") is True


def test_is_models_alias_uri_invalid_name():
    assert weaklink_model_service._is_models_alias_uri("models:/MyModel/1") is False


def test_is_models_alias_uri_invalid_syntax():
    assert weaklink_model_service._is_models_alias_uri("models:/MyModel") is False


def test_is_models_alias_uri_invalid_service():
    assert weaklink_model_service._is_models_alias_uri("s3://bucket/model") is False


def test_is_models_alias_uri_invalid_path():
    assert weaklink_model_service._is_models_alias_uri("models:/My/Model@prod") is False


"""_parse_models_alias_uri tests"""


def test_parse_models_alias_uri_valid_name_dev():
    name, _ = weaklink_model_service._parse_models_alias_uri(
        "models:/Project_Model@dev"
    )
    assert name == "Project_Model"


def test_parse_models_alias_uri_valid_alias_dev():
    _, alias = weaklink_model_service._parse_models_alias_uri(
        "models:/Project_Model@dev"
    )
    assert alias == "dev"


def test_parse_models_alias_uri_valid_name_prod():
    name, _ = weaklink_model_service._parse_models_alias_uri("models:/MyModel@prod")
    assert name == "MyModel"


def test_parse_models_alias_uri_valid_alias_prod():
    _, alias = weaklink_model_service._parse_models_alias_uri("models:/MyModel@prod")
    assert alias == "prod"


def test_parse_models_alias_uri_with_whitespace_name():
    name, _ = weaklink_model_service._parse_models_alias_uri("models:/ MyModel @ dev ")
    assert name == "MyModel"


def test_parse_models_alias_uri_with_whitespace_alias():
    _, alias = weaklink_model_service._parse_models_alias_uri("models:/ MyModel @ dev ")
    assert alias == "dev"


def test_parse_models_alias_uri_invalid_name():
    with pytest.raises(ValueError):
        weaklink_model_service._parse_models_alias_uri("models:/MyModel@")


def test_parse_models_alias_uri_invalid_alias():
    with pytest.raises(ValueError):
        weaklink_model_service._parse_models_alias_uri("models:/@alias")


"""_resolve_alias_to_version_uri tests"""


def test_resolve_alias_prod_with_production_stage(monkeypatch):
    v1 = MagicMock()
    v1.version = "1"
    v1.current_stage = "None"

    v2 = MagicMock()
    v2.version = "2"
    v2.current_stage = "Production"

    v3 = MagicMock()
    v3.version = "3"
    v3.current_stage = "Staging"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [v1, v2, v3]

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    uri = weaklink_model_service._resolve_alias_to_version_uri("MyModel", "prod")
    assert uri == "models:/MyModel/2"


def test_resolve_alias_prod_without_production_stage(monkeypatch):
    v1 = MagicMock()
    v1.version = "1"
    v1.current_stage = "None"

    v2 = MagicMock()
    v2.version = "2"
    v2.current_stage = "Staging"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [v1, v2]

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    uri = weaklink_model_service._resolve_alias_to_version_uri("MyModel", "production")
    assert uri == "models:/MyModel/2"


def test_resolve_alias_latest(monkeypatch):
    v1 = MagicMock()
    v1.version = "1"

    v2 = MagicMock()
    v2.version = "2"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [v2, v1]

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    uri = weaklink_model_service._resolve_alias_to_version_uri("MyModel", "latest")
    assert uri == "models:/MyModel/2"


def test_resolve_alias_dev(monkeypatch):
    v1 = MagicMock()
    v1.version = "1"

    v2 = MagicMock()
    v2.version = "2"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [v1, v2]

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    uri = weaklink_model_service._resolve_alias_to_version_uri("MyModel", "dev")
    assert uri == "models:/MyModel/2"


def test_resolve_alias_backup(monkeypatch):
    v1 = MagicMock()
    v1.version = "1"

    v2 = MagicMock()
    v2.version = "2"

    v3 = MagicMock()
    v3.version = "3"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [v1, v2, v3]

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    uri = weaklink_model_service._resolve_alias_to_version_uri("MyModel", "backup")
    assert uri == "models:/MyModel/2"


def test_resolve_alias_backup_single_version(monkeypatch):
    v1 = MagicMock()
    v1.version = "1"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [v1]

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    uri = weaklink_model_service._resolve_alias_to_version_uri("MyModel", "backup")
    assert uri == "models:/MyModel/1"


def test_resolve_alias_unknown(monkeypatch):
    v1 = MagicMock()
    v1.version = "1"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [v1]

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    uri = weaklink_model_service._resolve_alias_to_version_uri(
        "MyModel", "unknown_alias"
    )
    assert uri == "models:/MyModel/1"


def test_resolve_alias_no_versions(monkeypatch):
    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = []

    monkeypatch.setattr(weaklink_model_service, "MlflowClient", lambda: fake_client)

    with pytest.raises(RuntimeError, match="No versions found"):
        weaklink_model_service._resolve_alias_to_version_uri("MyModel", "prod")


"""get_model tests"""


def test_get_model_success_model(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", "models:/WeakLink/1")

    fake_model = MagicMock()

    monkeypatch.setattr(
        weaklink_model_service,
        "_load_model_with_alias_fallback",
        lambda uri: (fake_model, uri),
    )

    monkeypatch.setattr(
        weaklink_model_service,
        "_fetch_run_id",
        lambda uri: "run_123",
    )

    weaklink_model_service._cache.clear()

    model, _, _ = weaklink_model_service.get_model("champion")
    assert model == fake_model


def test_get_model_success_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", "models:/WeakLink/1")

    fake_model = MagicMock()

    monkeypatch.setattr(
        weaklink_model_service,
        "_load_model_with_alias_fallback",
        lambda uri: (fake_model, uri),
    )

    monkeypatch.setattr(
        weaklink_model_service,
        "_fetch_run_id",
        lambda uri: "run_123",
    )

    weaklink_model_service._cache.clear()

    _, uri, _ = weaklink_model_service.get_model("champion")
    assert uri == "models:/WeakLink/1"


def test_get_model_uses_cache_load_one_model(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", "models:/WeakLink/1")

    fake_model = MagicMock()
    load_count = [0]

    def mock_load(uri):
        load_count[0] += 1
        return fake_model, uri

    monkeypatch.setattr(
        weaklink_model_service, "_load_model_with_alias_fallback", mock_load
    )

    monkeypatch.setattr(
        weaklink_model_service,
        "_fetch_run_id",
        lambda uri: "run_123",
    )

    weaklink_model_service._cache.clear()

    _, _, _ = weaklink_model_service.get_model("champion")
    assert load_count[0] == 1


def test_get_model_uses_cache_load_second_model(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", "models:/WeakLink/1")

    fake_model = MagicMock()
    load_count = [0]

    def mock_load(uri):
        load_count[0] += 1
        return fake_model, uri

    monkeypatch.setattr(
        weaklink_model_service, "_load_model_with_alias_fallback", mock_load
    )

    monkeypatch.setattr(
        weaklink_model_service,
        "_fetch_run_id",
        lambda uri: "run_123",
    )

    weaklink_model_service._cache.clear()

    _, _, _ = weaklink_model_service.get_model("champion")
    assert load_count[0] == 1


def test_get_model_uses_cache_load_same_models(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", "models:/WeakLink/1")

    fake_model = MagicMock()
    load_count = [0]

    def mock_load(uri):
        load_count[0] += 1
        return fake_model, uri

    monkeypatch.setattr(
        weaklink_model_service, "_load_model_with_alias_fallback", mock_load
    )

    monkeypatch.setattr(
        weaklink_model_service,
        "_fetch_run_id",
        lambda uri: "run_123",
    )

    weaklink_model_service._cache.clear()

    model1, _, _ = weaklink_model_service.get_model("champion")

    model2, _, _ = weaklink_model_service.get_model("champion")
    assert model1 == model2


def test_get_model_no_uri_configured(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.delenv("WEAKLINK_MODEL_URI_PROD", raising=False)

    weaklink_model_service._cache.clear()

    with pytest.raises(RuntimeError, match="Weaklink model URI is not set"):
        weaklink_model_service.get_model("champion")


def test_get_model_missing_mlflow_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("WEAKLINK_MODEL_URI_PROD", "models:/WeakLink/1")

    weaklink_model_service._cache.clear()

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI is not set"):
        weaklink_model_service.get_model("champion")
