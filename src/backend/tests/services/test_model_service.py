import pytest

from app.services import model_service
from unittest.mock import MagicMock

"""Test cases for model_service module."""

"""_clean_uri tests"""


def test_model_clean_uri_variants_None():
    assert model_service._clean_uri(None) is None


def test_model_clean_uri_variants_remove_quotation():
    assert model_service._clean_uri('  "abc" ') == 'abc'


def test_model_clean_uri_variants_remove_apostrophe():
    assert model_service._clean_uri(" 'def' ") == "def"


"""_direct_uri_for_variant tests"""


def test_model_direct_uri_for_variant_uri_prod_champion(monkeypatch):
    monkeypatch.setenv("MODEL_URI_PROD", " prod_uri ")
    assert model_service._direct_uri_for_variant("champion") == "prod_uri"


def test_model_direct_uri_for_variant_uri_prod_best(monkeypatch):
    monkeypatch.setenv("MODEL_URI_PROD", " prod_uri ")
    assert model_service._direct_uri_for_variant("best") == "prod_uri"


def test_model_direct_uri_for_variant_uri_prod_prod(monkeypatch):
    monkeypatch.setenv("MODEL_URI_PROD", " prod_uri ")
    assert model_service._direct_uri_for_variant("prod") == "prod_uri"


def test_model_direct_uri_for_variant_uri_prod_production(monkeypatch):
    monkeypatch.setenv("MODEL_URI_PROD", " prod_uri ")
    assert model_service._direct_uri_for_variant("production") == "prod_uri"


def test_model_direct_uri_for_variant_uri_dev(monkeypatch):
    monkeypatch.setenv("MODEL_URI_DEV", " dev_uri ")
    assert model_service._direct_uri_for_variant("dev") == "dev_uri"


def test_model_direct_uri_for_variant_uri_development(monkeypatch):
    monkeypatch.setenv("MODEL_URI_DEV", " dev_uri ")
    assert model_service._direct_uri_for_variant("development") == "dev_uri"


def test_model_direct_uri_for_variant_uri_latest(monkeypatch):
    monkeypatch.setenv("MODEL_URI_DEV", " dev_uri ")
    assert model_service._direct_uri_for_variant("latest") == "dev_uri"


def test_model_direct_uri_for_variant_uri_backup(monkeypatch):
    monkeypatch.setenv("MODEL_URI_BACKUP", " backup_uri ")
    assert model_service._direct_uri_for_variant("backup") == "backup_uri"


def test_model_direct_uri_for_variant_uri_unknown(monkeypatch):
    assert model_service._direct_uri_for_variant("unknown") is None


"""init_mlflow tests."""


def test_init_mlflow_request_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError):
        model_service._init_mlflow()


def test_init_mlflow_sets_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")

    monkeypatch.setattr(model_service.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(model_service.mlflow, "set_registry_uri", lambda uri: None)

    uri = model_service._init_mlflow()
    assert uri == "http://mlflow:5050"


"""_model_name_for_variant tests."""


def test_model_name_for_variant_champion(monkeypatch):
    monkeypatch.setenv("MLFLOW_BEST_MODEL_NAME", "best_model")
    assert model_service._model_name_for_variant("champion") == "best_model"


def test_model_name_for_variant_best(monkeypatch):
    monkeypatch.setenv("MLFLOW_BEST_MODEL_NAME", "best_model")
    assert model_service._model_name_for_variant("best") == "best_model"


def test_model_name_for_variant_latest(monkeypatch):
    monkeypatch.setenv("MLFLOW_LATEST_MODEL_NAME", "latest_model")
    assert model_service._model_name_for_variant("latest") == "latest_model"


def test_model_name_for_variant_missing_env_best(monkeypatch):
    monkeypatch.delenv("MLFLOW_BEST_MODEL_NAME", raising=False)
    with pytest.raises(RuntimeError):
        model_service._model_name_for_variant("champion")


def test_model_name_for_variant_missing_env_latest(monkeypatch):
    monkeypatch.delenv("MLFLOW_LATEST_MODEL_NAME", raising=False)
    with pytest.raises(RuntimeError):
        model_service._model_name_for_variant("latest")


def test_model_name_for_variant_invalid():
    with pytest.raises(RuntimeError):
        model_service._model_name_for_variant("unknown")


"""_latest_source_uri tests."""


def test_latest_source_uri(monkeypatch):
    fake_version = MagicMock()
    fake_version.version = "3"
    fake_version.source = "models:/MyModel/3"

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [fake_version]

    monkeypatch.setattr(
        model_service,
        "MlflowClient",
        lambda: fake_client,
    )

    uri = model_service._latest_source_uri("MyModel")
    assert uri == "models:/MyModel/3"


"""_is_models_alias_uri tests"""


def test_is_models_alias_uri_valid():
    assert model_service._is_models_alias_uri("models:/MyModel@dev")


def test_is_models_alias_uri_invalid_alias():
    assert not model_service._is_models_alias_uri("models:/MyModel/1")


def test_is_models_alias_uri_invalid_path():
    assert not model_service._is_models_alias_uri("http://example.com/model")


"""_parse_models_alias_uri tests"""


def test_parse_models_alias_uri_valid_prod_name():
    name, _ = model_service._parse_models_alias_uri("models:/Project_Model@dev")
    assert name == "Project_Model"


def test_parse_models_alias_uri_valid_prod_alias():
    _, alias = model_service._parse_models_alias_uri("models:/Project_Model@dev")
    assert alias == "dev"


def test_parse_models_alias_uri_valid_dev_name():
    name, _ = model_service._parse_models_alias_uri("models:/MyModel@prod")
    assert name == "MyModel"


def test_parse_models_alias_uri_valid_dev_alias():
    _, alias = model_service._parse_models_alias_uri("models:/MyModel@prod")
    assert alias == "prod"


def test_parse_models_alias_uri_with_whitespace_name():
    name, _ = model_service._parse_models_alias_uri("models:/ MyModel @ dev ")
    assert name == "MyModel"


def test_parse_models_alias_uri_with_whitespace_alas():
    _, alias = model_service._parse_models_alias_uri("models:/ MyModel @ dev ")
    assert alias == "dev"


def test_parse_models_alias_uri_invalid_empty_alias():
    with pytest.raises(ValueError):
        model_service._parse_models_alias_uri("models:/MyModel@")


def test_parse_models_alias_uri_invalid_empty_name():
    with pytest.raises(ValueError):
        model_service._parse_models_alias_uri("models:/@alias")


"""get_model tests."""


def test_get_model_with_direct_uri_correct_model(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.setenv("MODEL_URI_PROD", "models:/Champion/1")

    fake_model = MagicMock()

    monkeypatch.setattr(
        model_service,
        "_load_model_with_alias_fallback",
        lambda uri: (fake_model, uri),
    )

    model_service.clear_model_cache()
    model, _ = model_service.get_model("champion")

    assert model == fake_model


def test_get_model_with_direct_uri_check(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.setenv("MODEL_URI_PROD", "models:/Champion/1")

    fake_model = MagicMock()

    monkeypatch.setattr(
        model_service,
        "_load_model_with_alias_fallback",
        lambda uri: (fake_model, uri),
    )

    model_service.clear_model_cache()
    _, uri = model_service.get_model("champion")

    assert uri == "models:/Champion/1"


"""predict_one tests."""


def test_predict_one_success_pred(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.75]

    monkeypatch.setattr(
        model_service,
        "get_model",
        lambda variant: (fake_model, "model-uri"),
    )

    monkeypatch.setattr(
        model_service,
        "_expected_feature_count_from_model",
        lambda model: 3,
    )

    pred, _ = model_service.predict_one([1.0, 2.0, 3.0], "champion")
    assert pred == 0.75


def test_predict_one_success_uri(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.75]

    monkeypatch.setattr(
        model_service,
        "get_model",
        lambda variant: (fake_model, "model-uri"),
    )

    monkeypatch.setattr(
        model_service,
        "_expected_feature_count_from_model",
        lambda model: 3,
    )

    _, uri = model_service.predict_one([1.0, 2.0, 3.0], "champion")
    assert uri == "model-uri"


def test_predict_one_wrong_feature_count(monkeypatch):
    fake_model = MagicMock()

    monkeypatch.setattr(
        model_service,
        "get_model",
        lambda variant: (fake_model, "model-uri"),
    )

    monkeypatch.setattr(
        model_service,
        "_expected_feature_count_from_model",
        lambda model: 2,
    )

    with pytest.raises(ValueError, match="Model expects 2 features, got 3"):
        model_service.predict_one([1.0, 2.0, 3.0], "latest")


"""clear_model_cache tests."""


def test_clear_model_cache():
    model_service._cache["text"] = ("model", "uri")

    model_service.clear_model_cache()

    assert model_service._cache == {}


"""expected_feature_count tests."""


def test_expected_feature_count(monkeypatch):
    fake_sklearn = MagicMock()
    fake_sklearn.n_features_in_ = 5

    fake_impl = MagicMock()
    fake_impl.sklearn_model = fake_sklearn

    fake_model = MagicMock()
    fake_model._model_impl = fake_impl

    monkeypatch.setattr(
        model_service,
        "get_model",
        lambda variant: (fake_model, "uri"),
    )

    n = model_service.expected_feature_count("latest")

    assert n == 5
