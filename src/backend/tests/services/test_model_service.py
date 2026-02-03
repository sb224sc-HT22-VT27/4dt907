import os
import pytest

from app.services.model_service import _model_name_for_variant
from app.services import model_service
from unittest.mock import MagicMock, patch

"""Test cases for model_service module."""

"""_clean_uri tests"""

def test_clean_uri_variants():
    assert model_service._clean_uri(None) is None
    assert model_service._clean_uri('  "abc" ') == 'abc'
    assert model_service._clean_uri(" 'def' ") == "def"

def test_direct_uri_for_variant(monkeypatch):
    monkeypatch.setenv("MODEL_URI_PROD", " prod_uri ")
    monkeypatch.setenv("MODEL_URI_DEV", " dev_uri ")
    monkeypatch.setenv("MODEL_URI_BACKUP", " backup_uri ")

    assert model_service._direct_uri_for_variant("champion") == "prod_uri"
    assert model_service._direct_uri_for_variant("best") == "prod_uri"
    assert model_service._direct_uri_for_variant("latest") == "dev_uri"
    assert model_service._direct_uri_for_variant("development") == "dev_uri"
    assert model_service._direct_uri_for_variant("backup") == "backup_uri"
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

    assert _model_name_for_variant("champion") == "best_model"
    assert _model_name_for_variant("best") == "best_model"

def test_model_name_for_variant_latest(monkeypatch):
    monkeypatch.setenv("MLFLOW_LATEST_MODEL_NAME", "latest_model")

    assert _model_name_for_variant("latest") == "latest_model"

def test_model_name_for_variant_missing_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_BEST_MODEL_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_LATEST_MODEL_NAME", raising=False)

    with pytest.raises(RuntimeError):
        _model_name_for_variant("champion")

    with pytest.raises(RuntimeError):
        _model_name_for_variant("latest")

def test_model_name_for_variant_invalid():
    with pytest.raises(RuntimeError):
        _model_name_for_variant("unknown")

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

"""get_model tests."""

"""predict_one tests."""

def test_predict_one_success(monkeypatch):
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

    pred, uri = model_service.predict_one([1.0, 2.0, 3.0], "champion")

    assert pred == 0.75
    assert uri == "model-uri"


"""clear_model_cache tests."""

def test_clear_model_cache():
    model_service._cache["text"] = ("model", "uri")

    model_service.clear_model_cache()

    assert model_service._cache == {}

"""expected_feature_count tests."""

def test_predict_one_wrong_feature_count(monkeypatch):
    fake_model = MagicMock()

    monkeypatch.setattr(
        "app.services.model_service.get_model",
        lambda variant: (fake_model, "model-uri"),
    )

    monkeypatch.setattr(
        "app.services.model_service.expected_feature_count",
        lambda variant: 2,
    )

    with pytest.raises(ValueError):
        model_service.predict_one([1.0, 2.0, 3.0], "latest")

def test_expected_feature_count(monkeypatch):
    fake_sklearn = MagicMock()
    fake_sklearn.n_features_in_ = 5

    fake_impl = MagicMock()
    fake_impl.sklearn_model = fake_sklearn

    fake_model = MagicMock()
    fake_model._model_impl = fake_impl

    monkeypatch.setattr(
        "app.services.model_service.get_model",
        lambda variant: (fake_model, "uri"),
    )

    n = model_service.expected_feature_count("latest")

    assert n == 5

