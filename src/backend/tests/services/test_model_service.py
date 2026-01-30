import os
import pytest

from app.services.model_service import _model_name_for_variant
from app.services import model_service

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