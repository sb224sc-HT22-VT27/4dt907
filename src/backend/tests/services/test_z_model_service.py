import pytest
from unittest.mock import MagicMock

from app.services import z_model_service


def test_direct_uri_for_variant_uri_prod_champion(monkeypatch):
    monkeypatch.setenv("Z_MODEL_URI_PROD", " prod_uri ")
    assert z_model_service._direct_uri_for_variant("champion") == "prod_uri"


def test_direct_uri_for_variant_uri_dev_latest(monkeypatch):
    monkeypatch.setenv("Z_MODEL_URI_DEV", " dev_uri ")
    assert z_model_service._direct_uri_for_variant("latest") == "dev_uri"


def test_direct_uri_for_variant_uri_backup(monkeypatch):
    monkeypatch.setenv("Z_MODEL_URI_BACKUP", " backup_uri ")
    assert z_model_service._direct_uri_for_variant("backup") == "backup_uri"


def test_get_model_no_uri_configured(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    monkeypatch.delenv("Z_MODEL_URI_PROD", raising=False)
    z_model_service._cache.clear()
    with pytest.raises(RuntimeError, match="Z model URI is not set"):
        z_model_service.get_model("champion")


def test_predict_one_success(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.42]
    monkeypatch.setattr(
        z_model_service,
        "get_model",
        lambda variant="champion": (fake_model, "models:/ZModel/1", "run_1"),
    )
    monkeypatch.setattr(z_model_service, "_expected_feature_count_from_model", lambda _m: 2)

    pred, uri, run_id = z_model_service.predict_one([0.1, 0.2], "champion")
    assert pred == 0.42
    assert uri == "models:/ZModel/1"
    assert run_id == "run_1"
