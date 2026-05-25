import numpy as np

from app.services import scoring_model_service


def test_direct_uri_for_variant_uri_prod_champion(monkeypatch):
    monkeypatch.setenv("SQUAT_SCORING_MODEL_URI_PROD", " prod_uri ")
    assert scoring_model_service._direct_uri_for_variant("champion") == "prod_uri"


def test_direct_uri_for_variant_uri_dev_latest(monkeypatch):
    monkeypatch.setenv("SQUAT_SCORING_MODEL_URI_DEV", " dev_uri ")
    assert scoring_model_service._direct_uri_for_variant("latest") == "dev_uri"


def test_direct_uri_for_variant_uri_backup(monkeypatch):
    monkeypatch.setenv("SQUAT_SCORING_MODEL_URI_BACKUP", " backup_uri ")
    assert scoring_model_service._direct_uri_for_variant("backup") == "backup_uri"


def test_normalize_score_rounds_and_clamps():
    assert scoring_model_service._normalize_score(np.array([[4.6]], dtype=np.float32)) == 4
    assert scoring_model_service._normalize_score(np.array([[-2.0]], dtype=np.float32)) == 0


def test_normalize_score_argmax_for_multiclass():
    raw = np.array([[0.1, 0.4, 0.2, 0.15, 0.15]], dtype=np.float32)
    assert scoring_model_service._normalize_score(raw) == 1
