import numpy as np
from unittest.mock import MagicMock

from app.services import scoring_model_service
from app.services import goodbad_model_service


def test_direct_uri_for_variant_uri_prod_champion(monkeypatch):
    monkeypatch.setenv("SCORING_MODEL_URI_PROD", " prod_uri ")
    assert scoring_model_service._direct_uri_for_variant("champion") == "prod_uri"


def test_direct_uri_for_variant_uri_dev_latest(monkeypatch):
    monkeypatch.setenv("SCORING_MODEL_URI_DEV", " dev_uri ")
    assert scoring_model_service._direct_uri_for_variant("latest") == "dev_uri"


def test_direct_uri_for_variant_uri_backup(monkeypatch):
    monkeypatch.setenv("SCORING_MODEL_URI_BACKUP", " backup_uri ")
    assert scoring_model_service._direct_uri_for_variant("backup") == "backup_uri"


def test_predict_session_clamps_output_to_0_4(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [6.2]

    monkeypatch.setattr(
        scoring_model_service,
        "get_model",
        lambda variant="champion": (fake_model, "models:/ScoreModel/1", "run_1", 2, 61, None),
    )
    monkeypatch.setattr(
        goodbad_model_service,
        "_build_base_features",
        lambda _f: [0.0] * 39,
    )
    monkeypatch.setattr(
        goodbad_model_service,
        "_add_dist_angle_features",
        lambda arr: np.zeros((arr.shape[0], 61), dtype=np.float32),
    )
    monkeypatch.setattr(
        goodbad_model_service,
        "_resample_to_fixed",
        lambda arr, target: np.zeros((target, 61), dtype=np.float32),
    )

    score = scoring_model_service.predict_session([[{"name": "nose", "x": 0, "y": 0, "z": 0}]])
    assert score == 4.0
