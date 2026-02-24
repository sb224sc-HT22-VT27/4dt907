"""app.api.v1.endpoints.model_info

Model metadata endpoints.

These routes expose provenance information (model URI, run_id, and
expected feature count) for different model variants (latest/champion).
They are intentionally read-only and should avoid heavy model calls where possible.
"""
from fastapi import APIRouter
from app.services.model_service import get_model, expected_feature_count
from app.services import weaklink_model_service

# Router for model provenance/metadata endpoints (used by UI and debugging tools).
router = APIRouter()


@router.get("/model-info/latest")
def model_info_latest():
    """Return metadata for the most recently registered primary model."""
    _model, uri, run_id = get_model("latest")
    return {
        "variant": "latest",
        "model_uri": uri,
        "run_id": run_id,
        "expected_features": expected_feature_count("latest"),
    }


@router.get("/model-info/champion")
def model_info_champion():
    """Return metadata for the champion/best primary model."""
    _model, uri, run_id = get_model("champion")
    return {
        "variant": "champion",
        "model_uri": uri,
        "run_id": run_id,
        "expected_features": expected_feature_count("champion"),
    }


@router.get("/model-info/weakest-link/latest")
def model_info_weakest_link_latest():
    """Return metadata for the most recently registered weakest-link model."""
    _model, uri, run_id = weaklink_model_service.get_model("latest")
    return {
        "variant": "latest",
        "model_uri": uri,
        "run_id": run_id,
        "expected_features": weaklink_model_service.expected_feature_count("latest"),
    }


@router.get("/model-info/weakest-link/champion")
def model_info_weakest_link_champion():
    """Return metadata for the champion/best weakest-link model."""
    _model, uri, run_id = weaklink_model_service.get_model("champion")
    return {
        "variant": "champion",
        "model_uri": uri,
        "run_id": run_id,
        "expected_features": weaklink_model_service.expected_feature_count("champion"),
    }
