from fastapi import APIRouter
from app.services.model_service import get_model, expected_feature_count

router = APIRouter()

@router.get("/model-info/latest")
def model_info_latest():
    _model, uri = get_model("latest")
    return {"variant": "latest", "model_uri": uri, "expected_features": expected_feature_count("latest")}

@router.get("/model-info/champion")
def model_info_champion():
    _model, uri = get_model("champion")
    return {"variant": "champion", "model_uri": uri, "expected_features": expected_feature_count("champion")}
