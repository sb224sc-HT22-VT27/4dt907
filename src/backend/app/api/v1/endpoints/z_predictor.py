"""app.api.v1.endpoints.z_predictor

Z-predictor endpoints (v1).
"""
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.z_model_service import predict_one

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/z-predictor/champion", response_model=PredictResponse)
def z_predictor_champion(req: PredictRequest):
    try:
        pred, uri, run_id = predict_one(req.features, "champion")
        return PredictResponse(prediction=pred, model_uri=uri, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Z-predictor champion prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")


@router.post("/z-predictor/latest", response_model=PredictResponse)
def z_predictor_latest(req: PredictRequest):
    try:
        pred, uri, run_id = predict_one(req.features, "latest")
        return PredictResponse(prediction=pred, model_uri=uri, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Z-predictor latest prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
