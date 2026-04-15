"""app.api.v1.endpoints.predict

Prediction endpoints for the “champion” and “latest” model variants.

These routes are HTTP wrappers around the model service layer:
- validate/parse request schema
- call the service (`predict_one`)
- translate service exceptions into HTTP responses
"""

import logging
from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.model_service import predict_one

# Module-level logger + router used by the v1 API aggregator.
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict/champion", response_model=PredictResponse)
# Predict using champion registered model
def predict_champion(req: PredictRequest):
    try:
        pred, uri, run_id = predict_one(req.features, "champion")
        return PredictResponse(prediction=pred, model_uri=uri, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")


@router.post("/predict/latest", response_model=PredictResponse)
# Predict using champion latest model
def predict_latest(req: PredictRequest):
    try:
        pred, uri, run_id = predict_one(req.features, "latest")
        return PredictResponse(prediction=pred, model_uri=uri, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
