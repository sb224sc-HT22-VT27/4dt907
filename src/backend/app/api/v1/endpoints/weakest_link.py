import logging
from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictRequest
from app.schemas.weakest_link import WeakestLinkResponse
from app.services.weaklink_model_service import predict_one

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/weakest-link/champion", response_model=WeakestLinkResponse)
def weakest_link_champion(req: PredictRequest):
    try:
        pred, uri = predict_one(req.features, "champion")
        return WeakestLinkResponse(prediction=pred, model_uri=uri)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Weakest-link prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")

@router.post("/weakest-link/latest", response_model=WeakestLinkResponse)
def weakest_link_latest(req: PredictRequest):
    try:
        pred, uri = predict_one(req.features, "latest")
        return WeakestLinkResponse(prediction=pred, model_uri=uri)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Weakest-link prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
