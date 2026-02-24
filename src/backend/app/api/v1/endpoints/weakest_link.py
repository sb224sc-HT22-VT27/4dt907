"""app.api.v1.endpoints.weakest_link

Weakest-link prediction endpoints (v1).

These routes are HTTP wrappers around `weaklink_model_service.predict_one`:
- validate/parse request schema
- call the service (variant = "champion" | "latest")
- convert service exceptions into HTTP responses
"""
import logging
from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictRequest
from app.schemas.weakest_link import WeakestLinkResponse
from app.services.weaklink_model_service import predict_one

# Module-level logger + router used by the v1 API aggregator.
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/weakest-link/champion", response_model=WeakestLinkResponse)
def weakest_link_champion(req: PredictRequest):
    """Predict weakest-link outcome using the champion/best weakest-link model."""
    try:
        pred, uri, run_id = predict_one(req.features, "champion")
        return WeakestLinkResponse(prediction=pred, model_uri=uri, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Weakest-link prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")


@router.post("/weakest-link/latest", response_model=WeakestLinkResponse)
def weakest_link_latest(req: PredictRequest):
    """Predict weakest-link outcome using the most recently registered weakest-link model."""
    try:
        pred, uri, run_id = predict_one(req.features, "latest")
        return WeakestLinkResponse(prediction=pred, model_uri=uri, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Weakest-link prediction failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
