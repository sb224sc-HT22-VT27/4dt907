"""app.schemas.prediction

Pydantic schemas for prediction endpoints.

These models define the request/response contract for the primary model API.
Keep them stable to avoid breaking clients (frontend, scripts, etc.).
"""

from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    """Request payload for a prediction call"""

    features: List[float]


class PredictResponse(BaseModel):
    """Response returned from a prediction call"""

    prediction: float
    model_uri: str
    run_id: str | None = None
