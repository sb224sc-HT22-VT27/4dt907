from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: float
    model_uri: str
    run_id: str | None = None