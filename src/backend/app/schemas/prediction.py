"""app.schemas.prediction

Pydantic schemas for prediction endpoints.

These models define the request/response contract for the primary model API.
Keep them stable to avoid breaking clients (frontend, scripts, etc.).
"""

from pydantic import BaseModel
from typing import List, Optional


class PredictRequest(BaseModel):
    """Request payload for a prediction call"""

    features: List[float]


class PredictResponse(BaseModel):
    """Response returned from a prediction call"""

    prediction: float
    model_uri: str
    run_id: Optional[str] = None


class ZSequenceRequest(BaseModel):
    """Request for sequence-based z prediction using the GRU/LSTM model.

    sequence: 2-D list of shape (n_frames, n_joints * 2).
    Typically (30, 26) for 13 squat joints over 30 frames.
    Rows are frames ordered oldest → newest.
    Columns per frame: [j0_x, j0_y, j1_x, j1_y, …] in canonical joint order
    (nose, left_shoulder, left_elbow, right_shoulder, right_elbow,
     left_wrist, right_wrist, left_hip, right_hip, left_knee,
     right_knee, left_ankle, right_ankle).
    Coordinates are MediaPipe world-landmark values (hip-centred, metres).
    """

    sequence: List[List[float]]


class ZSequencePredictResponse(BaseModel):
    """Response from sequence-based z prediction.

    predictions: z values in canonical joint order (one value per joint).
    """

    predictions: List[float]
    model_uri: str
    run_id: Optional[str] = None
