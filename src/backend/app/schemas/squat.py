"""app.schemas.squat

Pydantic schemas for the squat analysis endpoints.

These models define the request/response contract for the squat classification API.
The frontend sends 3D keypoints captured by MediaPipe; the backend returns
knee angles and a squat-depth classification.
"""

from typing import List, Optional

from pydantic import BaseModel


class Keypoint3D(BaseModel):
    """3D keypoint from MediaPipe pose detection."""

    name: str
    x: float
    y: float
    z: float
    score: Optional[float] = None


class SquatRequest(BaseModel):
    """Request payload for squat classification.

    The array must include at minimum:
    left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle.
    """

    keypoints_3d: List[Keypoint3D]


class SquatResponse(BaseModel):
    """Response returned from the squat classification endpoint."""

    classification: str  # "Deep", "Shallow", or "Invalid"
    left_knee_angle: float
    right_knee_angle: float
    confidence: Optional[float] = None


class SquatBatchRequest(BaseModel):
    frames: List[List[Keypoint3D]]


class SquatBatchResponse(BaseModel):
    results: List[SquatResponse]
