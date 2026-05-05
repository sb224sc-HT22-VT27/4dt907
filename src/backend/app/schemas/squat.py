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


class SessionAnalysisRequest(BaseModel):
    """All frames from a recorded session sent at once for the full pipeline."""

    frames: List[List[Keypoint3D]]


class FrameAnalysisResult(BaseModel):
    """Per-frame result from the session analysis pipeline."""

    start_stop: int  # 0 = not exercise, 1 = in exercise (after smoothing)
    classification: str  # "Deep", "Shallow", "Invalid", or "NotExercise"
    left_knee_angle: float
    right_knee_angle: float
    confidence: Optional[float] = None
    predicted_z: dict  # {joint_name: z_value} for all 13 joints


class SessionAnalysisResponse(BaseModel):
    """Response from the session analysis pipeline."""

    results: List[FrameAnalysisResult]
