"""app.schemas.squat

Pydantic schemas for the squat analysis endpoints.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class Keypoint3D(BaseModel):
    """3D keypoint from MediaPipe pose detection."""

    name: str
    x: float
    y: float
    z: float
    score: Optional[float] = None


class SessionAnalysisRequest(BaseModel):
    """All frames from a recorded session sent at once for the full pipeline."""

    frames: List[List[Keypoint3D]]
    norm_frames: Optional[List[List[Keypoint3D]]] = None


class FrameAnalysisResult(BaseModel):
    """Per-frame result from the session analysis pipeline."""

    start_stop: int
    predicted_z: Dict[str, float]
    good_bad_score: Optional[float] = None


class SessionAnalysisResponse(BaseModel):
    """Response from the session analysis pipeline."""

    results: List[FrameAnalysisResult]
    timings: Optional[Dict[str, float]] = None
