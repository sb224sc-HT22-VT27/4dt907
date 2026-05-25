"""app.api.v1.endpoints.squat

Squat analysis endpoint (v1).

Accepts 3-D joint coordinates from the React frontend, runs the full
Start/Stop → Z-pred → GoodBad pipeline, and returns per-frame results.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.squat import (
    FrameAnalysisResult,
    SessionAnalysisRequest,
    SessionAnalysisResponse,
)
from app.services import session_analysis_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/squat/analyze-session", response_model=SessionAnalysisResponse)
def squat_analyze_session(req: SessionAnalysisRequest):
    """Full pipeline: Cut (start/stop) → Z-pred → GoodBad → Scoring → Results.

    All frames are sent at once. The backend runs Start_Stop_Predictor_ModelV2,
    smooths short gaps (< 10 frames), predicts z for every joint, and scores
    form quality for each continuous exercise segment.

    Non-exercise frames are returned with ``start_stop=0`` and
    ``good_bad_score=None``.
    """
    try:
        frames = [[kp.model_dump() for kp in frame] for frame in req.frames]
        norm_frames = (
            [[kp.model_dump() for kp in frame] for frame in req.norm_frames]
            if req.norm_frames
            else None
        )
        frame_results, timings = session_analysis_service.analyze_session(
            frames, norm_frames=norm_frames
        )
        results = [
            FrameAnalysisResult(
                start_stop=fr.start_stop,
                predicted_z=fr.predicted_z,
                good_bad_score=fr.good_bad_score,
                squat_score=fr.squat_score,
            )
            for fr in frame_results
        ]
        return SessionAnalysisResponse(results=results, timings=timings)
    except Exception:
        logger.exception("Session analysis failed")
        raise HTTPException(status_code=503, detail="Service unavailable")
