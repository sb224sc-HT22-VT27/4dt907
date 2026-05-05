"""app.api.v1.endpoints.squat

Squat analysis endpoints (v1).

Accepts 3-D joint coordinates captured by the MediaPipe pose detector running
in the React frontend, calculates knee angles, and returns a squat-depth
classification.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.squat import (
    FrameAnalysisResult,
    SessionAnalysisRequest,
    SessionAnalysisResponse,
    SquatBatchRequest,
    SquatBatchResponse,
    SquatRequest,
    SquatResponse,
)
from app.services.squat_service import classify_squat
from app.services import session_analysis_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/squat/classify", response_model=SquatResponse)
def squat_classify(req: SquatRequest):
    """Classify squat depth from MediaPipe 3-D keypoints.

    The request must contain at minimum the following named keypoints:
    ``left_hip``, ``left_knee``, ``left_ankle``,
    ``right_hip``, ``right_knee``, ``right_ankle``.

    Returns one of **Deep**, **Shallow**, or **Invalid** together with the
    calculated knee angles.
    """
    try:
        kp_3d = [kp.model_dump() for kp in req.keypoints_3d]

        classification, left_angle, right_angle, confidence = classify_squat(kp_3d)

        return SquatResponse(
            classification=classification,
            left_knee_angle=left_angle,
            right_knee_angle=right_angle,
            confidence=confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Squat classification failed")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.post("/squat/classify-batch", response_model=SquatBatchResponse)
def squat_classify_batch(req: SquatBatchRequest):
    """Classify squat depth for every frame in a recorded batch."""
    results = []
    for frame_kps in req.frames:
        try:
            kp_3d = [kp.model_dump() for kp in frame_kps]
            classification, left_angle, right_angle, confidence = classify_squat(kp_3d)
            results.append(
                SquatResponse(
                    classification=classification,
                    left_knee_angle=left_angle,
                    right_knee_angle=right_angle,
                    confidence=confidence,
                )
            )
        except Exception:
            results.append(
                SquatResponse(
                    classification="Invalid",
                    left_knee_angle=0.0,
                    right_knee_angle=0.0,
                    confidence=None,
                )
            )
    return SquatBatchResponse(results=results)


@router.post("/squat/analyze-session", response_model=SessionAnalysisResponse)
def squat_analyze_session(req: SessionAnalysisRequest):
    """Full pipeline: Cut (start/stop) → Z-pred → Classify → 3-D results.

    All frames from a recording are sent at once.  The backend runs the
    Start_Stop_Predictor_ModelV2 to cut the recording, smooths short gaps
    (< 10 frames), predicts z for every joint in every frame, and classifies
    squat depth for exercise frames.

    Non-exercise frames are returned with ``start_stop=0`` and
    ``classification="NotExercise"``.
    """
    try:
        frames = [[kp.model_dump() for kp in frame] for frame in req.frames]
        frame_results = session_analysis_service.analyze_session(frames)
        results = [
            FrameAnalysisResult(
                start_stop=fr.start_stop,
                classification=fr.classification,
                left_knee_angle=fr.left_knee_angle,
                right_knee_angle=fr.right_knee_angle,
                confidence=fr.confidence,
                predicted_z=fr.predicted_z,
            )
            for fr in frame_results
        ]
        return SessionAnalysisResponse(results=results)
    except Exception:
        logger.exception("Session analysis failed")
        raise HTTPException(status_code=503, detail="Service unavailable")
