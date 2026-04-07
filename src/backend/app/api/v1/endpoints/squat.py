"""app.api.v1.endpoints.squat

Squat analysis endpoints (v1).

Accepts 2-D and 3-D joint coordinates captured by the MediaPipe pose
detector running in the React frontend, calculates knee angles, and returns
a squat-depth classification.
"""
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.squat import SquatRequest, SquatResponse
from app.services.squat_service import classify_squat

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/squat/classify", response_model=SquatResponse)
def squat_classify(req: SquatRequest):
    """Classify squat depth from MediaPipe keypoints.

    The request must contain at minimum the following named keypoints in
    both the 2-D and 3-D arrays:
    ``left_hip``, ``left_knee``, ``left_ankle``,
    ``right_hip``, ``right_knee``, ``right_ankle``.

    Returns one of **Deep**, **Shallow**, or **Invalid** together with the
    calculated knee angles.
    """
    try:
        kp_3d = [kp.model_dump() for kp in req.keypoints_3d]
        kp_2d = [kp.model_dump() for kp in req.keypoints_2d]

        classification, left_angle, right_angle, confidence = classify_squat(kp_3d, kp_2d)

        return SquatResponse(
            classification=classification,
            left_knee_angle=left_angle,
            right_knee_angle=right_angle,
            confidence=confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Squat classification failed")
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
