"""app.services.session_analysis_service

Full session analysis pipeline: Cut → Z-pred → Classify → Results.

Pipeline per session (all frames at once):
1. Build per-frame feature vectors (39 floats: 13 joints × [x, y, z]).
2. Run Start_Stop_Predictor_ModelV2 on all frames → [0/1, ...].
3. Apply gap-fill smoothing: 0-runs < 10 frames between two 1-regions → 1.
4. For every frame: predict z for all 13 joints.
5. For exercise frames (start_stop=1): classify squat depth using predicted z.
6. Return per-frame results.
"""

from typing import Dict, List, Optional, Tuple

from app.services import start_stop_model_service
from app.services.squat_service import calculate_knee_angle, _rule_based
from app.services.z_model_service import predict_one as predict_z

# Joint names in Kinect SDK column order — must match training-time column order
# (sample.csv: nose, left_shoulder, right_shoulder, left_elbow, ...).
# The training notebook selects feat_cols WITHOUT sorting, so CSV order is preserved.
_MODEL_JOINT_NAMES: List[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _build_features(kp3d: List[Dict]) -> List[float]:
    """39-float feature vector [j0_x, j0_y, j0_z, j1_x, j1_y, j1_z, …] from kp3d list.

    Order matches the alphabetically-sorted _3d_x / _3d_y / _3d_z columns
    used during training (13 joints × 3 = 39).
    """
    kp_map = {kp["name"]: kp for kp in kp3d}
    feats: List[float] = []
    for name in _MODEL_JOINT_NAMES:
        kp = kp_map.get(name)
        feats.append(float(kp["x"]) if kp else 0.0)
        feats.append(float(kp["y"]) if kp else 0.0)
        feats.append(float(kp["z"]) if kp else 0.0)
    return feats


def _predict_z_all_joints(kp3d: List[Dict]) -> Dict[str, float]:
    """Predict z for all 13 joints in one frame via the z-predictor."""
    kp_map = {kp["name"]: kp for kp in kp3d}
    result: Dict[str, float] = {}
    for name in _MODEL_JOINT_NAMES:
        kp = kp_map.get(name)
        if kp:
            try:
                z, _, _ = predict_z([float(kp["x"]), float(kp["y"])], "champion")
                result[name] = float(z)
            except Exception:
                result[name] = float(kp.get("z", 0.0))
        else:
            result[name] = 0.0
    return result


def _smooth_start_stop(predictions: List[int], gap_threshold: int = 10) -> List[int]:
    """Fill 0-gaps < gap_threshold between two 1-regions with 1."""
    result = list(predictions)
    n = len(result)
    i = 0
    while i < n:
        if result[i] == 1:
            # Skip over consecutive 1s
            j = i + 1
            while j < n and result[j] == 1:
                j += 1
            # j is now the first 0 after a 1-run
            gap_start = j
            while j < n and result[j] == 0:
                j += 1
            # j is now the next 1 (or end of array)
            if j < n:
                gap_len = j - gap_start
                if gap_len < gap_threshold:
                    for k in range(gap_start, j):
                        result[k] = 1
                i = j
            else:
                break
        else:
            i += 1
    return result


def _classify_with_z(
    kp3d: List[Dict], predicted_z: Dict[str, float]
) -> Tuple[str, float, float, Optional[float]]:
    """Classify squat depth using pre-computed z values."""
    kp_map = {kp["name"]: kp for kp in kp3d}
    required = ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]
    if not all(k in kp_map for k in required):
        return "Invalid", 0.0, 0.0, None

    def _with_z(name: str) -> Dict:
        kp = kp_map[name]
        z = predicted_z.get(name, kp.get("z", 0.0))
        return {**kp, "z": float(z)}

    left_angle = calculate_knee_angle(
        _with_z("left_hip"), _with_z("left_knee"), _with_z("left_ankle")
    )
    right_angle = calculate_knee_angle(
        _with_z("right_hip"), _with_z("right_knee"), _with_z("right_ankle")
    )
    classification, confidence = _rule_based(left_angle, right_angle)
    return classification, left_angle, right_angle, confidence


class FrameResult:
    __slots__ = ("start_stop", "classification", "left_knee_angle",
                 "right_knee_angle", "confidence", "predicted_z")

    def __init__(
        self,
        start_stop: int,
        classification: str,
        left_knee_angle: float,
        right_knee_angle: float,
        confidence: Optional[float],
        predicted_z: Dict[str, float],
    ):
        self.start_stop = start_stop
        self.classification = classification
        self.left_knee_angle = left_knee_angle
        self.right_knee_angle = right_knee_angle
        self.confidence = confidence
        self.predicted_z = predicted_z


def analyze_session(frames: List[List[Dict]]) -> List[FrameResult]:
    """Run the full pipeline on all frames.

    Parameters
    ----------
    frames:
        List of N frames; each frame is a list of kp3d dicts
        ``{"name": str, "x": float, "y": float, "z": float}``.

    Returns
    -------
    List of FrameResult, one per input frame.
    """
    if not frames:
        return []

    # 1. Build feature vectors for start/stop model
    features_batch = [_build_features(f) for f in frames]

    # 2. Predict start/stop (fallback: all exercise)
    try:
        raw_start_stop = start_stop_model_service.predict_batch(features_batch, "champion")
    except Exception:
        raw_start_stop = [1] * len(frames)

    # 3. Smooth gaps
    smoothed = _smooth_start_stop(raw_start_stop)

    # 4 & 5. Z-pred + classify per frame
    results: List[FrameResult] = []
    for kp3d, ss in zip(frames, smoothed):
        predicted_z = _predict_z_all_joints(kp3d)

        if ss == 1:
            cls, la, ra, conf = _classify_with_z(kp3d, predicted_z)
        else:
            cls, la, ra, conf = "NotExercise", 0.0, 0.0, None

        results.append(FrameResult(
            start_stop=ss,
            classification=cls,
            left_knee_angle=la,
            right_knee_angle=ra,
            confidence=conf,
            predicted_z=predicted_z,
        ))

    return results
