"""app.services.squat_service

Squat analysis service.

Pipeline:
1. Predict z from x/y using the MLflow-hosted z-predictor model.
2. Calculate knee angles from reconstructed 3-D keypoints.
3. Classify squat depth using rule-based thresholds.
"""

import math
from typing import Dict, Optional, Tuple

from app.services.z_model_service import predict_one as predict_z


# Geometry helpers
def _dist3d(p1: Dict, p2: Dict) -> float:
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2 + (p1["z"] - p2["z"]) ** 2
    )


def calculate_knee_angle(hip: Dict, knee: Dict, ankle: Dict) -> float:
    """Calculate the knee angle using the law of cosines.

    Parameters
    ----------
    hip, knee, ankle:
        Dicts with at least ``x``, ``y``, ``z`` keys.

    Returns
    -------
    float
        Angle in degrees (0–180).
    """
    a = _dist3d(hip, knee)   # hip → knee
    b = _dist3d(knee, ankle)  # knee → ankle
    c = _dist3d(hip, ankle)   # hip → ankle

    if a == 0 or b == 0:
        return 180.0

    cosine = (a**2 + b**2 - c**2) / (2 * a * b)
    cosine = max(-1.0, min(1.0, cosine))  # clamp to valid arccos domain
    return math.degrees(math.acos(cosine))


# Classification
def _rule_based(left_angle: float, right_angle: float) -> Tuple[str, float]:
    """Simple threshold-based fallback classifier."""
    avg = (left_angle + right_angle) / 2
    asymmetry = abs(left_angle - right_angle)

    if asymmetry > 25:
        return "Invalid", 0.70
    if avg < 100:
        return "Deep", 0.80
    if avg < 140:
        return "Shallow", 0.80
    return "Invalid", 0.70


def _with_predicted_z(kp: Dict) -> Dict:
    """Return keypoint with z predicted from x/y using the z-predictor model."""
    try:
        z_value, _uri, _run_id = predict_z([float(kp["x"]), float(kp["y"])], "champion")
        return {**kp, "z": float(z_value)}
    except Exception:
        # Preserve existing z if z-predictor is unavailable.
        return {**kp, "z": float(kp.get("z", 0.0))}


def classify_squat(
    keypoints_3d: list,
) -> Tuple[str, float, float, Optional[float]]:
    """Classify a squat from MediaPipe 3-D keypoint data.

    Parameters
    ----------
    keypoints_3d:
        List of dicts with keys ``name``, ``x``, ``y``, ``z``.

    Returns
    -------
    tuple
        ``(classification, left_knee_angle, right_knee_angle, confidence)``
    """
    kp3 = {kp["name"]: kp for kp in keypoints_3d}

    required = [
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hip",
        "right_knee",
        "right_ankle",
    ]

    if not all(k in kp3 for k in required):
        return "Invalid", 0.0, 0.0, None

    kp3["left_hip"] = _with_predicted_z(kp3["left_hip"])
    kp3["left_knee"] = _with_predicted_z(kp3["left_knee"])
    kp3["left_ankle"] = _with_predicted_z(kp3["left_ankle"])
    kp3["right_hip"] = _with_predicted_z(kp3["right_hip"])
    kp3["right_knee"] = _with_predicted_z(kp3["right_knee"])
    kp3["right_ankle"] = _with_predicted_z(kp3["right_ankle"])

    left_angle = calculate_knee_angle(kp3["left_hip"], kp3["left_knee"], kp3["left_ankle"])
    right_angle = calculate_knee_angle(
        kp3["right_hip"], kp3["right_knee"], kp3["right_ankle"]
    )

    classification, confidence = _rule_based(left_angle, right_angle)
    return classification, left_angle, right_angle, confidence
