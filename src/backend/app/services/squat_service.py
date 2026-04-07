"""app.services.squat_service

Squat analysis service.

Implements two phases of the squat-classification pipeline:

1. **Angle calculation** — uses the law of cosines to derive knee angles
   from hip, knee, and ankle 3-D (or 2-D) coordinates sent by the React
   frontend:

       θ = arccos((a² + b² − c²) / (2ab))

   where  a = hip→knee,  b = knee→ankle,  c = hip→ankle.

2. **Classification** — tries a small PyTorch model first; if the model
   file is absent or `torch` is not installed, falls back to a fast
   rule-based classifier so the endpoint is always responsive.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# * Current testing garbage model made by AI
# Ordered labels matching the PyTorch model's output classes.
LABELS = ["Deep", "Shallow", "Invalid"]
# Path where the trained model weights are expected to live.
_MODEL_PATH = Path(__file__).parent.parent / "models" / "squat_model.pt"


# Geometry helpers
def _dist3d(p1: Dict, p2: Dict) -> float:
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2 + (p1["z"] - p2["z"]) ** 2
    )


def _dist2d(p1: Dict, p2: Dict) -> float:
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def calculate_knee_angle(
    hip: Dict, knee: Dict, ankle: Dict, use_3d: bool = True
) -> float:
    """Calculate the knee angle using the law of cosines.

    Parameters
    ----------
    hip, knee, ankle:
        Dicts with at least ``x``, ``y`` keys (and ``z`` when *use_3d* is True).
    use_3d:
        If True use the z-component; fall back to 2-D otherwise.

    Returns
    -------
    float
        Angle in degrees (0–180).
    """
    dist = _dist3d if use_3d else _dist2d

    a = dist(hip, knee)  # hip → knee
    b = dist(knee, ankle)  # knee → ankle
    c = dist(hip, ankle)  # hip → ankle

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


def _pytorch_classify(left_angle: float, right_angle: float) -> Tuple[str, float]:
    """Classify using a pre-trained small PyTorch network.

    Falls back to rule-based if torch is unavailable or the model file is
    missing.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.debug("torch not installed — using rule-based classification")
        return _rule_based(left_angle, right_angle)

    if not _MODEL_PATH.exists():
        logger.debug("squat_model.pt not found — using rule-based classification")
        return _rule_based(left_angle, right_angle)

    class SquatNet(nn.Module):
        """Tiny 2→16→8→3 feed-forward network for squat classification."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 3),
            )

        def forward(self, x):
            return self.fc(x)

    try:
        model = SquatNet()
        checkpoint = torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)

        # Checkpoint may be a plain state_dict or a dict with state_dict +
        # normalisation constants saved by the notebook.
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            feat_mean = torch.tensor(
                checkpoint.get("mean", [0.0, 0.0]), dtype=torch.float32
            )
            feat_std = torch.tensor(
                checkpoint.get("std", [1.0, 1.0]), dtype=torch.float32
            )
        else:
            model.load_state_dict(checkpoint)
            # No normalisation stored — identity transform (raw angles).
            feat_mean = torch.zeros(2)
            feat_std = torch.ones(2)

        model.eval()

        with torch.no_grad():
            raw = torch.tensor([[left_angle, right_angle]], dtype=torch.float32)
            x = (raw - feat_mean) / (feat_std + 1e-8)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, idx].item())

        return LABELS[idx], confidence
    except Exception:
        logger.warning(
            "PyTorch inference failed — falling back to rule-based", exc_info=True
        )
        return _rule_based(left_angle, right_angle)


def classify_squat(
    keypoints_3d: list,
    keypoints_2d: list,
) -> Tuple[str, float, float, Optional[float]]:
    """Classify a squat from MediaPipe keypoint data.

    Parameters
    ----------
    keypoints_3d:
        List of dicts with keys ``name``, ``x``, ``y``, ``z``.
    keypoints_2d:
        List of dicts with keys ``name``, ``x``, ``y``.

    Returns
    -------
    tuple
        ``(classification, left_knee_angle, right_knee_angle, confidence)``
    """
    kp3 = {kp["name"]: kp for kp in keypoints_3d}
    kp2 = {kp["name"]: kp for kp in keypoints_2d}

    required = [
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hip",
        "right_knee",
        "right_ankle",
    ]

    # Prefer 3-D if all required keys are present.
    if all(k in kp3 for k in required):
        kp = kp3
        use_3d = True
    elif all(k in kp2 for k in required):
        kp = kp2
        use_3d = False
    else:
        return "Invalid", 0.0, 0.0, None

    left_angle = calculate_knee_angle(
        kp["left_hip"], kp["left_knee"], kp["left_ankle"], use_3d
    )
    right_angle = calculate_knee_angle(
        kp["right_hip"], kp["right_knee"], kp["right_ankle"], use_3d
    )

    classification, confidence = _pytorch_classify(left_angle, right_angle)
    return classification, left_angle, right_angle, confidence
