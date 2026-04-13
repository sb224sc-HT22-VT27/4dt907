"""app.services.squat_service

Squat analysis service.

Implements two phases of the squat-classification pipeline:

1. **Angle calculation** — uses the law of cosines to derive knee angles
   from hip, knee, and ankle 3-D coordinates sent by the React frontend:

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

# Module-level cache for the loaded PyTorch model and normalisation tensors.
# Populated on first successful load; None signals "not yet attempted".
_cached_model = None
_cached_feat_mean = None
_cached_feat_std = None
_model_load_attempted = False


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


def _pytorch_classify(left_angle: float, right_angle: float) -> Tuple[str, float]:
    """Classify using a pre-trained small PyTorch network.

    Falls back to rule-based if torch is unavailable or the model file is
    missing.  The model is loaded once and cached at the module level to avoid
    the per-request overhead of reading the checkpoint from disk.
    """
    global _cached_model, _cached_feat_mean, _cached_feat_std, _model_load_attempted

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.debug("torch not installed — using rule-based classification")
        return _rule_based(left_angle, right_angle)

    if not _MODEL_PATH.exists():
        logger.debug("squat_model.pt not found — using rule-based classification")
        return _rule_based(left_angle, right_angle)

    # Load and cache on first call; subsequent calls reuse the cached objects.
    if not _model_load_attempted:
        _model_load_attempted = True

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
            _cached_model = model
            _cached_feat_mean = feat_mean
            _cached_feat_std = feat_std
        except Exception:
            logger.warning(
                "PyTorch model load failed — falling back to rule-based", exc_info=True
            )

    if _cached_model is None:
        return _rule_based(left_angle, right_angle)

    try:
        import torch

        with torch.no_grad():
            raw = torch.tensor([[left_angle, right_angle]], dtype=torch.float32)
            x = (raw - _cached_feat_mean) / (_cached_feat_std + 1e-8)
            logits = _cached_model(x)
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

    left_angle = calculate_knee_angle(
        kp3["left_hip"], kp3["left_knee"], kp3["left_ankle"]
    )
    right_angle = calculate_knee_angle(
        kp3["right_hip"], kp3["right_knee"], kp3["right_ankle"]
    )

    classification, confidence = _pytorch_classify(left_angle, right_angle)
    return classification, left_angle, right_angle, confidence
