"""Tests for squat_service module."""
import math

import pytest

from app.services.squat_service import calculate_knee_angle, classify_squat


# ---------------------------------------------------------------------------
# calculate_knee_angle
# ---------------------------------------------------------------------------


def _pt3(x, y, z):
    return {"x": x, "y": y, "z": z}


def _pt2(x, y):
    return {"x": x, "y": y}


def test_calculate_knee_angle_straight_leg_3d():
    # Hip directly above knee, knee directly above ankle → 180°
    hip = _pt3(0, 2, 0)
    knee = _pt3(0, 1, 0)
    ankle = _pt3(0, 0, 0)
    angle = calculate_knee_angle(hip, knee, ankle, use_3d=True)
    assert math.isclose(angle, 180.0, abs_tol=1e-6)


def test_calculate_knee_angle_right_angle_3d():
    # Hip directly above knee, ankle to the side → knee angle = 90°
    hip = _pt3(0, 1, 0)
    knee = _pt3(0, 0, 0)
    ankle = _pt3(1, 0, 0)
    angle = calculate_knee_angle(hip, knee, ankle, use_3d=True)
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_calculate_knee_angle_uses_2d_fallback():
    hip = _pt2(0, 1)
    knee = _pt2(0, 0)
    ankle = _pt2(1, 0)
    angle = calculate_knee_angle(hip, knee, ankle, use_3d=False)
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_calculate_knee_angle_zero_length_returns_180():
    knee = _pt3(0, 0, 0)
    angle = calculate_knee_angle(knee, knee, _pt3(1, 0, 0), use_3d=True)
    assert angle == 180.0


# ---------------------------------------------------------------------------
# classify_squat
# ---------------------------------------------------------------------------


def _make_kp3(name, x, y, z):
    return {"name": name, "x": x, "y": y, "z": z}


def _make_kp2(name, x, y):
    return {"name": name, "x": x, "y": y}


def _straight_leg_kp3(side):
    prefix = side
    return [
        _make_kp3(f"{prefix}_hip", 0, 2, 0),
        _make_kp3(f"{prefix}_knee", 0, 1, 0),
        _make_kp3(f"{prefix}_ankle", 0, 0, 0),
    ]


def _bent_leg_kp3(side, knee_angle_deg):
    """Build keypoints that produce exactly *knee_angle_deg* at the knee.

    Using the identity: arccos(-cos(r)) = 180 - r_deg, so to get the desired
    knee angle we set r = 180 - knee_angle_deg.
    """
    r = math.radians(180 - knee_angle_deg)
    hip = _make_kp3(f"{side}_hip", 0, 1, 0)
    knee = _make_kp3(f"{side}_knee", 0, 0, 0)
    ankle = _make_kp3(f"{side}_ankle", math.sin(r), -math.cos(r), 0)
    return [hip, knee, ankle]


def _full_kp(left_angle, right_angle):
    """Generate full left+right keypoints for both arrays."""
    kp3 = _bent_leg_kp3("left", left_angle) + _bent_leg_kp3("right", right_angle)
    kp2 = [_make_kp2(k["name"], k["x"], k["y"]) for k in kp3]
    return kp3, kp2


def test_classify_squat_deep():
    kp3, kp2 = _full_kp(80, 82)
    classification, left, right, confidence = classify_squat(kp3, kp2)
    assert classification == "Deep"
    assert 0 < left < 110
    assert 0 < right < 110


def test_classify_squat_shallow():
    kp3, kp2 = _full_kp(120, 118)
    classification, left, right, confidence = classify_squat(kp3, kp2)
    assert classification == "Shallow"
    assert 100 <= left < 140
    assert 100 <= right < 140


def test_classify_squat_invalid_straight_legs():
    kp3, kp2 = _full_kp(160, 162)
    classification, left, right, confidence = classify_squat(kp3, kp2)
    assert classification == "Invalid"


def test_classify_squat_invalid_asymmetric():
    kp3, kp2 = _full_kp(80, 130)
    classification, left, right, confidence = classify_squat(kp3, kp2)
    assert classification == "Invalid"


def test_classify_squat_missing_keypoints_returns_invalid():
    # Send empty keypoint lists
    classification, left, right, confidence = classify_squat([], [])
    assert classification == "Invalid"
    assert left == 0.0
    assert right == 0.0
    assert confidence is None


def test_classify_squat_returns_confidence():
    kp3, kp2 = _full_kp(85, 87)
    _, _, _, confidence = classify_squat(kp3, kp2)
    assert confidence is not None
    assert 0.0 <= confidence <= 1.0


def test_classify_squat_prefers_3d_over_2d():
    """3-D keypoints should be used when available; result should be consistent."""
    kp3, kp2 = _full_kp(90, 92)
    cls_3d, left_3d, _, _ = classify_squat(kp3, kp2)
    # Pass empty 3-D list → forces 2-D path
    cls_2d, left_2d, _, _ = classify_squat([], kp2)
    # Both should produce the same classification for symmetric angles
    assert cls_3d == cls_2d
