"""Tests for squat_service module."""

import math

import pytest

from app.services.squat_service import calculate_knee_angle, classify_squat


# calculate_knee_angle
def _pt3(x, y, z):
    return {"x": x, "y": y, "z": z}


def test_calculate_knee_angle_straight_leg_3d():
    # Hip directly above knee, knee directly above ankle → 180°
    hip = _pt3(0, 2, 0)
    knee = _pt3(0, 1, 0)
    ankle = _pt3(0, 0, 0)
    angle = calculate_knee_angle(hip, knee, ankle)
    assert math.isclose(angle, 180.0, abs_tol=1e-6)


def test_calculate_knee_angle_right_angle_3d():
    # Hip directly above knee, ankle to the side → knee angle = 90°
    hip = _pt3(0, 1, 0)
    knee = _pt3(0, 0, 0)
    ankle = _pt3(1, 0, 0)
    angle = calculate_knee_angle(hip, knee, ankle)
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_calculate_knee_angle_zero_length_returns_180():
    knee = _pt3(0, 0, 0)
    angle = calculate_knee_angle(knee, knee, _pt3(1, 0, 0))
    assert angle == 180.0


# classify_squat
def _make_kp3(name, x, y, z):
    return {"name": name, "x": x, "y": y, "z": z}


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


def _full_kp3(left_angle, right_angle):
    """Generate full left+right 3-D keypoints."""
    return _bent_leg_kp3("left", left_angle) + _bent_leg_kp3("right", right_angle)


@pytest.fixture(autouse=True)
def _mock_predict_z(monkeypatch):
    monkeypatch.setattr(
        "app.services.squat_service.predict_z",
        lambda _features, _variant="champion": (0.0, "models:/ZModel/1", "run_1"),
    )


def test_classify_squat_deep():
    kp3 = _full_kp3(80, 82)
    classification, left, right, confidence = classify_squat(kp3)
    assert classification == "Deep"
    assert 0 < left < 110
    assert 0 < right < 110


def test_classify_squat_shallow():
    kp3 = _full_kp3(120, 118)
    classification, left, right, confidence = classify_squat(kp3)
    assert classification == "Shallow"
    assert 100 <= left < 140
    assert 100 <= right < 140


def test_classify_squat_invalid_straight_legs():
    kp3 = _full_kp3(160, 162)
    classification, left, right, confidence = classify_squat(kp3)
    assert classification == "Invalid"


def test_classify_squat_invalid_asymmetric():
    kp3 = _full_kp3(80, 130)
    classification, left, right, confidence = classify_squat(kp3)
    assert classification == "Invalid"


def test_classify_squat_missing_keypoints_returns_invalid():
    # Send empty keypoint list
    classification, left, right, confidence = classify_squat([])
    assert classification == "Invalid"
    assert left == 0.0
    assert right == 0.0
    assert confidence is None


def test_classify_squat_returns_confidence():
    kp3 = _full_kp3(85, 87)
    _, _, _, confidence = classify_squat(kp3)
    assert confidence is not None
    assert 0.0 <= confidence <= 1.0


def test_classify_squat_uses_z_predictor(monkeypatch):
    calls = {"count": 0}

    def _fake_predict(features, variant):
        calls["count"] += 1
        return 0.0, "models:/ZModel/1", "run_1"

    monkeypatch.setattr("app.services.squat_service.predict_z", _fake_predict)
    kp3 = _full_kp3(85, 87)
    classify_squat(kp3)
    assert calls["count"] == 6
