"""Tests for normalize_kinect_data.py

Verifies that Kinect joint names are correctly remapped to
MediaPipe-compatible names and that coordinate values are hip-centered
to match MediaPipe world-landmark convention.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from the scripts directory regardless of where pytest runs.
sys.path.insert(0, str(Path(__file__).parent))

from normalize_kinect_data import (
    KINECT_TO_MEDIAPIPE,
    hip_center_normalize,
    normalize_file,
    remap_column,
)


# ---------------------------------------------------------------------------
# remap_column
# ---------------------------------------------------------------------------


class TestRemapColumn:
    """Unit tests for the column-rename helper."""

    def test_head_remapped_to_nose(self):
        assert remap_column("head_x") == "nose_x"
        assert remap_column("head_y") == "nose_y"
        assert remap_column("head_z") == "nose_z"

    def test_left_hand_remapped_to_left_wrist(self):
        assert remap_column("left_hand_x") == "left_wrist_x"
        assert remap_column("left_hand_y") == "left_wrist_y"
        assert remap_column("left_hand_z") == "left_wrist_z"

    def test_right_hand_remapped_to_right_wrist(self):
        assert remap_column("right_hand_x") == "right_wrist_x"
        assert remap_column("right_hand_y") == "right_wrist_y"
        assert remap_column("right_hand_z") == "right_wrist_z"

    def test_left_foot_remapped_to_left_ankle(self):
        assert remap_column("left_foot_x") == "left_ankle_x"
        assert remap_column("left_foot_y") == "left_ankle_y"
        assert remap_column("left_foot_z") == "left_ankle_z"

    def test_right_foot_remapped_to_right_ankle(self):
        assert remap_column("right_foot_x") == "right_ankle_x"
        assert remap_column("right_foot_y") == "right_ankle_y"
        assert remap_column("right_foot_z") == "right_ankle_z"

    def test_shared_joints_unchanged(self):
        """Joints whose names are identical in Kinect and MediaPipe must not change."""
        shared = [
            "left_shoulder_x",
            "left_shoulder_y",
            "left_shoulder_z",
            "right_shoulder_x",
            "left_elbow_x",
            "right_elbow_x",
            "left_hip_x",
            "right_hip_x",
            "left_knee_x",
            "right_knee_x",
        ]
        for col in shared:
            assert remap_column(col) == col, f"Expected {col!r} to be unchanged"

    def test_frameno_unchanged(self):
        assert remap_column("FrameNo") == "FrameNo"

    def test_leading_whitespace_stripped(self):
        """Column names in Kinect CSVs sometimes have a leading space."""
        assert remap_column(" head_x") == "nose_x"
        assert remap_column(" FrameNo") == "FrameNo"

    def test_no_partial_match(self):
        """'right_hand' prefix must not match unrelated column names."""
        assert remap_column("right_handgrip_x") == "right_handgrip_x"


# ---------------------------------------------------------------------------
# normalize_file
# ---------------------------------------------------------------------------


def _make_kinect_csv(tmp_path: Path) -> Path:
    """Write a minimal Kinect-format CSV to *tmp_path* and return its path."""
    src = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "FrameNo": [0, 1],
            "head_x": [0.1, 0.2],
            "head_y": [0.3, 0.4],
            "head_z": [0.5, 0.6],
            "left_shoulder_x": [0.7, 0.8],
            "left_shoulder_y": [0.9, 1.0],
            "left_shoulder_z": [1.1, 1.2],
            "left_hand_x": [1.3, 1.4],
            "left_hand_y": [1.5, 1.6],
            "left_hand_z": [1.7, 1.8],
            "right_hand_x": [1.9, 2.0],
            "right_hand_y": [2.1, 2.2],
            "right_hand_z": [2.3, 2.4],
            "left_hip_x": [-0.07, -0.06],
            "left_hip_y": [0.0, 0.01],
            "left_hip_z": [-0.03, -0.02],
            "right_hip_x": [0.07, 0.08],
            "right_hip_y": [-0.01, -0.01],
            "right_hip_z": [-0.04, -0.03],
            "left_foot_x": [2.5, 2.6],
            "left_foot_y": [2.7, 2.8],
            "left_foot_z": [2.9, 3.0],
            "right_foot_x": [3.1, 3.2],
            "right_foot_y": [3.3, 3.4],
            "right_foot_z": [3.5, 3.6],
        }
    )
    df.to_csv(src, index=False)
    return src


class TestNormalizeFile:
    """Integration tests for file-level normalization."""

    def test_output_file_created(self, tmp_path):
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "out" / "sample.csv"
        normalize_file(src, dst)
        assert dst.exists()

    def test_renamed_columns_present(self, tmp_path):
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "out" / "sample.csv"
        normalize_file(src, dst)

        result = pd.read_csv(dst)
        for mediapipe_name in KINECT_TO_MEDIAPIPE.values():
            for axis in ("x", "y", "z"):
                col = f"{mediapipe_name}_{axis}"
                assert col in result.columns, f"Expected column {col!r} in output"

    def test_original_kinect_columns_absent(self, tmp_path):
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "out" / "sample.csv"
        normalize_file(src, dst)

        result = pd.read_csv(dst)
        for kinect_name in KINECT_TO_MEDIAPIPE:
            for axis in ("x", "y", "z"):
                col = f"{kinect_name}_{axis}"
                assert col not in result.columns, f"Column {col!r} should be renamed"

    def test_data_values_hip_centered(self, tmp_path):
        """Normalized values must be hip-centered (hip midpoint = 0)."""
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "out" / "sample.csv"
        normalize_file(src, dst)

        result = pd.read_csv(dst)
        mid_x = (result["left_hip_x"] + result["right_hip_x"]) / 2.0
        mid_y = (result["left_hip_y"] + result["right_hip_y"]) / 2.0
        mid_z = (result["left_hip_z"] + result["right_hip_z"]) / 2.0
        assert (mid_x.abs() < 1e-9).all(), "Hip midpoint x must be 0 after normalization"
        assert (mid_y.abs() < 1e-9).all(), "Hip midpoint y must be 0 after normalization"
        assert (mid_z.abs() < 1e-9).all(), "Hip midpoint z must be 0 after normalization"

    def test_frameno_preserved(self, tmp_path):
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "out" / "sample.csv"
        normalize_file(src, dst)

        original = pd.read_csv(src)
        result = pd.read_csv(dst)
        assert len(result) == len(original)
        assert list(result["FrameNo"]) == list(original["FrameNo"])

    def test_parent_directory_created(self, tmp_path):
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "nested" / "deep" / "sample.csv"
        normalize_file(src, dst)
        assert dst.exists()


# ---------------------------------------------------------------------------
# hip_center_normalize
# ---------------------------------------------------------------------------


def _make_hip_df(
    left_hip_x, left_hip_y, left_hip_z,
    right_hip_x, right_hip_y, right_hip_z,
    head_x, head_y, head_z,
):
    """Create a single-row DataFrame with the given hip and head values."""
    return pd.DataFrame({
        "FrameNo": [0],
        "nose_x": [head_x],
        "nose_y": [head_y],
        "nose_z": [head_z],
        "left_hip_x": [left_hip_x],
        "left_hip_y": [left_hip_y],
        "left_hip_z": [left_hip_z],
        "right_hip_x": [right_hip_x],
        "right_hip_y": [right_hip_y],
        "right_hip_z": [right_hip_z],
    })


class TestHipCenterNormalize:
    """Unit tests for the hip-centering value normalization."""

    def test_hips_centered_at_origin(self):
        """After normalization the hip midpoint must be exactly (0, 0, 0)."""
        df = _make_hip_df(
            left_hip_x=-0.07, left_hip_y=-0.004, left_hip_z=-0.033,
            right_hip_x=0.07,  right_hip_y=-0.005, right_hip_z=-0.035,
            head_x=0.01, head_y=0.74, head_z=-0.06,
        )
        result = hip_center_normalize(df)
        mid_x = (result["left_hip_x"] + result["right_hip_x"]) / 2.0
        mid_y = (result["left_hip_y"] + result["right_hip_y"]) / 2.0
        mid_z = (result["left_hip_z"] + result["right_hip_z"]) / 2.0
        assert abs(float(mid_x.iloc[0])) < 1e-9
        assert abs(float(mid_y.iloc[0])) < 1e-9
        assert abs(float(mid_z.iloc[0])) < 1e-9

    def test_head_shifted_by_hip_midpoint(self):
        """Every joint must be shifted by the same per-frame hip midpoint."""
        lhx, rhx = -0.07, 0.05
        lhy, rhy = 0.0, -0.01
        lhz, rhz = -0.03, -0.04
        head_x, head_y, head_z = 0.5, 0.8, -0.06
        df = _make_hip_df(lhx, lhy, lhz, rhx, rhy, rhz, head_x, head_y, head_z)
        result = hip_center_normalize(df)
        mid_x = (lhx + rhx) / 2.0
        mid_y = (lhy + rhy) / 2.0
        mid_z = (lhz + rhz) / 2.0
        assert abs(float(result["nose_x"].iloc[0]) - (head_x - mid_x)) < 1e-9
        assert abs(float(result["nose_y"].iloc[0]) - (head_y - mid_y)) < 1e-9
        assert abs(float(result["nose_z"].iloc[0]) - (head_z - mid_z)) < 1e-9

    def test_already_centered_data_unchanged(self):
        """Data already centered at origin should be essentially unchanged."""
        df = _make_hip_df(
            left_hip_x=-0.07, left_hip_y=0.0, left_hip_z=-0.03,
            right_hip_x=0.07,  right_hip_y=0.0, right_hip_z=0.03,
            head_x=0.0, head_y=0.74, head_z=0.0,
        )
        result = hip_center_normalize(df)
        # Hip midpoint is exactly (0,0,0) so no shift happens
        assert abs(float(result["nose_x"].iloc[0]) - 0.0) < 1e-9
        assert abs(float(result["nose_y"].iloc[0]) - 0.74) < 1e-9

    def test_frameno_column_unchanged(self):
        """Non-coordinate column FrameNo must not be altered."""
        df = _make_hip_df(
            -0.07, 0.0, -0.03, 0.07, 0.0, 0.03, 0.0, 0.74, 0.0
        )
        result = hip_center_normalize(df)
        assert list(result["FrameNo"]) == [0]

    def test_original_dataframe_not_mutated(self):
        """hip_center_normalize must not mutate its input DataFrame."""
        df = _make_hip_df(-0.07, 0.0, -0.03, 0.07, 0.0, 0.03, 0.5, 0.74, 0.0)
        original_head_x = float(df["nose_x"].iloc[0])
        hip_center_normalize(df)
        assert float(df["nose_x"].iloc[0]) == original_head_x

    def test_multiple_frames_each_centered_independently(self):
        """Each frame uses its own hip midpoint, not a global average."""
        df = pd.DataFrame({
            "FrameNo": [0, 1],
            "left_hip_x": [-0.1, 0.5],
            "left_hip_y": [0.0, 0.0],
            "left_hip_z": [0.0, 0.0],
            "right_hip_x": [0.1, 0.7],
            "right_hip_y": [0.0, 0.0],
            "right_hip_z": [0.0, 0.0],
            "nose_x": [0.0, 0.6],
            "nose_y": [0.8, 0.8],
            "nose_z": [0.0, 0.0],
        })
        result = hip_center_normalize(df)
        # Frame 0: hip_mid_x = (-0.1+0.1)/2 = 0  → nose_x should stay 0.0
        # Frame 1: hip_mid_x = (0.5+0.7)/2 = 0.6 → nose_x should become 0.6-0.6 = 0.0
        assert abs(float(result["nose_x"].iloc[0]) - 0.0) < 1e-9
        assert abs(float(result["nose_x"].iloc[1]) - 0.0) < 1e-9
