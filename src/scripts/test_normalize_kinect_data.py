"""Tests for normalize_kinect_data.py

Verifies that Kinect joint names are correctly remapped to
MediaPipe-compatible names and that file normalization works end-to-end.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from the scripts directory regardless of where pytest runs.
sys.path.insert(0, str(Path(__file__).parent))

from normalize_kinect_data import KINECT_TO_MEDIAPIPE, normalize_file, remap_column


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

    def test_data_values_preserved(self, tmp_path):
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "out" / "sample.csv"
        normalize_file(src, dst)

        original = pd.read_csv(src)
        result = pd.read_csv(dst)
        assert len(result) == len(original)
        assert list(result["FrameNo"]) == list(original["FrameNo"])
        # Value originally in head_x should appear in nose_x
        assert list(result["nose_x"]) == list(original["head_x"])
        assert list(result["left_wrist_x"]) == list(original["left_hand_x"])
        assert list(result["left_ankle_x"]) == list(original["left_foot_x"])

    def test_parent_directory_created(self, tmp_path):
        src = _make_kinect_csv(tmp_path)
        dst = tmp_path / "nested" / "deep" / "sample.csv"
        normalize_file(src, dst)
        assert dst.exists()
