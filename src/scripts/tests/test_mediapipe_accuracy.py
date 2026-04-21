"""Tests for mediapipe_accuracy.py helper functions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the scripts package importable when running pytest from this directory
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from mediapipe_accuracy import (  # noqa: E402
    GroundTruthPair,
    PairStats,
    compute_pair_stats,
    euclidean_3d_cm,
    load_ground_truth_json,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(**kwargs) -> pd.Series:
    """Build a pandas Series representing a single CSV row."""
    return pd.Series(kwargs)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# euclidean_3d_cm
# ---------------------------------------------------------------------------


class TestEuclidean3dCm:
    def test_known_distance_along_y_axis(self):
        # kp1 at origin, kp2 at (0, 0.42, 0) → 42 cm
        row = _make_row(
            a_3d_x=0.0, a_3d_y=0.0, a_3d_z=0.0,
            b_3d_x=0.0, b_3d_y=0.42, b_3d_z=0.0,
        )
        result = euclidean_3d_cm(row, "a", "b")
        assert result is not None
        assert math.isclose(result, 42.0, abs_tol=1e-9)

    def test_3d_distance(self):
        # (3, 4, 0) metres → 5 metres = 500 cm
        row = _make_row(
            p_3d_x=0.0, p_3d_y=0.0, p_3d_z=0.0,
            q_3d_x=3.0, q_3d_y=4.0, q_3d_z=0.0,
        )
        result = euclidean_3d_cm(row, "p", "q")
        assert result is not None
        assert math.isclose(result, 500.0, abs_tol=1e-9)

    def test_returns_none_when_column_missing(self):
        row = _make_row(
            a_3d_x=0.0, a_3d_y=0.0,  # a_3d_z missing
            b_3d_x=0.1, b_3d_y=0.0, b_3d_z=0.0,
        )
        assert euclidean_3d_cm(row, "a", "b") is None

    def test_returns_none_when_value_is_nan(self):
        row = _make_row(
            a_3d_x=float("nan"), a_3d_y=0.0, a_3d_z=0.0,
            b_3d_x=0.1, b_3d_y=0.0, b_3d_z=0.0,
        )
        assert euclidean_3d_cm(row, "a", "b") is None

    def test_same_point_returns_zero(self):
        row = _make_row(
            a_3d_x=0.1, a_3d_y=0.2, a_3d_z=0.3,
            b_3d_x=0.1, b_3d_y=0.2, b_3d_z=0.3,
        )
        result = euclidean_3d_cm(row, "a", "b")
        assert result is not None
        assert math.isclose(result, 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# compute_pair_stats
# ---------------------------------------------------------------------------


class TestComputePairStats:
    def _hip_knee_row(self, offset: float = 0.0) -> dict:
        """Row where left_hip is at origin and left_knee is 0.42 m below."""
        return {
            "left_hip_3d_x": 0.0,
            "left_hip_3d_y": 0.0,
            "left_hip_3d_z": 0.0,
            "left_knee_3d_x": 0.0,
            "left_knee_3d_y": -(0.42 + offset),
            "left_knee_3d_z": 0.0,
        }

    def test_zero_error_when_estimate_matches_ground_truth(self):
        df = _make_df([self._hip_knee_row(), self._hip_knee_row()])
        pair = GroundTruthPair("left_hip", "left_knee", 42.0)
        stats = compute_pair_stats(df, pair)
        assert stats is not None
        assert stats.n_frames == 2
        assert math.isclose(stats.mean_error_cm, 0.0, abs_tol=1e-6)
        assert math.isclose(stats.std_error_cm, 0.0, abs_tol=1e-6)

    def test_positive_error_when_mediapipe_overestimates(self):
        # MediaPipe says 43 cm, ground truth is 42 cm → error = +1 cm
        df = _make_df([self._hip_knee_row(offset=0.01)])
        pair = GroundTruthPair("left_hip", "left_knee", 42.0)
        stats = compute_pair_stats(df, pair)
        assert stats is not None
        assert math.isclose(stats.mean_error_cm, 1.0, abs_tol=1e-6)

    def test_negative_error_when_mediapipe_underestimates(self):
        # MediaPipe says 41 cm, ground truth is 42 cm → error = -1 cm
        df = _make_df([self._hip_knee_row(offset=-0.01)])
        pair = GroundTruthPair("left_hip", "left_knee", 42.0)
        stats = compute_pair_stats(df, pair)
        assert stats is not None
        assert math.isclose(stats.mean_error_cm, -1.0, abs_tol=1e-6)

    def test_returns_none_when_columns_missing(self):
        df = _make_df([{"timestamp_ms": 1000, "classification": "Deep"}])
        pair = GroundTruthPair("left_hip", "left_knee", 42.0)
        assert compute_pair_stats(df, pair) is None

    def test_mean_estimated_cm_is_correct(self):
        rows = [self._hip_knee_row(offset=0.0), self._hip_knee_row(offset=0.02)]
        df = _make_df(rows)
        pair = GroundTruthPair("left_hip", "left_knee", 42.0)
        stats = compute_pair_stats(df, pair)
        assert stats is not None
        assert math.isclose(stats.mean_estimated_cm, 43.0, abs_tol=1e-6)

    def test_min_max_error(self):
        rows = [
            self._hip_knee_row(offset=0.01),   # 43 cm → +1 cm error
            self._hip_knee_row(offset=-0.01),  # 41 cm → -1 cm error
        ]
        df = _make_df(rows)
        pair = GroundTruthPair("left_hip", "left_knee", 42.0)
        stats = compute_pair_stats(df, pair)
        assert stats is not None
        assert math.isclose(stats.min_error_cm, -1.0, abs_tol=1e-6)
        assert math.isclose(stats.max_error_cm, 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# load_ground_truth_json
# ---------------------------------------------------------------------------


class TestLoadGroundTruthJson:
    def test_loads_valid_json(self, tmp_path: Path):
        data = [
            {"kp1": "left_hip", "kp2": "left_knee", "distance_cm": 42.0},
            {"kp1": "left_knee", "kp2": "left_ankle", "distance_cm": 38.5},
        ]
        json_file = tmp_path / "gt.json"
        json_file.write_text(json.dumps(data))
        pairs = load_ground_truth_json(json_file)
        assert len(pairs) == 2
        assert pairs[0] == GroundTruthPair("left_hip", "left_knee", 42.0)
        assert pairs[1] == GroundTruthPair("left_knee", "left_ankle", 38.5)

    def test_raises_on_missing_key(self, tmp_path: Path):
        data = [{"kp1": "left_hip", "distance_cm": 42.0}]  # kp2 missing
        json_file = tmp_path / "bad.json"
        json_file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Invalid entry"):
            load_ground_truth_json(json_file)

    def test_raises_on_non_numeric_distance(self, tmp_path: Path):
        data = [{"kp1": "left_hip", "kp2": "left_knee", "distance_cm": "not_a_number"}]
        json_file = tmp_path / "bad.json"
        json_file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Invalid entry"):
            load_ground_truth_json(json_file)


# ---------------------------------------------------------------------------
# main (CLI integration tests)
# ---------------------------------------------------------------------------


class TestMain:
    def _write_csv(self, tmp_path: Path) -> Path:
        """Write a minimal valid keypoints CSV and return its path."""
        rows = [
            {
                "timestamp_ms": 1000,
                "classification": "Deep",
                "confidence": 0.95,
                "left_hip_3d_x": 0.0,
                "left_hip_3d_y": 0.0,
                "left_hip_3d_z": 0.0,
                "left_knee_3d_x": 0.0,
                "left_knee_3d_y": -0.42,
                "left_knee_3d_z": 0.0,
            }
        ]
        csv_path = tmp_path / "session.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_returns_zero_on_success(self, tmp_path: Path):
        csv_path = self._write_csv(tmp_path)
        rc = main([str(csv_path), "--pair", "left_hip", "left_knee", "42.0"])
        assert rc == 0

    def test_returns_nonzero_on_missing_csv(self, tmp_path: Path):
        rc = main([str(tmp_path / "nonexistent.csv"), "--pair", "a", "b", "10.0"])
        assert rc != 0

    def test_returns_nonzero_on_invalid_pair_name(self, tmp_path: Path):
        csv_path = self._write_csv(tmp_path)
        # "nonexistent_joint" columns not in CSV → warning + partial result
        rc = main([str(csv_path), "--pair", "nonexistent_joint", "left_knee", "42.0"])
        assert rc != 0

    def test_json_input_returns_zero_on_success(self, tmp_path: Path):
        csv_path = self._write_csv(tmp_path)
        gt_data = [{"kp1": "left_hip", "kp2": "left_knee", "distance_cm": 42.0}]
        json_file = tmp_path / "gt.json"
        json_file.write_text(json.dumps(gt_data))
        rc = main([str(csv_path), "--ground-truth-json", str(json_file)])
        assert rc == 0

    def test_returns_nonzero_for_negative_distance(self, tmp_path: Path):
        csv_path = self._write_csv(tmp_path)
        rc = main([str(csv_path), "--pair", "left_hip", "left_knee", "-5.0"])
        assert rc != 0
