#!/usr/bin/env python3
"""mediapipe_accuracy.py

Compare MediaPipe keypoint distances against ground-truth real-world distances.

Given a keypoints CSV exported from the SquatAnalyzer (webcam / video / image)
and one or more ground-truth segment lengths measured in the real world, this
script reports how far off the MediaPipe-estimated distances are on a per-pair
basis.

The CSV must follow the format produced by the SquatAnalyzer's "Download CSV"
button, whose columns are::

    timestamp_ms, classification, confidence,
    <joint>_3d_x, <joint>_3d_y, <joint>_3d_z, ...

Coordinates are MediaPipe *world landmarks* expressed in metres with the hip
midpoint as origin.  This script converts them to centimetres before comparing
with the ground truth.

Usage examples
--------------
Compare the hip-to-knee and knee-to-ankle segment lengths::

    python mediapipe_accuracy.py session.csv \\
        --pair left_hip left_knee 42.0 \\
        --pair left_knee left_ankle 38.5

The --pair flag can be repeated as many times as needed.

You can also provide ground-truth distances via a JSON file::

    python mediapipe_accuracy.py session.csv \\
        --ground-truth-json ground_truth.json

where ``ground_truth.json`` is a list of objects::

    [
      {"kp1": "left_hip",  "kp2": "left_knee",  "distance_cm": 42.0},
      {"kp1": "left_knee", "kp2": "left_ankle", "distance_cm": 38.5}
    ]

Output
------
The script prints a table of per-pair statistics (in centimetres):

    Pair                           GT(cm)  Mean Est(cm)  Mean Err(cm)  Std Err(cm)  Min Err(cm)  Max Err(cm)
    left_hip → left_knee            42.00        41.23         -0.77         0.42        -1.50         0.12
    left_knee → left_ankle          38.50        37.90         -0.60         0.35        -1.20         0.05

A negative error means MediaPipe underestimates the distance; positive means
it overestimates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GroundTruthPair(NamedTuple):
    kp1: str
    kp2: str
    distance_cm: float


class PairStats(NamedTuple):
    kp1: str
    kp2: str
    ground_truth_cm: float
    mean_estimated_cm: float
    mean_error_cm: float
    std_error_cm: float
    min_error_cm: float
    max_error_cm: float
    n_frames: int


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def euclidean_3d_cm(row: "pd.Series", kp1: str, kp2: str) -> float | None:
    """Return the 3-D Euclidean distance between *kp1* and *kp2* in centimetres.

    Expects the row to have columns ``{kp}_3d_x``, ``{kp}_3d_y``,
    ``{kp}_3d_z`` for each keypoint.  Returns ``None`` if any required column
    is absent or contains NaN.

    Parameters
    ----------
    row:
        A single row from the keypoints DataFrame.
    kp1:
        Name of the first keypoint (e.g. ``"left_hip"``).
    kp2:
        Name of the second keypoint (e.g. ``"left_knee"``).

    Returns
    -------
    float | None
        Distance in centimetres, or ``None`` if data is unavailable.
    """
    cols_needed = [f"{kp}_{ax}" for kp in (kp1, kp2) for ax in ("3d_x", "3d_y", "3d_z")]
    if any(c not in row.index for c in cols_needed):
        return None
    values = [row[c] for c in cols_needed]
    if any(pd.isna(v) for v in values):
        return None
    x1, y1, z1 = values[0], values[1], values[2]
    x2, y2, z2 = values[3], values[4], values[5]
    dist_m = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist_m * 100.0  # metres → centimetres


def compute_pair_stats(
    df: "pd.DataFrame",
    pair: GroundTruthPair,
) -> PairStats | None:
    """Compute error statistics for a single keypoint pair against ground truth.

    Parameters
    ----------
    df:
        DataFrame loaded from the SquatAnalyzer CSV export.
    pair:
        Ground-truth pair with keypoint names and real-world distance.

    Returns
    -------
    PairStats | None
        Populated statistics object, or ``None`` if no valid frames were found.
    """
    distances = []
    for _, row in df.iterrows():
        d = euclidean_3d_cm(row, pair.kp1, pair.kp2)
        if d is not None:
            distances.append(d)

    if not distances:
        return None

    errors = [d - pair.distance_cm for d in distances]
    mean_est = sum(distances) / len(distances)
    mean_err = sum(errors) / len(errors)
    variance = sum((e - mean_err) ** 2 for e in errors) / len(errors)
    std_err = math.sqrt(variance)
    return PairStats(
        kp1=pair.kp1,
        kp2=pair.kp2,
        ground_truth_cm=pair.distance_cm,
        mean_estimated_cm=mean_est,
        mean_error_cm=mean_err,
        std_error_cm=std_err,
        min_error_cm=min(errors),
        max_error_cm=max(errors),
        n_frames=len(distances),
    )


def load_ground_truth_json(path: Path) -> list[GroundTruthPair]:
    """Load ground-truth pairs from a JSON file.

    The JSON file must be a list of objects with keys ``kp1``, ``kp2``, and
    ``distance_cm``::

        [
          {"kp1": "left_hip", "kp2": "left_knee", "distance_cm": 42.0}
        ]

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    list[GroundTruthPair]
        Parsed list of ground-truth pairs.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = []
    for i, entry in enumerate(data):
        try:
            pairs.append(
                GroundTruthPair(
                    kp1=entry["kp1"],
                    kp2=entry["kp2"],
                    distance_cm=float(entry["distance_cm"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid entry at index {i} in {path}: {exc}"
            ) from exc
    return pairs


def print_results(stats_list: list[PairStats]) -> None:
    """Print a formatted summary table of accuracy results.

    Parameters
    ----------
    stats_list:
        List of :class:`PairStats` objects to display.
    """
    col_widths = {
        "pair": 30,
        "gt": 10,
        "mean_est": 14,
        "mean_err": 14,
        "std_err": 13,
        "min_err": 12,
        "max_err": 12,
        "frames": 8,
    }

    header = (
        f"{'Pair':<{col_widths['pair']}}"
        f"{'GT (cm)':>{col_widths['gt']}}"
        f"{'Mean Est (cm)':>{col_widths['mean_est']}}"
        f"{'Mean Err (cm)':>{col_widths['mean_err']}}"
        f"{'Std Err (cm)':>{col_widths['std_err']}}"
        f"{'Min Err (cm)':>{col_widths['min_err']}}"
        f"{'Max Err (cm)':>{col_widths['max_err']}}"
        f"{'Frames':>{col_widths['frames']}}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for s in stats_list:
        pair_label = f"{s.kp1} → {s.kp2}"
        print(
            f"{pair_label:<{col_widths['pair']}}"
            f"{s.ground_truth_cm:>{col_widths['gt']}.2f}"
            f"{s.mean_estimated_cm:>{col_widths['mean_est']}.2f}"
            f"{s.mean_error_cm:>{col_widths['mean_err']}.2f}"
            f"{s.std_error_cm:>{col_widths['std_err']}.2f}"
            f"{s.min_error_cm:>{col_widths['min_err']}.2f}"
            f"{s.max_error_cm:>{col_widths['max_err']}.2f}"
            f"{s.n_frames:>{col_widths['frames']}}"
        )

    print(separator)
    print(
        "\nNote: error = MediaPipe estimate − ground truth. "
        "Negative → underestimate; positive → overestimate."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediapipe_accuracy.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv",
        type=Path,
        metavar="CSV_FILE",
        help="Path to the keypoints CSV exported from the SquatAnalyzer.",
    )

    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument(
        "--pair",
        dest="pairs",
        action="append",
        nargs=3,
        metavar=("KP1", "KP2", "DISTANCE_CM"),
        help=(
            "A keypoint pair and its real-world distance in centimetres. "
            "May be repeated for multiple pairs. "
            "Example: --pair left_hip left_knee 42.0"
        ),
    )
    gt_group.add_argument(
        "--ground-truth-json",
        type=Path,
        metavar="JSON_FILE",
        help=(
            "JSON file containing ground-truth distances. "
            "Format: [{\"kp1\": \"...\", \"kp2\": \"...\", \"distance_cm\": 42.0}, ...]"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the mediapipe_accuracy script.

    Parameters
    ----------
    argv:
        Optional list of CLI arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load CSV
    csv_path: Path = args.csv
    if not csv_path.is_file():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"ERROR: Could not read CSV file: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("ERROR: CSV file contains no data rows.", file=sys.stderr)
        return 1

    # Build ground-truth pairs
    ground_truth: list[GroundTruthPair]
    if args.ground_truth_json is not None:
        try:
            ground_truth = load_ground_truth_json(args.ground_truth_json)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        ground_truth = []
        for kp1, kp2, dist_str in args.pairs:
            try:
                dist_cm = float(dist_str)
            except ValueError:
                print(
                    f"ERROR: distance must be a number, got {dist_str!r}",
                    file=sys.stderr,
                )
                return 1
            if dist_cm <= 0:
                print(
                    f"ERROR: distance must be positive, got {dist_cm}",
                    file=sys.stderr,
                )
                return 1
            ground_truth.append(GroundTruthPair(kp1=kp1, kp2=kp2, distance_cm=dist_cm))

    if not ground_truth:
        print("ERROR: No ground-truth pairs provided.", file=sys.stderr)
        return 1

    # Compute statistics for each pair
    results: list[PairStats] = []
    has_error = False
    for pair in ground_truth:
        stats = compute_pair_stats(df, pair)
        if stats is None:
            print(
                f"WARNING: No valid frames found for pair "
                f"{pair.kp1!r} → {pair.kp2!r}. "
                f"Check that these keypoint names exist in the CSV.",
                file=sys.stderr,
            )
            has_error = True
        else:
            results.append(stats)

    if not results:
        print("ERROR: Could not compute statistics for any pair.", file=sys.stderr)
        return 1

    print(f"\nMediaPipe Accuracy Report")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Frames analyzed: {len(df)}\n")
    print_results(results)

    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
