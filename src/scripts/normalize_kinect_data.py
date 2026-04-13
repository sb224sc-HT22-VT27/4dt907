#!/usr/bin/env python3
"""normalize_kinect_data.py

Normalizes Kinect CSV data files to be compatible with MediaPipe world landmark
data stored in the database after webcam/video/image uploads.

Two normalizations are applied
-------------------------------
1. **Name normalization** — Kinect joint names are remapped to the MediaPipe
   landmark vocabulary:

     head       → nose
     left_hand  → left_wrist
     right_hand → right_wrist
     left_foot  → left_ankle
     right_foot → right_ankle

   All other joint names shared by both formats (left_shoulder, left_elbow,
   right_shoulder, right_elbow, left_hip, right_hip, left_knee, right_knee)
   are left unchanged.

2. **Value normalization (hip-centering)** — MediaPipe world landmarks use the
   hip midpoint as the coordinate origin (per the MediaPipe documentation).
   Each frame's joint coordinates are shifted by the per-frame hip midpoint so
   that the origin is consistently at the centre of the hips, exactly matching
   MediaPipe's world-coordinate convention.

Usage
-----
    python normalize_kinect_data.py                        # auto-detects src/data
    python normalize_kinect_data.py --data-dir /path/data  # explicit data dir

Normalized files are written to new directories named
``<original_dir>_mediapipe`` next to the source directories, e.g.
``kinect_good_preprocessed_A9`` → ``kinect_good_preprocessed_A9_mediapipe``.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Kinect joint name → MediaPipe landmark name.
# Only joints whose names differ between the two formats are listed here.
KINECT_TO_MEDIAPIPE: dict[str, str] = {
    "head": "nose",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}

# Source directory names (relative to src/data/) that contain Kinect CSV files.
KINECT_DIRS = [
    "kinect_good_preprocessed_A9",
    "kinect_good_preprocessed_not_cut_A11",
    "kinect_good_vs_bad_not_preprocessed_A13",
]


def remap_column(col: str) -> str:
    """Rename a single column using the Kinect → MediaPipe mapping.

    Column names follow the pattern ``{joint}_{axis}`` (e.g. ``head_x``).
    The ``FrameNo`` column is passed through unchanged.

    Parameters
    ----------
    col:
        Original column name (may contain leading/trailing whitespace).

    Returns
    -------
    str
        Renamed column name with whitespace stripped.
    """
    col = col.strip()
    for kinect_name, mediapipe_name in KINECT_TO_MEDIAPIPE.items():
        prefix = kinect_name + "_"
        if col.startswith(prefix):
            return mediapipe_name + "_" + col[len(prefix):]
        if col == kinect_name:
            return mediapipe_name
    return col


def hip_center_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Center all joint coordinates at the per-frame hip midpoint.

    MediaPipe world landmarks use the hip midpoint as the coordinate origin
    (per the MediaPipe Pose documentation).  This transform applies the same
    convention to Kinect data so both sources share a common coordinate frame.

    For each frame the midpoint of the left and right hip joints is computed
    and subtracted from every joint's x, y, and z coordinates.  After the
    transform the left and right hip joints are symmetric around (0, 0, 0).

    Parameters
    ----------
    df:
        DataFrame with columns like ``<joint>_x``, ``<joint>_y``,
        ``<joint>_z``.  Must contain ``left_hip_x/y/z`` and
        ``right_hip_x/y/z`` columns (i.e. column-name normalization has
        already been applied).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with all joint x/y/z values shifted so the hip midpoint
        is at (0, 0, 0) for every frame.  Non-coordinate columns (e.g.
        ``FrameNo``) are left unchanged.
    """
    df = df.copy()
    for axis in ("x", "y", "z"):
        left_col = f"left_hip_{axis}"
        right_col = f"right_hip_{axis}"
        if left_col not in df.columns or right_col not in df.columns:
            continue
        hip_mid = (df[left_col] + df[right_col]) / 2.0
        axis_cols = [c for c in df.columns if c.endswith(f"_{axis}")]
        for col in axis_cols:
            df[col] = df[col] - hip_mid
    return df


def normalize_file(src: Path, dst: Path) -> None:
    """Read a Kinect CSV file, normalize names and values, and write to *dst*.

    Applies both normalizations in sequence:

    1. :func:`remap_column` — rename joints to the MediaPipe vocabulary.
    2. :func:`hip_center_normalize` — shift coordinates so the hip midpoint is
       the origin, matching MediaPipe world-landmark convention.

    Parameters
    ----------
    src:
        Path to the source Kinect CSV file.
    dst:
        Path where the normalized file will be written.  The parent directory
        is created if it does not already exist.
    """
    df = pd.read_csv(src)
    df.columns = [remap_column(c) for c in df.columns]
    df = hip_center_normalize(df)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    logger.debug("Normalized %s → %s", src.name, dst)


def normalize_directory(src_dir: Path, dst_dir: Path) -> int:
    """Normalize all CSV files in *src_dir* and write results to *dst_dir*.

    Parameters
    ----------
    src_dir:
        Directory containing raw Kinect CSV files.
    dst_dir:
        Directory where normalized files are written.

    Returns
    -------
    int
        Number of files normalized.
    """
    count = 0
    for csv_file in sorted(src_dir.glob("*.csv")):
        normalize_file(csv_file, dst_dir / csv_file.name)
        count += 1
    return count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data_default = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=data_default,
        help="Path to the src/data directory (default: auto-detected relative to this script)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    if not data_dir.is_dir():
        parser.error(f"Data directory not found: {data_dir}")

    total = 0
    for kinect_dir_name in KINECT_DIRS:
        src_dir = data_dir / kinect_dir_name
        if not src_dir.is_dir():
            logger.warning("Source directory not found, skipping: %s", src_dir)
            continue
        dst_dir_name = kinect_dir_name + "_mediapipe"
        dst_dir = data_dir / dst_dir_name
        count = normalize_directory(src_dir, dst_dir)
        logger.info(
            "Normalized %d file(s): %s → %s", count, kinect_dir_name, dst_dir_name
        )
        total += count

    logger.info("Total files normalized: %d", total)


if __name__ == "__main__":
    main()
