#!/usr/bin/env python3
"""normalize_kinect_data.py

Normalizes Kinect CSV data files to use MediaPipe-compatible landmark names.

Kinect → MediaPipe mapping
--------------------------
  head       → nose
  left_hand  → left_wrist
  right_hand → right_wrist
  left_foot  → left_ankle
  right_foot → right_ankle

All other joint names shared by both formats (left_shoulder, left_elbow,
right_shoulder, right_elbow, left_hip, right_hip, left_knee, right_knee)
are left unchanged.

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


def normalize_file(src: Path, dst: Path) -> None:
    """Read a Kinect CSV file, rename columns, and write to *dst*.

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
