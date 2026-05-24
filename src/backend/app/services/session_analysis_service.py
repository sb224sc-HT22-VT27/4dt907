"""app.services.session_analysis_service

Full session analysis pipeline: Cut → Z-pred → GoodBad → Results.

Pipeline per session (all frames at once):
1. Build per-frame feature vectors (39 floats: 13 joints × [x, y, z]).
2. Run Start_Stop_Predictor_ModelV2 on all frames → [0/1, ...].
3. Apply gap-fill smoothing: 0-runs < 10 frames between two 1-regions → 1.
4. For every frame: predict z for all 13 joints.
5. Run GoodBad_ClassifierV2 on each continuous exercise segment → quality score [0,1].
6. Return per-frame results.
"""

import logging as _logging
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from app.services import start_stop_model_service
from app.services import goodbad_model_service
from app.services.z_model_service import predict_batch as predict_z_batch

_log = _logging.getLogger(__name__)

_MODEL_JOINT_NAMES: List[str] = [
    "nose",
    "left_shoulder",
    "left_elbow",
    "right_shoulder",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _build_features(kp3d: List[Dict]) -> List[float]:
    """39-float feature vector [j0_x, j0_y, j0_z, …] in _MODEL_JOINT_NAMES order."""
    kp_map = {kp["name"]: kp for kp in kp3d}
    feats: List[float] = []
    for name in _MODEL_JOINT_NAMES:
        kp = kp_map.get(name)
        feats.append(float(kp["x"]) if kp else 0.0)
        feats.append(float(kp["y"]) if kp else 0.0)
        feats.append(float(kp["z"]) if kp else 0.0)
    return feats


def _predict_z_all_frames(frames: List[List[Dict]]) -> List[Dict[str, float]]:
    """Predict z for all joints across all frames in a single batch model call."""
    entries = []
    for i, kp3d in enumerate(frames):
        kp_map = {kp["name"]: kp for kp in kp3d}
        for name in _MODEL_JOINT_NAMES:
            kp = kp_map.get(name)
            if kp:
                entries.append((i, name, [float(kp["x"]), float(kp["y"])]))

    results: List[Dict[str, float]] = [{} for _ in frames]
    if entries:
        try:
            z_values = predict_z_batch([e[2] for e in entries], "champion")
            for (i, name, _), z in zip(entries, z_values):
                results[i][name] = z
        except Exception as exc:
            _log.error("z-batch prediction failed: %s", exc)

    for i, kp3d in enumerate(frames):
        kp_map = {kp["name"]: kp for kp in kp3d}
        for name in _MODEL_JOINT_NAMES:
            if name not in results[i]:
                kp = kp_map.get(name)
                results[i][name] = float(kp.get("z", 0.0)) if kp else 0.0

    return results


def _smooth_start_stop(predictions: List[int], gap_threshold: int = 10) -> List[int]:
    """Fill 0-gaps < gap_threshold between two 1-regions with 1."""
    result = list(predictions)
    n = len(result)
    i = 0
    while i < n:
        if result[i] == 1:
            j = i + 1
            while j < n and result[j] == 1:
                j += 1
            gap_start = j
            while j < n and result[j] == 0:
                j += 1
            if j < n:
                gap_len = j - gap_start
                if gap_len < gap_threshold:
                    for k in range(gap_start, j):
                        result[k] = 1
                i = j
            else:
                break
        else:
            i += 1
    return result


class FrameResult:
    __slots__ = ("start_stop", "predicted_z", "good_bad_score")

    def __init__(
        self,
        start_stop: int,
        predicted_z: Dict[str, float],
        good_bad_score: Optional[float] = None,
    ):
        self.start_stop = start_stop
        self.predicted_z = predicted_z
        self.good_bad_score = good_bad_score


def analyze_session(
    frames: List[List[Dict]],
    norm_frames: Optional[List[List[Dict]]] = None,
) -> Tuple[List[FrameResult], Dict[str, float]]:
    """Run the full pipeline on all frames.

    Returns (results, timings) where timings maps step name → elapsed ms.
    """
    if not frames:
        return [], {}

    timings: Dict[str, float] = {}
    t_total = perf_counter()

    feature_source = (
        norm_frames if (norm_frames and len(norm_frames) == len(frames)) else frames
    )

    t = perf_counter()
    features_batch = [_build_features(f) for f in feature_source]
    timings["feature_build_ms"] = round((perf_counter() - t) * 1000, 1)

    t = perf_counter()
    try:
        raw_start_stop = start_stop_model_service.predict_batch(
            features_batch, "champion"
        )
        _log.info(
            "start_stop: %d frames → %d exercise, %d non-exercise",
            len(raw_start_stop),
            sum(raw_start_stop),
            raw_start_stop.count(0),
        )
        if sum(raw_start_stop) == 0:
            _log.warning("start_stop returned all-0 — falling back to all-exercise")
            raw_start_stop = [1] * len(frames)
    except Exception as exc:
        _log.error("start_stop model failed (%s), falling back to all-exercise", exc)
        raw_start_stop = [1] * len(frames)
    timings["start_stop_ms"] = round((perf_counter() - t) * 1000, 1)

    t = perf_counter()
    smoothed = _smooth_start_stop(raw_start_stop)
    timings["smooth_ms"] = round((perf_counter() - t) * 1000, 1)

    t = perf_counter()
    all_predicted_z = _predict_z_all_frames(frames)
    timings["z_prediction_ms"] = round((perf_counter() - t) * 1000, 1)

    results: List[FrameResult] = [
        FrameResult(start_stop=ss, predicted_z=pz)
        for ss, pz in zip(smoothed, all_predicted_z)
    ]

    t = perf_counter()
    _score_exercise_segments(norm_frames or frames, smoothed, results)
    timings["goodbad_ms"] = round((perf_counter() - t) * 1000, 1)

    timings["total_ms"] = round((perf_counter() - t_total) * 1000, 1)

    return results, timings


def _score_exercise_segments(
    source_frames: List[List[Dict]],
    smoothed: List[int],
    results: List[FrameResult],
) -> None:
    """Run GoodBad_ClassifierV2 on each continuous exercise segment in-place.

    source_frames must be image-normalised keypoints (x, y ∈ [0,1]).
    """
    n = len(smoothed)
    i = 0
    while i < n:
        if smoothed[i] == 1:
            seg_start = i
            while i < n and smoothed[i] == 1:
                i += 1
            seg_end = i

            seg_frames = source_frames[seg_start:seg_end]
            try:
                score = goodbad_model_service.predict_session(seg_frames, "champion")
            except Exception as exc:
                _log.error(
                    "GoodBad scoring failed for segment [%d:%d]: %s",
                    seg_start,
                    seg_end,
                    exc,
                )
                score = None

            _log.info(
                "GoodBad segment [%d:%d] (%d frames): score=%.4f",
                seg_start,
                seg_end,
                seg_end - seg_start,
                score,
            )
            for j in range(seg_start, seg_end):
                results[j].good_bad_score = score
        else:
            i += 1
