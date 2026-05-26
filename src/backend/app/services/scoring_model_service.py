"""app.services.scoring_model_service

Model loading + prediction utilities for squat scoring.

Produces a squat score in the range [0, 4], where:
  0.0 → good form
  4.0 → bad form
"""

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from app.services import goodbad_model_service

_log = logging.getLogger(__name__)
_lock = threading.Lock()
# cache: uri → (pyfunc_model, uri_used, run_id, c_frames, n_features, scaler_or_None)
_cache: Dict[str, Tuple] = {}

_initial_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if _initial_tracking_uri:
    mlflow.set_tracking_uri(_initial_tracking_uri)
    mlflow.set_registry_uri(_initial_tracking_uri)

_DEFAULT_C_FRAMES = 10
_DEFAULT_N_FEATURES = 61

# Joint order from kinect_good_preprocessed_not_cut_A11_mediapipe CSV columns.
# Confirmed by reading A1_kinect.csv header — matches FEAT_COLS in the A15 notebook.
# Different from goodbad_model_service._JOINT_NAMES (Kinect SDK order).
_A15_JOINTS: List[str] = [
    "nose",
    "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
_A15_FEAT_COLS: List[str] = [
    f"{j}_3d_{ax}" for j in _A15_JOINTS for ax in ["x", "y", "z"]
]


def _a15_build_base(kp3d: List[Dict]) -> List[float]:
    # MediaPipe JS world_landmarks vs A11 training CSV (Kinect/MediaPipe Python) differences:
    #   x: JS uses camera frame (person's left = positive x); training uses body frame (left = negative x). Negate x.
    #   y: JS uses y-down; training uses y-up. Negate y.
    #   z: JS z is camera-depth (large, −0.3 to −0.5); training z is body-forward (tiny, ~0.005–0.06). Zero out z.
    kp_map = {kp["name"]: kp for kp in kp3d}
    feats: List[float] = []
    for name in _A15_JOINTS:
        kp = kp_map.get(name)
        feats.append(-float(kp["x"]) if kp else 0.0)
        feats.append(-float(kp["y"]) if kp else 0.0)
        feats.append(0.0)
    return feats


def _a15_pos(arr: np.ndarray, joint: str) -> np.ndarray:
    idxs = [_A15_FEAT_COLS.index(f"{joint}_3d_{ax}") for ax in ["x", "y", "z"]]
    return arr[:, idxs]


def _a15_add_features(base_arr: np.ndarray) -> np.ndarray:
    """16 distance + 6 angle features using A15 column-order index lookup."""
    ls = _a15_pos(base_arr, "left_shoulder")
    rs = _a15_pos(base_arr, "right_shoulder")
    scale = float(np.linalg.norm(rs - ls, axis=1).mean()) + 1e-8
    extras: List[np.ndarray] = []
    for j1, j2 in goodbad_model_service._KEY_DIST_PAIRS:
        p1, p2 = _a15_pos(base_arr, j1), _a15_pos(base_arr, j2)
        extras.append(np.linalg.norm(p2 - p1, axis=1, keepdims=True) / scale)
    for ja, jv, jb in goodbad_model_service._KEY_ANGLE_TRIPLES:
        pa = _a15_pos(base_arr, ja)
        pv = _a15_pos(base_arr, jv)
        pb = _a15_pos(base_arr, jb)
        va, vb = pa - pv, pb - pv
        cos = np.sum(va * vb, axis=1) / (
            np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-8
        )
        extras.append(cos.reshape(-1, 1))
    return np.hstack([base_arr] + extras).astype(np.float32)


def _clean_uri(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().strip('"').strip("'")


def _direct_uri_for_variant(variant: str) -> Optional[str]:
    v = (variant or "").lower().strip()
    if v in {"champion", "best", "prod", "production"}:
        return _clean_uri(os.getenv("SCORING_MODEL_URI_PROD"))
    if v in {"latest", "dev", "development"}:
        return _clean_uri(os.getenv("SCORING_MODEL_URI_DEV"))
    return None


def _init_mlflow() -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")


def _is_models_alias_uri(uri: str) -> bool:
    return (
        uri.startswith("models:/")
        and "@" in uri
        and "/" not in uri.replace("models:/", "", 1).split("@", 1)[0]
    )


def _parse_models_alias_uri(uri: str) -> Tuple[str, str]:
    tail = uri.replace("models:/", "", 1)
    name, alias = tail.split("@", 1)
    return name.strip(), alias.strip()


def _resolve_alias_to_version_uri(model_name: str, alias: str) -> str:
    client = MlflowClient()
    versions = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda mv: int(mv.version),
    )
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")
    latest = versions[-1]
    second_latest = versions[-2] if len(versions) >= 2 else latest
    a = alias.lower().strip()
    if a in {"prod", "production"}:
        prod = [
            mv
            for mv in versions
            if (getattr(mv, "current_stage", "") or "").lower() == "production"
        ]
        chosen = prod[-1] if prod else latest
    elif a in {"backup"}:
        chosen = second_latest
    else:
        chosen = latest
    return f"models:/{model_name}/{chosen.version}"


def _load_model_with_alias_fallback(uri: str):
    try:
        return mlflow.pyfunc.load_model(uri), uri
    except RestException as e:
        if _is_models_alias_uri(uri) and "INVALID_PARAMETER_VALUE" in str(e):
            name, alias = _parse_models_alias_uri(uri)
            resolved = _resolve_alias_to_version_uri(name, alias)
            return mlflow.pyfunc.load_model(resolved), resolved
        raise


def _fetch_run_id(uri: str) -> Optional[str]:
    if not uri:
        return None
    if uri.startswith("runs:/"):
        tail = uri[len("runs:/") :]
        return tail.split("/", 1)[0] if tail else None
    client = MlflowClient()
    if _is_models_alias_uri(uri):
        name, alias = _parse_models_alias_uri(uri)
        try:
            mv = client.get_model_version_by_alias(name, alias)
            return getattr(mv, "run_id", None)
        except Exception:
            resolved = _resolve_alias_to_version_uri(name, alias)
            return _fetch_run_id(resolved)
    if uri.startswith("models:/"):
        tail = uri[len("models:/") :]
        parts = tail.split("/", 2)
        if len(parts) < 2:
            return None
        name = parts[0].strip()
        selector = parts[1].strip()
        if selector.isdigit():
            mv = client.get_model_version(name, selector)
            return getattr(mv, "run_id", None)
        latest = client.get_latest_versions(name, stages=[selector])
        if latest:
            return getattr(max(latest, key=lambda mv: int(mv.version)), "run_id", None)
    return None


def _fetch_run_params(run_id: Optional[str]) -> Tuple[int, int]:
    if not run_id:
        return _DEFAULT_C_FRAMES, _DEFAULT_N_FEATURES
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        c_frames = int(params.get("c_frames", _DEFAULT_C_FRAMES))
        n_features = int(params.get("n_features", _DEFAULT_N_FEATURES))
        return c_frames, n_features
    except Exception:
        return _DEFAULT_C_FRAMES, _DEFAULT_N_FEATURES


def _fetch_scaler(run_id: Optional[str]):
    if not run_id:
        return None
    try:
        import joblib
        import tempfile

        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        scaler_path = next(
            (a.path for a in artifacts if a.path.startswith("scaler_")),
            None,
        )
        if not scaler_path:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, scaler_path, tmp)
            return joblib.load(local)
    except Exception:
        return None


def get_model(variant: str = "champion"):
    """Return (model, uri_used, run_id, c_frames, n_features, scaler)."""
    _init_mlflow()
    direct_uri = _direct_uri_for_variant(variant)
    if not direct_uri:
        raise RuntimeError("Scoring model URI is not configured for this variant")

    with _lock:
        if direct_uri in _cache:
            return _cache[direct_uri]

        model, uri_used = _load_model_with_alias_fallback(direct_uri)
        run_id = _fetch_run_id(uri_used)
        c_frames, n_features = _fetch_run_params(run_id)
        scaler = _fetch_scaler(run_id)

        _cache[direct_uri] = (model, uri_used, run_id, c_frames, n_features, scaler)
        return model, uri_used, run_id, c_frames, n_features, scaler


def predict_session(
    exercise_frames: List[List[Dict]],
    variant: str = "champion",
) -> Optional[float]:
    """Score a single squat repetition on a 0..4 scale."""
    if not exercise_frames:
        return None

    try:
        model, _, _, c_frames, n_features, scaler = get_model(variant)

        base = np.array(
            [_a15_build_base(f) for f in exercise_frames],
            dtype=np.float32,
        )
        enriched = _a15_add_features(base)
        fixed = goodbad_model_service._resample_to_fixed(enriched, c_frames)

        if scaler is not None:
            flat = fixed.reshape(1, -1)
            flat = scaler.transform(flat).astype(np.float32)
            fixed = flat.reshape(c_frames, n_features)

        X = fixed[None]
        raw = model.predict(X)
        score = float(np.clip(float(np.asarray(raw, dtype=np.float32).flatten()[0]), 0.0, 4.0))
        _log.info("Scoring: %d frames → %.3f", len(exercise_frames), score)
        return score

    except Exception as exc:
        _log.error("Scoring prediction failed: %s", exc, exc_info=True)
        return None
