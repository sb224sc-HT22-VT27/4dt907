"""app.services.goodbad_model_service

Model loading + prediction utilities for GoodBad_ClassifierV2 (A13).

Classifies whether a squat repetition is performed with Good or Bad form.
Output is a sigmoid probability [0, 1]:
  ~1.0  → Good form
  ~0.0  → Bad form
  ~0.5  → uncertain

Model architecture: ACNN (1-D Conv)
  Input shape : (batch, c_frames=10, n_features=61)
  The forward() permutes to (batch, n_features, c_frames) for Conv1d internally.

Feature engineering (matches a13-mlflow.ipynb exactly):
  39 base features : 13 joints × (x, y, z) in Kinect CSV column order
                     (head→nose, left_shoulder, left_elbow, left_hand→left_wrist,
                      right_shoulder, right_elbow, right_hand→right_wrist,
                      left_hip, left_knee, left_foot→left_ankle,
                      right_hip, right_knee, right_foot→right_ankle)
  16 dist features : Euclidean distances between key joint pairs,
                     normalised by mean shoulder width across the clip.
   6 angle features: Cosine of the angle at a vertex joint for key triples.
  Total: 39 + 16 + 6 = 61 features per frame.

Feature order MUST match the CSV column order from the training data
(kinect_good_vs_bad_not_preprocessed_A13_mediapipe) which preserves the
original Kinect joint output order after joint-name remapping.
"""

import logging
import os
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

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

# Joint names in Kinect CSV column order (after joint-name remapping).
# MUST match FEAT_COLS from the A13 training notebook exactly.
# Kinect outputs: head→nose, left_shoulder, left_elbow, left_hand→left_wrist,
#   right_shoulder, right_elbow, right_hand→right_wrist,
#   left_hip, left_knee, left_foot→left_ankle,
#   right_hip, right_knee, right_foot→right_ankle
_JOINT_NAMES: List[str] = [
    "nose",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
]

# Canonical feature column names in training order (mirrors FEAT_COLS).
_FEAT_COLS: List[str] = [
    f"{joint}_3d_{axis}" for joint in _JOINT_NAMES for axis in ["x", "y", "z"]
]

# 16 joint-pair distances (normalised by shoulder width).
_KEY_DIST_PAIRS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

# 6 joint-angle triples (a, vertex, b) — angle measured at vertex.
_KEY_ANGLE_TRIPLES: List[Tuple[str, str, str]] = [
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_elbow", "left_shoulder", "left_hip"),
    ("right_elbow", "right_shoulder", "right_hip"),
]


# ──────────────────────────────────────────────────────────────────────────────
# MLflow helpers (mirrors start_stop_model_service pattern)
# ──────────────────────────────────────────────────────────────────────────────


def _clean_uri(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().strip('"').strip("'")


def _direct_uri_for_variant(variant: str) -> Optional[str]:
    v = (variant or "").lower().strip()
    if v in {"champion", "best", "prod", "production"}:
        return _clean_uri(os.getenv("GOODBAD_MODEL_URI_PROD"))
    if v in {"latest", "dev", "development"}:
        return _clean_uri(os.getenv("GOODBAD_MODEL_URI_DEV"))
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
    """Load via mlflow.pyfunc (works with mlflow-skinny)."""
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
    """Try to load the GoodBad scaler artifact. Returns None if unavailable."""
    if not run_id:
        return None
    try:
        import joblib

        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        scaler_path = next(
            (a.path for a in artifacts if a.path.startswith("scaler_goodbad")),
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
        raise RuntimeError("GoodBad model URI is not configured for this variant")

    with _lock:
        if direct_uri in _cache:
            return _cache[direct_uri]

        model, uri_used = _load_model_with_alias_fallback(direct_uri)
        run_id = _fetch_run_id(uri_used)
        c_frames, n_features = _fetch_run_params(run_id)
        scaler = _fetch_scaler(run_id)

        _log.info(
            "GoodBad model loaded: uri=%s  c_frames=%d  n_features=%d  scaler=%s",
            uri_used,
            c_frames,
            n_features,
            "yes" if scaler else "no",
        )
        _cache[direct_uri] = (model, uri_used, run_id, c_frames, n_features, scaler)
        return model, uri_used, run_id, c_frames, n_features, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering — exact replica of add_dist_angle_features() from
# a13-mlflow.ipynb, using name-based column lookup to be order-independent
# for the distance/angle extras.
# ──────────────────────────────────────────────────────────────────────────────


def _pos(base_arr: np.ndarray, joint: str) -> np.ndarray:
    """Return (n_frames, 3) xyz slice for a joint, using _FEAT_COLS index lookup."""
    idxs = [_FEAT_COLS.index(f"{joint}_3d_{ax}") for ax in ["x", "y", "z"]]
    return base_arr[:, idxs]


def _build_base_features(kp3d: List[Dict]) -> List[float]:
    """39 floats: 13 joints × (x, y, z) in _JOINT_NAMES / _FEAT_COLS order."""
    kp_map = {kp["name"]: kp for kp in kp3d}
    feats: List[float] = []
    for name in _JOINT_NAMES:
        kp = kp_map.get(name)
        feats.append(float(kp["x"]) if kp else 0.0)
        feats.append(float(kp["y"]) if kp else 0.0)
        feats.append(float(kp["z"]) if kp else 0.0)
    return feats


def _add_dist_angle_features(base_arr: np.ndarray) -> np.ndarray:
    """Add 16 distance + 6 angle features to a (n_frames, 39) array.

    Matches add_dist_angle_features() from a13-mlflow.ipynb exactly:
    - Distances are normalised by the MEAN shoulder width across the clip.
    - Angle features are cosines computed per frame.
    - Uses _pos() name-lookup so feature order matches training.

    Returns (n_frames, 61) float32 array.
    """
    # Body scale: mean shoulder width across ALL frames in the clip.
    ls = _pos(base_arr, "left_shoulder")
    rs = _pos(base_arr, "right_shoulder")
    scale = float(np.linalg.norm(rs - ls, axis=1).mean()) + 1e-8

    extras: List[np.ndarray] = []

    for j1, j2 in _KEY_DIST_PAIRS:
        p1 = _pos(base_arr, j1)
        p2 = _pos(base_arr, j2)
        d = np.linalg.norm(p2 - p1, axis=1, keepdims=True) / scale
        extras.append(d)

    for ja, jv, jb in _KEY_ANGLE_TRIPLES:
        pa = _pos(base_arr, ja)
        pv = _pos(base_arr, jv)
        pb = _pos(base_arr, jb)
        va = pa - pv
        vb = pb - pv
        cos_val = np.sum(va * vb, axis=1) / (
            np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-8
        )
        extras.append(cos_val.reshape(-1, 1))

    return np.hstack([base_arr] + extras).astype(np.float32)


def _resample_to_fixed(arr: np.ndarray, target: int) -> np.ndarray:
    """Sample `target` equidistant rows from arr (matches to_fixed_length in notebook)."""
    n = len(arr)
    if n == target:
        return arr
    if n < target:
        pad = np.zeros((target - n, arr.shape[1]), dtype="float32")
        return np.vstack([arr, pad])
    idx = np.round(np.linspace(0, n - 1, target)).astype(int)
    return arr[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Public inference API
# ──────────────────────────────────────────────────────────────────────────────


def predict_session(
    exercise_frames: List[List[Dict]],
    variant: str = "champion",
) -> Optional[float]:
    """Score a single squat repetition.

    Parameters
    ----------
    exercise_frames:
        All frames belonging to one continuous squat rep (start_stop == 1).
        Each element is a list of keypoint dicts with keys ``name``, ``x``,
        ``y``, ``z``.  Must be image-normalised coordinates (x, y ∈ [0, 1])
        the coordinate system used in training(kinect_good_vs_bad_not_preprocessed_A13_mediapipe).
        Passing world-space (hip-centred) coords will produce all-Bad predictions.
    variant:
        Model variant (``"champion"`` / ``"prod"`` or ``"dev"``).

    Returns
    -------
    float or None
        Sigmoid probability in [0, 1].
        ~1.0 = Good form, ~0.0 = Bad form.
        Returns None on any error so the caller can hide the result.
    """
    if not exercise_frames:
        return 0.5

    try:
        model, _, _, c_frames, n_features, scaler = get_model(variant)

        # 1. Build base (39) features for ALL exercise frames first.
        #    Enrichment before resampling matches the training notebook order,
        #    so shoulder-width scale is computed from the full segment.
        base = np.array(
            [_build_base_features(f) for f in exercise_frames], dtype=np.float32
        )  # (n_frames, 39)

        # 2. Enrich with distance + angle features → (n_frames, 61).
        enriched = _add_dist_angle_features(base)

        _log.info(
            "GoodBad features: n_frames=%d shape=%s mean=%.4f min=%.4f max=%.4f",
            len(exercise_frames),
            enriched.shape,
            float(enriched.mean()),
            float(enriched.min()),
            float(enriched.max()),
        )

        # 3. Resample to exactly c_frames equidistant frames → (c_frames, 61).
        fixed = _resample_to_fixed(enriched, c_frames)

        # 4. Optional scaler (applied flat, same as training: reshape→transform→reshape).
        if scaler is not None:
            flat = fixed.reshape(1, -1)  # (1, c_frames * n_features)
            flat = scaler.transform(flat).astype(np.float32)
            fixed = flat.reshape(c_frames, n_features)

        # 5. Shape (1, c_frames, n_features) → pyfunc predict → raw logit.
        X = fixed[None]  # (1, c_frames, n_features)
        raw = model.predict(X)
        logit = float(np.asarray(raw, dtype=np.float32).flatten()[0])
        score = float(1.0 / (1.0 + np.exp(-logit)))

        _log.info(
            "GoodBad logit=%.4f → score=%.4f (frames=%d → resampled to %d)",
            logit,
            score,
            len(exercise_frames),
            c_frames,
        )
        return score

    except Exception as exc:
        _log.error("GoodBad prediction failed: %s", exc, exc_info=True)
        return None
