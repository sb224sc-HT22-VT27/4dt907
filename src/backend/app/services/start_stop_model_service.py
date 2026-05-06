"""app.services.start_stop_model_service

Model loading + prediction utilities for Start_Stop_Predictor_ModelV2.

Predicts per-frame whether a pose frame belongs to an exercise (1) or not (0).
Input features: 39 floats = 13 joints × [x, y, z] in Kinect SDK column order
(nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist,
right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle).

The RNN model requires sliding windows of seq_length consecutive frames.
seq_length is read from the MLflow run params at load time (logged by training).
If use_scaling=True was logged, a MinMaxScaler is attempted from run artifacts.
"""

import os
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

_lock = threading.Lock()
# cache: uri → (model, uri_used, run_id, seq_len, scaler_or_None)
_cache: Dict[str, Tuple[object, str, Optional[str], int, Optional[object]]] = {}

_initial_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if _initial_tracking_uri:
    mlflow.set_tracking_uri(_initial_tracking_uri)
    mlflow.set_registry_uri(_initial_tracking_uri)

_DEFAULT_SEQ_LEN = 5


def _clean_uri(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().strip('"').strip("'")


def _direct_uri_for_variant(variant: str) -> Optional[str]:
    v = (variant or "").lower().strip()
    if v in {"champion", "best", "prod", "production"}:
        return _clean_uri(os.getenv("START_STOP_MODEL_URI_PROD"))
    if v in {"latest", "dev", "development"}:
        return _clean_uri(os.getenv("START_STOP_MODEL_URI_DEV"))
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
        prod = [mv for mv in versions if (getattr(mv, "current_stage", "") or "").lower() == "production"]
        chosen = prod[-1] if prod else latest
    elif a in {"backup"}:
        chosen = second_latest
    else:
        chosen = latest
    return f"models:/{model_name}/{chosen.version}"


def _load_model_with_alias_fallback(uri: str) -> Tuple[object, str]:
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
        tail = uri[len("runs:/"):]
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
        tail = uri[len("models:/"):]
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


def _fetch_run_params(run_id: Optional[str]) -> Tuple[int, bool]:
    """Read seq_length and use_scaling logged by the training notebook."""
    if not run_id:
        return _DEFAULT_SEQ_LEN, False
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        seq_len = int(params.get("seq_length", _DEFAULT_SEQ_LEN))
        use_scaling = str(params.get("use_scaling", "False")).strip().lower() == "true"
        return seq_len, use_scaling
    except Exception:
        return _DEFAULT_SEQ_LEN, False


def _fetch_scaler(run_id: Optional[str]) -> Optional[object]:
    """Try to download MinMaxScaler from MLflow artifacts. Returns None if unavailable."""
    if not run_id:
        return None
    try:
        import joblib
        client = MlflowClient()
        tmp = tempfile.mkdtemp()
        path = client.download_artifacts(run_id, "scaler_best.joblib", tmp)
        return joblib.load(path)
    except Exception:
        return None


def get_model(variant: str = "champion") -> Tuple[object, str, Optional[str], int, Optional[object]]:
    """Return (model, uri_used, run_id, seq_len, scaler_or_None)."""
    _init_mlflow()
    direct_uri = _direct_uri_for_variant(variant)
    if not direct_uri:
        raise RuntimeError("Start/stop model URI is not configured for this variant")

    cache_key = direct_uri
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]
        model, uri_used = _load_model_with_alias_fallback(direct_uri)
        run_id = _fetch_run_id(uri_used)
        seq_len, use_scaling = _fetch_run_params(run_id)
        scaler = _fetch_scaler(run_id) if use_scaling else None
        _cache[cache_key] = (model, uri_used, run_id, seq_len, scaler)
        return model, uri_used, run_id, seq_len, scaler


def predict_batch(
    features_list: List[List[float]], variant: str = "champion"
) -> List[int]:
    """Predict start/stop label for each frame.

    Builds sliding windows of seq_length frames (read from MLflow run params),
    zero-padding the first seq_length-1 frames.  If a MinMaxScaler was logged
    as an artifact, it is applied before inference.

    Parameters
    ----------
    features_list:
        List of N feature vectors, each 39 floats
        [nose_x, nose_y, nose_z, left_shoulder_x, …] in Kinect SDK column order
        (13 joints × 3 = 39), matching the training column layout exactly.
    variant:
        Model variant (``"champion"`` or ``"latest"``).

    Returns
    -------
    List of ints, one per frame: 0 = not exercise, 1 = in exercise.
    """
    model, _, _, seq_len, scaler = get_model(variant)
    n = len(features_list)
    n_feats = len(features_list[0]) if features_list else 39
    X = np.array(features_list, dtype=np.float32)  # (N, 39)

    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)

    # Build sliding windows: frame i uses frames [i-seq_len+1 … i], zero-padded at start.
    windows = np.zeros((n, seq_len, n_feats), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        chunk = X[start : i + 1]
        windows[i, seq_len - len(chunk) :] = chunk

    # model.predict returns raw logits (BCEWithLogitsLoss, no sigmoid in forward).
    # logit > 0  ↔  sigmoid(logit) > 0.5
    logits = np.asarray(model.predict(windows), dtype=np.float32).flatten()
    return [int(l > 0.0) for l in logits]


def get_mae_total_average(variant: str = "champion") -> Optional[float]:
    """Fetch MAE_Total_Average from the MLflow run for the start/stop model."""
    try:
        _, _, run_id, _, _ = get_model(variant)
        if not run_id:
            return None
        client = MlflowClient()
        run = client.get_run(run_id)
        val = run.data.metrics.get("MAE_Total_Average")
        return float(val) if val is not None else None
    except Exception:
        return None
