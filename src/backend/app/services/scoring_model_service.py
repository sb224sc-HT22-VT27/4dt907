"""app.services.scoring_model_service

Model loading + prediction utilities for squat scoring model.
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

from app.services import goodbad_model_service

_log = logging.getLogger(__name__)
_lock = threading.Lock()
# cache: uri → (pyfunc_model, uri_used, run_id, c_frames, n_features, scaler_or_None)
_cache: Dict[str, Tuple] = {}

_initial_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if _initial_tracking_uri:
    mlflow.set_tracking_uri(_initial_tracking_uri)
    mlflow.set_registry_uri(_initial_tracking_uri)

# Number of frames each session segment is resampled to before inference.
_DEFAULT_C_FRAMES = 10
# Expected number of engineered features per frame for the scoring model input.
_DEFAULT_N_FEATURES = 61


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
    if v in {"backup"}:
        return _clean_uri(os.getenv("SCORING_MODEL_URI_BACKUP"))
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
    if not run_id:
        return None
    try:
        import joblib

        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        scaler_path = next(
            (
                a.path
                for a in artifacts
                if a.path.startswith("scaler_scoring")
                or a.path.startswith("scaler_goodbad")
                or a.path.startswith("scaler")
            ),
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


def _normalize_score(raw: np.ndarray) -> Optional[float]:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim >= 2 and arr.shape[-1] > 1:
        value = float(np.argmax(arr[0]))
    else:
        value = float(arr.reshape(-1)[0])
    if not np.isfinite(value):
        return None
    return float(np.clip(value, 0.0, 4.0))


def predict_session(
    exercise_frames: List[List[Dict]],
    variant: str = "champion",
) -> Optional[float]:
    """Score a squat segment with an float score in [0.0, 4.0]."""
    if not exercise_frames:
        return None

    try:
        model, _, _, c_frames, n_features, scaler = get_model(variant)

        base = np.array(
            [goodbad_model_service._build_base_features(f) for f in exercise_frames],
            dtype=np.float32,
        )
        enriched = goodbad_model_service._add_dist_angle_features(base)
        fixed = goodbad_model_service._resample_to_fixed(enriched, c_frames)

        if scaler is not None:
            flat = fixed.reshape(1, -1)
            flat = scaler.transform(flat).astype(np.float32)
            fixed = flat.reshape(c_frames, n_features)

        raw = model.predict(fixed[None])
        score = _normalize_score(raw)
        _log.info("Scoring prediction=%s", score)
        return score
    except Exception as exc:
        _log.error("Scoring prediction failed: %s", exc, exc_info=True)
        return None
