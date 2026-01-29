import os
import threading
from typing import Dict, Tuple

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

_lock = threading.Lock()
_cache: Dict[str, Tuple[object, str]] = {}  # cache_key -> (model, uri)
_mlflow_inited = False


def _init_mlflow() -> str:
    global _mlflow_inited
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    # Only set once to avoid repeated setup on every request
    if not _mlflow_inited:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        _mlflow_inited = True

    return uri


def _model_name_for_variant(variant: str) -> str:
    v = (variant or "").lower().strip()
    if v in {"champion", "best"}:
        name = os.getenv("MLFLOW_BEST_MODEL_NAME")
        if not name:
            raise RuntimeError("MLFLOW_BEST_MODEL_NAME is not set")
        return name

    if v in {"latest"}:
        name = os.getenv("MLFLOW_LATEST_MODEL_NAME")
        if not name:
            raise RuntimeError("MLFLOW_LATEST_MODEL_NAME is not set")
        return name

    raise RuntimeError("variant must be 'champion' or 'latest'")


def _latest_source_uri(model_name: str) -> str:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    # pick highest numeric version
    latest_mv = max(versions, key=lambda mv: int(mv.version))

    
    if not getattr(latest_mv, "source", None):
        raise RuntimeError(f"Model version {latest_mv.version} has no source URI")
    return latest_mv.source



def get_model(variant: str = "champion") -> Tuple[object, str]:
    _init_mlflow()
    model_name = _model_name_for_variant(variant)

    # Cache per model name (since best/latest are two different registry models)
    cache_key = model_name

    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        uri = _latest_source_uri(model_name)
        model = mlflow.pyfunc.load_model(uri)


        _cache[cache_key] = (model, uri)
        return model, uri


def predict_one(features: list[float], variant: str = "champion") -> Tuple[float, str]:
    model, uri = get_model(variant)

    expected = expected_feature_count(variant)
    if expected is not None and len(features) != expected:
        raise ValueError(f"Model expects {expected} features, got {len(features)}")

    X = np.array([features], dtype=float)
    y = model.predict(X)
    return (float(y[0]) if hasattr(y, "__len__") else float(y)), uri


def clear_model_cache() -> None:
    """Optional helper (useful for debugging/reloads)."""
    global _cache
    with _lock:
        _cache = {}

def expected_feature_count(variant: str = "latest") -> int | None:
    model, _uri = get_model(variant)

    # MLflow sklearn pyfunc wrapper stores the sklearn model here
    impl = getattr(model, "_model_impl", None)
    sk_model = getattr(impl, "sklearn_model", None) if impl else None

    # scikit-learn models typically expose n_features_in_
    n = getattr(sk_model, "n_features_in_", None) if sk_model else None
    return int(n) if n is not None else None


