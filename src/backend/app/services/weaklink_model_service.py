# src/backend/app/services/weaklink_model_service.py
import os
import threading
from typing import Dict, Tuple

import mlflow
import numpy as np
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

_lock = threading.Lock()
_cache: Dict[str, Tuple[object, str]] = {}  # cache_key -> (model, uri_used)


def _clean_uri(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().strip('"').strip("'")


_initial_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if _initial_tracking_uri:
    mlflow.set_tracking_uri(_initial_tracking_uri)
    mlflow.set_registry_uri(_initial_tracking_uri)


def _direct_uri_for_variant(variant: str) -> str | None:
    v = (variant or "").lower().strip()
    if v in {"champion", "best", "prod", "production"}:
        return _clean_uri(os.getenv("WEAKLINK_MODEL_URI_PROD"))
    if v in {"latest", "dev", "development"}:
        return _clean_uri(os.getenv("WEAKLINK_MODEL_URI_DEV"))
    if v in {"backup"}:
        return _clean_uri(os.getenv("WEAKLINK_MODEL_URI_BACKUP"))
    return None


def _init_mlflow() -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")


def _is_models_alias_uri(uri: str) -> bool:
    return (
        uri.startswith("models:/")
        and ("@" in uri)
        and ("/" not in uri.replace("models:/", "", 1).split("@", 1)[0])
    )


def _parse_models_alias_uri(uri: str) -> tuple[str, str]:
    tail = uri.replace("models:/", "", 1)
    name, alias = tail.split("@", 1)
    name = name.strip()
    alias = alias.strip()
    if not name or not alias:
        raise ValueError(f"Invalid models alias uri: {uri}")
    return name, alias


def _resolve_alias_to_version_uri(model_name: str, alias: str) -> str:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    versions_sorted = sorted(versions, key=lambda mv: int(mv.version))
    latest = versions_sorted[-1]
    second_latest = versions_sorted[-2] if len(versions_sorted) >= 2 else latest

    a = alias.lower().strip()
    if a in {"prod", "production"}:
        prod_candidates = [
            mv
            for mv in versions_sorted
            if (getattr(mv, "current_stage", "") or "").lower() == "production"
        ]
        chosen = prod_candidates[-1] if prod_candidates else latest
        return f"models:/{model_name}/{chosen.version}"

    if a in {"dev", "latest"}:
        return f"models:/{model_name}/{latest.version}"

    if a in {"backup"}:
        return f"models:/{model_name}/{second_latest.version}"

    return f"models:/{model_name}/{latest.version}"


def _load_model_with_alias_fallback(uri: str) -> Tuple[object, str]:
    try:
        model = mlflow.pyfunc.load_model(uri)
        return model, uri
    except RestException as e:
        if _is_models_alias_uri(uri) and "INVALID_PARAMETER_VALUE" in str(e):
            model_name, alias = _parse_models_alias_uri(uri)
            resolved_uri = _resolve_alias_to_version_uri(model_name, alias)
            model = mlflow.pyfunc.load_model(resolved_uri)
            return model, resolved_uri
        raise


def get_model(variant: str = "champion") -> Tuple[object, str]:
    _init_mlflow()
    direct_uri = _direct_uri_for_variant(variant)
    if not direct_uri:
        raise RuntimeError("Weaklink model URI is not set for this variant")

    cache_key = direct_uri
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]
        model, uri_used = _load_model_with_alias_fallback(direct_uri)
        _cache[cache_key] = (model, uri_used)
        return model, uri_used


def _expected_feature_count_from_model(model: object) -> int | None:
    impl = getattr(model, "_model_impl", None)
    sk_model = getattr(impl, "sklearn_model", None) if impl else None
    n = getattr(sk_model, "n_features_in_", None) if sk_model else None
    return int(n) if n is not None else None


def expected_feature_count(variant: str = "champion") -> int | None:
    model, _ = get_model(variant)
    return _expected_feature_count_from_model(model)


def predict_one(features: list[float], variant: str = "champion") -> Tuple[str, str]:
    model, uri = get_model(variant)

    expected = _expected_feature_count_from_model(model)
    if expected is not None and len(features) != expected:
        raise ValueError(f"Model expects {expected} features, got {len(features)}")

    X = np.array([features], dtype=float)
    y = model.predict(X)

    # y[0] should be the class label (string)
    pred = y[0] if hasattr(y, "__len__") else y
    return str(pred), uri
