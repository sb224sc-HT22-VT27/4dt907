import os
import threading
from typing import Dict, Tuple

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

_lock = threading.Lock()
_cache: Dict[str, Tuple[object, str]] = {}  # cache_key -> (model, uri_used)
_mlflow_inited = False


def _clean_uri(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().strip('"').strip("'")


# * Expected "entry point"
def _direct_uri_for_variant(variant: str) -> str | None:
    v = (variant or "").lower().strip()
    if v in {"champion", "best", "prod", "production"}:
        return _clean_uri(os.getenv("MODEL_URI_PROD"))
    if v in {"latest", "dev", "development"}:
        return _clean_uri(os.getenv("MODEL_URI_DEV"))
    if v in {"backup"}:
        return _clean_uri(os.getenv("MODEL_URI_BACKUP"))
    return None


def _init_mlflow() -> str:
    global _mlflow_inited
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    if not _mlflow_inited:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        _mlflow_inited = True

    return uri


# * Current "entry point"
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

    latest_mv = max(versions, key=lambda mv: int(mv.version))

    if not getattr(latest_mv, "source", None):
        raise RuntimeError(f"Model version {latest_mv.version} has no source URI")

    return latest_mv.source


def _is_models_alias_uri(uri: str) -> bool:
    # True for: models:/Name@alias
    return (
        uri.startswith("models:/")
        and ("@" in uri)
        and ("/" not in uri.replace("models:/", "", 1).split("@", 1)[0])
    )


def _parse_models_alias_uri(uri: str) -> tuple[str, str]:
    # models:/Project_Model@dev  -> ("Project_Model", "dev")
    tail = uri.replace("models:/", "", 1)
    name, alias = tail.split("@", 1)
    name = name.strip()
    alias = alias.strip()
    if not name or not alias:
        raise ValueError(f"Invalid models alias uri: {uri}")
    return name, alias


def _resolve_alias_to_version_uri(model_name: str, alias: str) -> str:
    """
    Dynamic fallback if alias lookup is not supported by registry.

    Alias semantics:
    - prod/production: prefer latest version in Production stage if any, else latest version overall
    - dev/latest: latest version overall
    - backup: second-latest version overall (or latest if only one exists)
    """
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

    # Unknown alias -> safest default
    return f"models:/{model_name}/{latest.version}"


def _load_model_with_alias_fallback(uri: str) -> Tuple[object, str]:
    """
    Try loading the provided URI. If it is an alias URI (models:/Name@alias) and
    the registry rejects alias lookup, resolve to a concrete version URI dynamically.
    Returns (model, uri_used).
    """
    try:
        model = mlflow.pyfunc.load_model(uri)
        return model, uri
    except RestException as e:
        # DagsHub/registry may reject get_model_version_by_alias with INVALID_PARAMETER_VALUE
        if _is_models_alias_uri(uri) and "INVALID_PARAMETER_VALUE" in str(e):
            model_name, alias = _parse_models_alias_uri(uri)
            resolved_uri = _resolve_alias_to_version_uri(model_name, alias)
            model = mlflow.pyfunc.load_model(resolved_uri)
            return model, resolved_uri
        raise


def get_model(variant: str = "champion") -> Tuple[object, str] | None:
    _init_mlflow()
    try:
        direct_uri = _direct_uri_for_variant(variant)
        if direct_uri:
            cache_key = direct_uri
            with _lock:
                if cache_key in _cache:
                    return _cache[cache_key]

                model, uri_used = _load_model_with_alias_fallback(direct_uri)
                _cache[cache_key] = (model, uri_used)
                return model, uri_used
    except RestException:
        raise


def _expected_feature_count_from_model(model: object) -> int | None:
    impl = getattr(model, "_model_impl", None)
    sk_model = getattr(impl, "sklearn_model", None) if impl else None
    n = getattr(sk_model, "n_features_in_", None) if sk_model else None
    return int(n) if n is not None else None


def expected_feature_count(variant: str = "champion") -> int | None:
    model, _uri = get_model(variant)
    return _expected_feature_count_from_model(model)


def predict_one(features: list[float], variant: str = "champion") -> Tuple[float, str]:
    model, uri = get_model(variant)

    expected = _expected_feature_count_from_model(model)
    if expected is not None and len(features) != expected:
        raise ValueError(f"Model expects {expected} features, got {len(features)}")

    X = np.array([features], dtype=float)
    y = model.predict(X)
    return (float(y[0]) if hasattr(y, "__len__") else float(y)), uri


def clear_model_cache() -> None:
    global _cache
    with _lock:
        _cache = {}
