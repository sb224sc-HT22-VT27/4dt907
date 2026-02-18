# src/backend/app/services/weaklink_model_service.py
import os
import re
import threading
from typing import Dict, Tuple

import mlflow
import numpy as np
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

_lock = threading.Lock()
_cache: Dict[str, Tuple[object, str, str | None]] = (
    {}
)  # cache_key -> (model, uri_used, run_id)


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
        # Some registries don’t support @alias. If so, resolve manually.
        if _is_models_alias_uri(uri) and "INVALID_PARAMETER_VALUE" in str(e):
            model_name, alias = _parse_models_alias_uri(uri)
            resolved_uri = _resolve_alias_to_version_uri(model_name, alias)
            model = mlflow.pyfunc.load_model(resolved_uri)
            return model, resolved_uri
        raise


def _fetch_run_id(uri: str) -> str | None:
    if not uri:
        return None

    # runs:/<run_id>/...
    if uri.startswith("runs:/"):
        tail = uri[len("runs:/") :]
        return tail.split("/", 1)[0] if tail else None

    client = MlflowClient()

    # models:/Name@alias
    if _is_models_alias_uri(uri):
        name, alias = _parse_models_alias_uri(uri)
        try:
            mv = client.get_model_version_by_alias(name, alias)
            return getattr(mv, "run_id", None)
        except Exception:
            # Fall back to your manual resolver
            resolved = _resolve_alias_to_version_uri(name, alias)
            return _fetch_run_id(resolved)

    # models:/Name/<version>[/...]
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

        # stage name case (optional)
        latest = client.get_latest_versions(name, stages=[selector])
        if latest:
            chosen = max(latest, key=lambda mv: int(mv.version))
            return getattr(chosen, "run_id", None)

    return None


def get_model(variant: str = "champion") -> Tuple[object, str, str | None]:
    _init_mlflow()
    direct_uri = _direct_uri_for_variant(variant)
    if not direct_uri:
        raise RuntimeError("Weaklink model URI is not set for this variant")

    cache_key = direct_uri
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        model, uri_used = _load_model_with_alias_fallback(direct_uri)
        run_id = _fetch_run_id(uri_used)

        _cache[cache_key] = (model, uri_used, run_id)
        return model, uri_used, run_id


def _expected_feature_count_from_model(model: object) -> int | None:
    # Primary: sklearn n_features_in_ if exposed
    impl = getattr(model, "_model_impl", None)
    sk_model = getattr(impl, "sklearn_model", None) if impl else None
    n = getattr(sk_model, "n_features_in_", None) if sk_model else None
    if n is not None:
        return int(n)

    # Fallback: infer from predict error message (no hardcoding)
    try:
        model.predict(np.zeros((1, 1), dtype=float))
        return None
    except Exception as e:
        msg = str(e)
        m = re.search(r"expecting\s+(\d+)\s+features", msg, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"expects\s+(\d+)\s+features", msg, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None


def expected_feature_count(variant: str = "champion") -> int | None:
    model, _uri, _run_id = get_model(variant)
    return _expected_feature_count_from_model(model)


def predict_one(
    features: list[float], variant: str = "champion"
) -> Tuple[str, str, str | None]:
    model, uri, run_id = get_model(variant)

    expected = _expected_feature_count_from_model(model)
    if expected is not None and len(features) != expected:
        raise ValueError(f"Model expects {expected} features, got {len(features)}")

    X = np.array([features], dtype=float)
    y = model.predict(X)

    pred = y[0] if hasattr(y, "__len__") else y
    return str(pred), uri, run_id
