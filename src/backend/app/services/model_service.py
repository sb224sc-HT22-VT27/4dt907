import logging
import os
import threading
import time
from typing import Dict, Tuple

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Dict[str, Tuple[object, str, float]] = {}  # cache_key -> (model, uri_used, timestamp)
_mlflow_inited = False
_CACHE_TTL = 3600  # Cache TTL in seconds (1 hour)


def _clean_uri(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().strip('"').strip("'")


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
        logger.error("MLFLOW_TRACKING_URI environment variable is not set")
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    if not _mlflow_inited:
        logger.info(f"Initializing MLflow with tracking URI: {uri}")
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
    logger.debug(f"Resolving alias '{alias}' for model '{model_name}'")
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.error(f"No versions found for model '{model_name}'")
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
            resolved_uri = f"models:/{model_name}/{chosen.version}"
            logger.info(f"Resolved '{alias}' to version {chosen.version}")
            return resolved_uri

        if a in {"dev", "latest"}:
            resolved_uri = f"models:/{model_name}/{latest.version}"
            logger.info(f"Resolved '{alias}' to version {latest.version}")
            return resolved_uri

        if a in {"backup"}:
            resolved_uri = f"models:/{model_name}/{second_latest.version}"
            logger.info(f"Resolved '{alias}' to version {second_latest.version}")
            return resolved_uri

        # Unknown alias -> safest default
        logger.warning(f"Unknown alias '{alias}', using latest version {latest.version}")
        return f"models:/{model_name}/{latest.version}"
    except Exception as e:
        logger.error(f"Error resolving alias '{alias}' for model '{model_name}': {e}")
        raise


def _load_model_with_alias_fallback(uri: str) -> Tuple[object, str]:
    """
    Try loading the provided URI. If it is an alias URI (models:/Name@alias) and
    the registry rejects alias lookup, resolve to a concrete version URI dynamically.
    Returns (model, uri_used).
    """
    logger.debug(f"Attempting to load model from URI: {uri}")
    try:
        model = mlflow.pyfunc.load_model(uri)
        logger.info(f"Successfully loaded model from URI: {uri}")
        return model, uri
    except RestException as e:
        # DagsHub/registry may reject get_model_version_by_alias with INVALID_PARAMETER_VALUE
        if _is_models_alias_uri(uri) and "INVALID_PARAMETER_VALUE" in str(e):
            logger.warning(f"Alias URI not supported, falling back to version resolution for: {uri}")
            model_name, alias = _parse_models_alias_uri(uri)
            resolved_uri = _resolve_alias_to_version_uri(model_name, alias)
            model = mlflow.pyfunc.load_model(resolved_uri)
            logger.info(f"Successfully loaded model from resolved URI: {resolved_uri}")
            return model, resolved_uri
        logger.error(f"Failed to load model from URI {uri}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model from URI {uri}: {e}")
        raise


def get_model(variant: str = "champion") -> Tuple[object, str]:
    _init_mlflow()
    
    logger.debug(f"Requesting model with variant: {variant}")

    direct_uri = _direct_uri_for_variant(variant)
    if direct_uri:
        cache_key = direct_uri
        now = time.time()
        
        with _lock:
            if cache_key in _cache:
                model, uri, timestamp = _cache[cache_key]
                # Check if cache is still valid
                if now - timestamp < _CACHE_TTL:
                    logger.debug(f"Cache hit for variant '{variant}' (age: {now - timestamp:.1f}s)")
                    return model, uri
                else:
                    logger.info(f"Cache expired for variant '{variant}' (age: {now - timestamp:.1f}s), reloading")
            else:
                logger.info(f"Cache miss for variant '{variant}', loading model")

            model, uri_used = _load_model_with_alias_fallback(direct_uri)
            _cache[cache_key] = (model, uri_used, now)
            return model, uri_used

    # Fallback: old behavior using registered model names (if still used somewhere)
    logger.debug(f"Using fallback model name lookup for variant: {variant}")
    model_name = _model_name_for_variant(variant)
    cache_key = model_name
    now = time.time()
    
    with _lock:
        if cache_key in _cache:
            model, uri, timestamp = _cache[cache_key]
            if now - timestamp < _CACHE_TTL:
                logger.debug(f"Cache hit for model '{model_name}' (age: {now - timestamp:.1f}s)")
                return model, uri
            else:
                logger.info(f"Cache expired for model '{model_name}' (age: {now - timestamp:.1f}s), reloading")
        else:
            logger.info(f"Cache miss for model '{model_name}', loading model")

        uri = _latest_source_uri(model_name)
        model = mlflow.pyfunc.load_model(uri)
        _cache[cache_key] = (model, uri, now)
        return model, uri


def _expected_feature_count_from_model(model: object) -> int | None:
    impl = getattr(model, "_model_impl", None)
    sk_model = getattr(impl, "sklearn_model", None) if impl else None
    n = getattr(sk_model, "n_features_in_", None) if sk_model else None
    return int(n) if n is not None else None


def expected_feature_count(variant: str = "latest") -> int | None:
    model, _uri = get_model(variant)
    return _expected_feature_count_from_model(model)


def predict_one(features: list[float], variant: str = "champion") -> Tuple[float, str]:
    logger.info(f"Prediction request: variant={variant}, n_features={len(features)}")
    
    model, uri = get_model(variant)

    expected = _expected_feature_count_from_model(model)
    if expected is not None and len(features) != expected:
        logger.warning(
            f"Feature count mismatch: expected {expected}, got {len(features)} for variant '{variant}'"
        )
        raise ValueError(f"Model expects {expected} features, got {len(features)}")

    X = np.array([features], dtype=float)
    y = model.predict(X)
    prediction = float(y[0]) if hasattr(y, "__len__") else float(y)
    
    logger.info(f"Prediction successful: variant={variant}, result={prediction:.4f}, model_uri={uri}")
    return prediction, uri


def clear_model_cache() -> None:
    global _cache
    logger.info("Clearing model cache")
    with _lock:
        _cache = {}
    logger.info("Model cache cleared successfully")
