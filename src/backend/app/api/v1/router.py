"""app.api.v1.router

API v1 router aggregator.

This module collects all v1 endpoint routers into a single versioned router.
Keep it declarative (imports + include_router calls) so it’s easy to scan and
diff when endpoints are added/removed.
"""
from fastapi import APIRouter
from app.api.v1.endpoints.predict import router as predict_router
from app.api.v1.endpoints.model_info import router as model_info_router
from app.api.v1.endpoints.weakest_link import router as weakest_link_router
from app.api.v1.endpoints.squat import router as squat_router
from app.api.v1.endpoints.z_predictor import router as z_predictor_router

# Versioned router for all v1 endpoints.
router = APIRouter()

# Endpoint groups are tagged for nicer OpenAPI organization.
router.include_router(predict_router, tags=["prediction"])
router.include_router(model_info_router, tags=["model-info"])
router.include_router(weakest_link_router, tags=["weakest-link"])
router.include_router(z_predictor_router, tags=["z-predictor"])
router.include_router(squat_router, tags=["squat"])
