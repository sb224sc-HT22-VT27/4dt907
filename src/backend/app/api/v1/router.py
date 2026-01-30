from fastapi import APIRouter
from app.api.v1.endpoints.predict import router as predict_router
from app.api.v1.endpoints.model_info import router as model_info_router

router = APIRouter()
router.include_router(predict_router, tags=["prediction"])

router.include_router(model_info_router, tags=["model-info"])
