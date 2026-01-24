from fastapi import APIRouter
from app.api.v1.endpoints.hello import router as hello_router

router = APIRouter()
router.include_router(hello_router, tags=["hello"])
