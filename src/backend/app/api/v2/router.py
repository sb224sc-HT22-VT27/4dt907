from fastapi import APIRouter
from app.api.v2.endpoints.hello import router as hello_router

router = APIRouter()
router.include_router(hello_router, tags=["hello"])
