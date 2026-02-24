"""app.api.health

Health/liveness endpoint(s) for monitoring and deployment checks.

This router is intentionally tiny: it provides a fast, dependency-free way for
container platforms, and uptime monitors to confirm the service
is running.
"""
from fastapi import APIRouter

# Router dedicated to lightweight operational endpoints.
router = APIRouter()

# Routers 
@router.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
