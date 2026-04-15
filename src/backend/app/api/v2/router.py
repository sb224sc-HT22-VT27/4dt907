"""app.api.v2.router

API v2 router aggregator.

v2 exists to allow endpoint evolution without breaking older clients. Keep this
module declarative (imports + include_router calls) so version boundaries are clear.
"""

from fastapi import APIRouter

# Versioned router for all v2 endpoints.
router = APIRouter()


# Keep v2 available even if it currently mirrors v1, new endpoints land here first.
@router.get("/status")
def status():
    return {"version": "v2", "status": "ok"}
