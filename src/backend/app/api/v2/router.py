from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
def status():
    return {"version": "v2", "status": "ok"}
