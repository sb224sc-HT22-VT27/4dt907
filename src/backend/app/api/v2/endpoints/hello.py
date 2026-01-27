from datetime import datetime, timezone

from fastapi import APIRouter
from app.schemas.hello import HelloResponseV2
from app.services.hello_service import get_hello_message

router = APIRouter()

@router.get("/hello", response_model=HelloResponseV2)
def hello_v2():
    return HelloResponseV2(
        message=get_hello_message(),
        server_time_utc=datetime.now(timezone.utc),
    )
