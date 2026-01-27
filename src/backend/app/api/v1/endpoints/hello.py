from fastapi import APIRouter
from app.schemas.hello import HelloResponseV1
from app.services.hello_service import get_hello_message

router = APIRouter()

@router.get("/hello", response_model=HelloResponseV1)
def hello_v1():
    return HelloResponseV1(message=get_hello_message())
