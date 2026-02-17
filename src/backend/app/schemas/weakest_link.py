from pydantic import BaseModel


class WeakestLinkResponse(BaseModel):
    prediction: str
    model_uri: str
    run_id: str | None = None
