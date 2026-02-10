from pydantic import BaseModel

class WeakestLinkResponse(BaseModel):
    prediction: str
    model_uri: str
