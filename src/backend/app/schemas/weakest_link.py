"""app.schemas.weakest_link

Pydantic schemas for weakest-link prediction endpoints.

These models define the response contract for the weakest-link model API.
Keep them stable to avoid breaking clients.
"""

from pydantic import BaseModel
from typing import Optional


class WeakestLinkResponse(BaseModel):
    """Response returned from a weakest-link prediction call"""

    prediction: str
    model_uri: str
    run_id: Optional[str] = None
