"""app.schemas.hello

Example/placeholder schemas.

If this module is not referenced by any router, consider moving it to an
`examples/` folder or removing it to keep the production schema surface small. N.A
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HelloResponseV1(BaseModel):
    """Minimal example response model used for smoke testing or scaffolding."""

    message: str
    api_version: Literal["v1"] = "v1"


class HelloResponseV2(BaseModel):
    """Minimal example response model used for smoke testing or scaffolding."""

    message: str
    api_version: Literal["v2"] = "v2"
    server_time_utc: datetime = Field(..., description="UTC time on server")
