from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HelloResponseV1(BaseModel):
    message: str
    api_version: Literal["v1"] = "v1"


class HelloResponseV2(BaseModel):
    message: str
    api_version: Literal["v2"] = "v2"
    server_time_utc: datetime = Field(..., description="UTC time on server")
