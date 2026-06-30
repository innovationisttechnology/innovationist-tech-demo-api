from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SyncFlagCreate(BaseModel):
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=500)
    enabled: bool = True


class SyncFlagUpdate(BaseModel):
    value: Optional[str] = Field(default=None, min_length=1, max_length=500)
    enabled: Optional[bool] = None


class SyncFlagRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    key: str
    value: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


class SyncEvent(BaseModel):
    """Payload broadcast over SSE.

    `type` is one of: flag.snapshot | flag.created | flag.updated | flag.deleted.
    """

    type: str
    key: Optional[str] = None
    flag: Optional[SyncFlagRead] = None
    flags: Optional[list[SyncFlagRead]] = None
