from typing import Any, Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int
    details: Optional[Any] = None
