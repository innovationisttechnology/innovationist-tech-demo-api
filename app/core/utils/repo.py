from typing import Any, Type, TypeVar

from beanie import Document
from fastapi import HTTPException

DocumentType = TypeVar("DocumentType", bound=Document)


async def find_one_or_raise(
    model: Type[DocumentType],
    *args: Any,
    status_code: int = 404,
    detail: str = "Not found",
) -> DocumentType:
    doc = await model.find_one(*args)
    if doc is None:
        raise HTTPException(status_code=status_code, detail=detail)
    return doc
