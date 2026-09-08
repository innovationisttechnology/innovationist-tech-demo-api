from fastapi import Depends, HTTPException

from app.core.db.db_config import is_db_configured


def require_db() -> None:
    if not is_db_configured():
        raise HTTPException(
            status_code=503,
            detail="Config not found; This endpoint is unavailable",
        )


RequiresDatabase = Depends(require_db)
