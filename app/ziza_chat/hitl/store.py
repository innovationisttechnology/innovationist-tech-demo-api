from functools import lru_cache
from typing import Any

from app.ziza_chat.hitl.models import PendingApproval


class MongoApprovalStore:
    @staticmethod
    async def replace(
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        intent: str,
        details: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> PendingApproval:
        await PendingApproval.find(
            PendingApproval.session_id == session_id
        ).delete()
        pending = PendingApproval(
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            intent=intent,
            details=details,
            messages=messages,
        )
        await pending.insert()
        return pending

    @staticmethod
    async def find(session_id: str) -> PendingApproval | None:
        return await PendingApproval.find_one(
            PendingApproval.session_id == session_id
        )

    @staticmethod
    async def discard(session_id: str) -> None:
        await PendingApproval.find(
            PendingApproval.session_id == session_id
        ).delete()


@lru_cache
def get_approval_store() -> MongoApprovalStore:
    return MongoApprovalStore()
