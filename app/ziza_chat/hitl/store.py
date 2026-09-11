from functools import lru_cache
from typing import Any, Sequence

from app.ziza_chat.hitl.models import DeferredCall, PausedRun


class MongoPausedRunStore:
    @staticmethod
    async def replace(
        session_id: str,
        intent: str,
        calls: Sequence[DeferredCall],
        messages: list[dict[str, Any]],
    ) -> PausedRun:
        await PausedRun.find(PausedRun.session_id == session_id).delete()
        paused = PausedRun(
            session_id=session_id,
            intent=intent,
            calls=list(calls),
            messages=messages,
        )
        await paused.insert()
        return paused

    @staticmethod
    async def find(session_id: str) -> PausedRun | None:
        return await PausedRun.find_one(PausedRun.session_id == session_id)

    @staticmethod
    async def save(paused: PausedRun) -> PausedRun:
        await paused.save()
        return paused

    @staticmethod
    async def discard(session_id: str) -> None:
        await PausedRun.find(PausedRun.session_id == session_id).delete()


@lru_cache
def get_paused_run_store() -> MongoPausedRunStore:
    return MongoPausedRunStore()
