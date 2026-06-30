import asyncio


class ConnectionManager:
    """Tracks open SSE connections, grouped by session so events only reach
    the clients that belong to the same test session."""

    def __init__(self) -> None:
        self._sessions: dict[str, set[asyncio.Queue[str | None]]] = {}

    def connect(self, session_id: str) -> asyncio.Queue[str | None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._sessions.setdefault(session_id, set()).add(queue)
        return queue

    def disconnect(self, session_id: str, queue: asyncio.Queue[str | None]) -> None:
        queues = self._sessions.get(session_id)
        if queues is None:
            return
        queues.discard(queue)
        if not queues:
            self._sessions.pop(session_id, None)

    async def broadcast(self, session_id: str, message: str) -> None:
        for queue in self._sessions.get(session_id, set()):
            queue.put_nowait(message)

    def close_all(self) -> None:
        for queues in self._sessions.values():
            for queue in queues:
                queue.put_nowait(None)
        self._sessions.clear()


connection_manager = ConnectionManager()
