from pydantic_ai import ApprovalRequired, RunContext

from app.ziza_chat.deps import ChatDeps
from app.ziza_chat.knowledge_base import clear_knowledge, describe_holdings


async def clear_knowledge_base(context: RunContext[ChatDeps]) -> str:
    """Delete every document in this session's knowledge base.

    Call this when the visitor asks to clear, reset, delete, or start over with
    their uploaded material. It removes every document, every image
    description, and the record of this conversation.

    Call it as soon as they ask. Do not ask them to confirm first and do not
    describe what would happen instead of calling it — confirming the deletion
    is handled for you, and nothing is removed until it is confirmed. Only call
    it when they have actually asked; never to tidy up on your own initiative.
    """
    session_id = context.deps.session_id
    if not context.tool_call_approved:
        documents = await describe_holdings(session_id)
        raise ApprovalRequired(
            metadata={
                "action": "clear_knowledge_base",
                "documents": documents,
                "summary": (
                    f"Delete {len(documents)} document(s) from this session, "
                    "along with their image descriptions and this conversation."
                ),
            }
        )
    chunks_deleted = await clear_knowledge(session_id)
    return (
        f"Deleted {chunks_deleted} passage(s). This session's knowledge base is "
        "now empty."
    )
