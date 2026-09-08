import asyncio
import logging
from pathlib import PurePosixPath
from typing import AsyncIterator, Sequence
from urllib.parse import urlparse

from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits

from app.core.db.db_config import is_db_configured
from app.ziza_chat.agents.chat import get_chat_agent
from app.ziza_chat.agents.classifier import (
    build_classifier_prompt,
    get_classifier_agent,
)
from app.ziza_chat.agents.outputs import ClassifyResult, Scope
from app.ziza_chat.agents.page_summary import summarize_page
from app.ziza_chat.agents.vision import describe_image
from app.ziza_chat.caption_cache import (
    carries_no_information,
    get_cached_caption,
    hash_image,
    store_caption,
)
from app.ziza_chat.config import ziza_settings
from app.ziza_chat.deps import ChatDeps
from app.ziza_chat.document_loaders import (
    ExtractedImage,
    UnsupportedDocumentError,
    load_document,
)
from app.ziza_chat.document_loaders.web import fetch_page, html_to_text
from app.ziza_chat.history_store.repair import last_visitor_message
from app.ziza_chat.history_store.service import (
    append_turn,
    load_history,
    record_refusal,
)
from app.ziza_chat.history_store.store import deserialize_turns
from app.ziza_chat.hitl.models import PendingApproval
from app.ziza_chat.hitl.service import (
    claim_pending,
    decline_abandoned,
    record_pending,
)
from app.ziza_chat.schemas import (
    ApprovalDecisionRequest,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    PendingApprovalRead,
)
from app.ziza_chat.tool_logging import log_tool_events
from app.ziza_chat.vector_store.store import (
    MIN_SCORE,
    DocumentSection,
    get_vector_store,
)

logger = logging.getLogger(__name__)


class SessionLimitError(Exception):
    pass


async def assert_capacity(session_id: str, document: str) -> None:
    """Refuse a new document once the session is full.

    Checked before extraction, fetching, and captioning so a rejected upload
    costs nothing. Re-adding a document the session already holds is allowed —
    it occupies a slot it has already paid for.
    """
    existing = await get_vector_store().list_documents(session_id)
    if document in existing:
        return
    limit = ziza_settings.max_documents_per_session
    if len(existing) >= limit:
        raise SessionLimitError(
            f"This session already holds {len(existing)} of {limit} documents. "
            "Remove the existing ones to add more."
        )


async def classify(
    message: str, previous_message: str | None = None
) -> ClassifyResult:
    result = await get_classifier_agent().run(
        build_classifier_prompt(message, previous_message)
    )
    classification = result.output
    logger.info(
        'classified "%s" -> scope=%s intents=%s needs_rag=%s rag_query=%r ambiguous=%s',
        message if len(message) <= 80 else f"{message[:77]}...",
        classification.scope.value,
        " + ".join(intent.value for intent in classification.intents),
        classification.needs_rag,
        classification.rag_query,
        classification.rag_ambiguous,
    )
    return classification


# Deliberately the same threshold the agent answers from: if nothing clears it,
# the agent would have no passage to ground an answer in, so letting the message
# through only invites it to answer from memory. Measured on unrelated queries
# against a small knowledge base, cosine lands at 0.38-0.47 — a lower gate is
# below the noise floor and passes everything.
GATE_MIN_SCORE = MIN_SCORE


def refusal_for(documents: list[str]) -> str:
    elsewhere = (
        "For general questions, the full Ziza assistant at "
        f"{ziza_settings.general_assistant_url} can help."
    )
    if not documents:
        return (
            "I can only answer from documents added to this session, and there "
            "aren't any yet. Upload a file or paste a link and I'll answer from "
            f"it.\n\n{elsewhere}"
        )
    listed = ", ".join(documents[:5])
    if len(documents) > 5:
        listed += f", and {len(documents) - 5} more"
    return (
        "I can only answer from the documents in this session, and none of them "
        f"cover that. This session holds: {listed}.\n\n{elsewhere}"
    )


async def resolve_scope(session_id: str, classification: ClassifyResult) -> str | None:
    """Return a refusal when the message falls outside the demo, else None.

    Enforced here rather than in the chat agent's instructions: a prompt is a
    default the model may talk itself out of, while this cannot be argued with,
    reframed as fiction, or overridden by text inside an uploaded document. The
    scope decision comes from the classifier, which only ever sees the
    visitor's message — never retrieved content — so an injected instruction in
    a document cannot reach it.
    """
    store = get_vector_store() if is_db_configured() else None
    documents = await store.list_documents(session_id) if store else []

    if classification.scope is Scope.OUT_OF_SCOPE:
        logger.info("refused out-of-scope message for %s", session_id)
        return refusal_for(documents)

    if classification.scope is Scope.ASSISTANT:
        return None

    if store is None:
        return refusal_for([])

    query = classification.rag_query or ""
    if not query:
        return None
    matches = await store.search(session_id, query, limit=1, min_score=GATE_MIN_SCORE)
    if not matches:
        logger.info(
            "refused %r for %s — nothing in the knowledge base matched",
            query,
            session_id,
        )
        return refusal_for(documents)
    return None


async def build_chat_deps(session_id: str, intent: str) -> ChatDeps:
    vector_store = get_vector_store() if is_db_configured() else None
    documents = await vector_store.list_documents(session_id) if vector_store else []
    return ChatDeps(
        session_id=session_id,
        intent=intent,
        vector_store=vector_store,
        documents=documents,
    )


CHAT_USAGE_LIMITS = UsageLimits(request_limit=5, tool_calls_limit=6)


APPROVAL_FALLBACK_PROMPT = (
    "That needs your confirmation before I can do it."
)

APPROVAL_UNAVAILABLE = (
    "That action needs confirmation, which isn't available in this session."
)


def describe_pending(pending: PendingApproval) -> str:
    summary = pending.details.get("summary")
    prompt = str(summary) if summary else APPROVAL_FALLBACK_PROMPT
    return f"{prompt} Confirm to continue, or say no to leave things as they are."


def to_pending_read(pending: PendingApproval) -> PendingApprovalRead:
    return PendingApprovalRead(
        tool_call_id=pending.tool_call_id,
        tool_name=pending.tool_name,
        details=pending.details,
    )


async def build_response(
    session_id: str,
    intent: str,
    output: str | DeferredToolRequests,
    paused_messages: Sequence[ModelMessage] = (),
) -> ChatResponse:
    """Turn a finished run into a reply, or park a paused one.

    A paused run's messages hold a tool call with no result, which Anthropic
    rejects on replay. They go on the pending record instead of the transcript,
    so the stored history stays valid for any other continuation.
    """
    if not isinstance(output, DeferredToolRequests):
        return ChatResponse(
            session_id=session_id, response=output, intent=intent
        )
    pending = await record_pending(session_id, output, intent, paused_messages)
    if pending is None:
        return ChatResponse(
            session_id=session_id, response=APPROVAL_UNAVAILABLE, intent=intent
        )
    return ChatResponse(
        session_id=session_id,
        response=describe_pending(pending),
        intent=intent,
        pending_approval=to_pending_read(pending),
    )


async def chat(request: ChatRequest) -> ChatResponse:
    await decline_abandoned(request.session_id)
    history = await load_history(request.session_id)
    classification = await classify(
        request.message, last_visitor_message(history)
    )
    intent = classification.primary.value
    refusal = await resolve_scope(request.session_id, classification)
    if refusal is not None:
        await record_refusal(request.session_id, request.message, refusal)
        return ChatResponse(
            session_id=request.session_id, response=refusal, intent=intent
        )
    deps = await build_chat_deps(request.session_id, intent)
    result = await get_chat_agent().run(
        request.message,
        deps=deps,
        message_history=history,
        event_stream_handler=log_tool_events,
        usage_limits=CHAT_USAGE_LIMITS,
    )
    if isinstance(result.output, DeferredToolRequests):
        return await build_response(
            request.session_id, intent, result.output, result.new_messages()
        )
    await append_turn(request.session_id, result.new_messages())
    return await build_response(request.session_id, intent, result.output)


async def resolve_approval(request: ApprovalDecisionRequest) -> ChatResponse:
    pending = await claim_pending(request.session_id, request.tool_call_id)
    decision: ToolApproved | ToolDenied = (
        ToolApproved()
        if request.approved
        else ToolDenied("The visitor declined this action, so nothing was changed.")
    )
    results = DeferredToolResults()
    results.approvals[request.tool_call_id] = decision
    paused_messages = deserialize_turns([pending.messages])
    deps = await build_chat_deps(request.session_id, pending.intent)
    result = await get_chat_agent().run(
        deps=deps,
        message_history=await load_history(request.session_id) + paused_messages,
        deferred_tool_results=results,
        event_stream_handler=log_tool_events,
        usage_limits=CHAT_USAGE_LIMITS,
    )
    resolved_turn = paused_messages + list(result.new_messages())
    if isinstance(result.output, DeferredToolRequests):
        return await build_response(
            request.session_id, pending.intent, result.output, resolved_turn
        )
    await append_turn(request.session_id, resolved_turn)
    return await build_response(request.session_id, pending.intent, result.output)


async def stream_chat(request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
    await decline_abandoned(request.session_id)
    history = await load_history(request.session_id)
    classification = await classify(
        request.message, last_visitor_message(history)
    )
    intent = classification.primary.value
    refusal = await resolve_scope(request.session_id, classification)
    if refusal is not None:
        await record_refusal(request.session_id, request.message, refusal)
        yield ChatStreamEvent(type="chat.chunk", chunk=refusal)
        return
    deps = await build_chat_deps(request.session_id, intent)
    async with get_chat_agent().run_stream(
        request.message,
        deps=deps,
        message_history=history,
        event_stream_handler=log_tool_events,
        usage_limits=CHAT_USAGE_LIMITS,
    ) as result:
        deferred: DeferredToolRequests | None = None
        streamed = ""
        async for output in result.stream_output():
            if isinstance(output, DeferredToolRequests):
                deferred = output
                continue
            if output.startswith(streamed) and len(output) > len(streamed):
                yield ChatStreamEvent(
                    type="chat.chunk", chunk=output[len(streamed) :]
                )
                streamed = output
        paused_messages = list(result.new_messages()) if deferred else []
        if deferred is None:
            await append_turn(request.session_id, result.new_messages())
    if deferred is None:
        return
    pending = await record_pending(
        request.session_id, deferred, intent, paused_messages
    )
    if pending is None:
        yield ChatStreamEvent(type="chat.chunk", chunk=APPROVAL_UNAVAILABLE)
        return
    yield ChatStreamEvent(type="chat.chunk", chunk=describe_pending(pending))
    yield ChatStreamEvent(
        type="chat.approval_required", pending_approval=to_pending_read(pending)
    )


async def ingest_knowledge(request: KnowledgeIngestRequest) -> KnowledgeIngestResponse:
    await assert_capacity(request.session_id, request.source)
    store = get_vector_store()
    chunk_count = await store.add_document(
        session_id=request.session_id,
        text=request.text,
        source=request.source,
    )
    searchable = await store.wait_until_searchable(request.session_id)
    return KnowledgeIngestResponse(
        session_id=request.session_id,
        source=request.source,
        chunks_ingested=chunk_count,
        searchable=searchable,
        documents_used=await store.count_documents(request.session_id),
        documents_allowed=ziza_settings.max_documents_per_session,
    )


def format_source(filename: str, locator: str | None, is_image: bool = False) -> str:
    parts = [part for part in (locator, "image" if is_image else None) if part]
    return f"{filename} ({', '.join(parts)})" if parts else filename


async def caption_images(
    images: list[ExtractedImage], filename: str, session_id: str
) -> tuple[list[DocumentSection], int]:
    """Describe each image and return it as a searchable text section, plus a
    count of images that could not be described.

    One failed image must not fail the whole upload, so failures are logged and
    skipped — but the count is reported back so a systematically broken vision
    model doesn't look like a successful upload of an image-only document.
    """
    semaphore = asyncio.Semaphore(ziza_settings.max_concurrent_captions)

    async def describe(image: ExtractedImage) -> str:
        image_hash = hash_image(image.data)
        cached = await get_cached_caption(session_id, image_hash)
        if cached is not None:
            logger.info("caption cache hit for %s", image_hash[:12])
            return cached
        async with semaphore:
            description = await describe_image(image.data, image.media_type)
        caption = description.to_search_text()
        await store_caption(session_id, image_hash, caption)
        return caption

    describable = [
        image for image in images if not carries_no_information(image.data)
    ]
    skipped = len(images) - len(describable)
    if skipped:
        logger.info("skipped %d image(s) in %s with nothing to index", skipped, filename)

    results = await asyncio.gather(
        *(describe(image) for image in describable), return_exceptions=True
    )
    images = describable

    sections: list[DocumentSection] = []
    failures = 0
    for image, result in zip(images, results, strict=True):
        source = format_source(filename, image.locator, is_image=True)
        if isinstance(result, BaseException):
            logger.warning("Could not describe %s: %s", source, result)
            failures += 1
            continue
        logger.info("indexed %s -> %s", source, result[:100].replace("\n", " "))
        sections.append(DocumentSection(text=result, source=source))
    return sections, failures


async def ingest_file(
    session_id: str,
    filename: str,
    content_type: str | None,
    data: bytes,
    document_name: str | None = None,
) -> KnowledgeIngestResponse:
    document_name = document_name or filename
    await assert_capacity(session_id, document_name)
    document = load_document(data, filename, content_type)

    sections = [
        DocumentSection(
            text=extracted.text,
            source=format_source(filename, extracted.locator),
            document=document_name,
        )
        for extracted in document.texts
    ]

    images = document.images
    if len(images) > ziza_settings.max_images_per_document:
        logger.warning(
            "%s has %d images; describing only the first %d",
            filename,
            len(images),
            ziza_settings.max_images_per_document,
        )
        images = images[: ziza_settings.max_images_per_document]
    image_sections, images_failed = await caption_images(images, filename, session_id)
    sections.extend(
        DocumentSection(
            text=section.text, source=section.source, document=document_name
        )
        for section in image_sections
    )

    store = get_vector_store()
    chunk_count = await store.add_sections(session_id, sections)
    searchable = await store.wait_until_searchable(session_id)
    return KnowledgeIngestResponse(
        session_id=session_id,
        source=filename,
        chunks_ingested=chunk_count,
        images_described=len(image_sections),
        images_failed=images_failed,
        searchable=searchable,
        documents_used=await store.count_documents(session_id),
        documents_allowed=ziza_settings.max_documents_per_session,
    )


async def ingest_url(session_id: str, url: str) -> KnowledgeIngestResponse:
    # The submitted URL is the document's identity, not the URL it redirects
    # to, so re-submitting the same link never consumes a second slot.
    await assert_capacity(session_id, url)
    page = await fetch_page(url)

    if "html" not in page.content_type:
        # A URL pointing at a PDF, image, or plain text is the file pipeline's
        # job — content sniffing and per-type extraction already handle it.
        filename = PurePosixPath(urlparse(page.url).path).name or page.url
        return await ingest_file(
            session_id=session_id,
            filename=filename,
            content_type=page.content_type,
            data=page.body,
            document_name=url,
        )

    text = html_to_text(page.body)
    if not text.strip():
        raise UnsupportedDocumentError(f"{page.url} has no readable text.")

    source = f"{page.title} ({page.url})" if page.title else page.url
    sections = [DocumentSection(text=text, source=source, document=url)]

    summarised = 0
    try:
        summary = await summarize_page(page.title, page.url, text)
        sections.append(
            DocumentSection(
                text=summary.to_search_text(),
                source=f"{source} [summary]",
                document=url,
            )
        )
        summarised = 1
        logger.info("summarised %s -> %s", page.url, summary.summary[:100])
    except Exception as failure:
        # The page text is already indexed; losing the overview is not fatal.
        logger.warning("Could not summarise %s: %s", page.url, failure)

    store = get_vector_store()
    chunk_count = await store.add_sections(session_id, sections)
    searchable = await store.wait_until_searchable(session_id)
    return KnowledgeIngestResponse(
        session_id=session_id,
        source=source,
        chunks_ingested=chunk_count,
        pages_summarised=summarised,
        searchable=searchable,
        documents_used=await store.count_documents(session_id),
        documents_allowed=ziza_settings.max_documents_per_session,
    )



