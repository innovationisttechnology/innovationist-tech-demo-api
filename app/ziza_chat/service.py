import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import AsyncIterator, Sequence
from urllib.parse import urlparse

import httpx
from pydantic_ai import DeferredToolRequests
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
from app.ziza_chat.document_loaders.web import (
    UnsafeUrlError,
    fetch_page,
    html_to_text,
)
from app.ziza_chat.history_store.repair import last_visitor_message
from app.ziza_chat.history_store.service import (
    append_turn,
    load_history,
    record_refusal,
)
from app.ziza_chat.history_store.store import deserialize_turns
from app.ziza_chat.hitl.models import (
    CallResolution,
    DeferredCall,
    DeferredKind,
    PausedRun,
)
from app.ziza_chat.hitl.service import (
    decline_abandoned,
    discard_paused,
    record_pause,
    require_call,
    resolve_call,
    to_results,
    unresolved_calls,
)
from app.ziza_chat.page_links import load_page_links
from app.ziza_chat.schemas import (
    ApprovalDecisionRequest,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    KnowledgeIngestResponse,
    LinkSelectionRequest,
    PendingCallRead,
    Suggestion,
    SuggestionKind,
)
from app.ziza_chat.tool_logging import log_step, log_tool_events
from app.ziza_chat.vector_store.store import (
    MIN_SCORE,
    DocumentSection,
    get_vector_store,
)

logger = logging.getLogger(__name__)


class SessionLimitError(Exception):
    pass


class UnofferedLinkError(Exception):
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
    message: str, previous_message: str | None = None, session_id: str = "-"
) -> ClassifyResult:
    log_step(
        session_id,
        "classify",
        f"{ziza_settings.ziza_classifier_model} on {message!r}"
        + (f" (after {previous_message!r})" if previous_message else ""),
    )
    result = await get_classifier_agent().run(
        build_classifier_prompt(message, previous_message)
    )
    classification = result.output
    log_step(
        session_id,
        "←classify",
        f"scope={classification.scope.value} "
        f"intents={' + '.join(intent.value for intent in classification.intents)} "
        f"needs_rag={classification.needs_rag} "
        f"rag_query={classification.rag_query!r} "
        f"ambiguous={classification.rag_ambiguous}",
    )
    return classification


# Deliberately the same threshold the agent answers from: if nothing clears it,
# the agent would have no passage to ground an answer in, so letting the message
# through only invites it to answer from memory. Measured on unrelated queries
# against a small knowledge base, cosine lands at 0.38-0.47 — a lower gate is
# below the noise floor and passes everything.
GATE_MIN_SCORE = MIN_SCORE

# Only the unambiguous form. A bare domain cannot be told from a filename or a
# library by pattern — "report.md", "node.js", "index.html" all match any rule
# loose enough to catch "xyz.com" — so that judgement is the classifier's and
# this is the deterministic floor under it.
EXPLICIT_URL = re.compile(r"https?://\S+")

URL_SHAPE = re.compile(r"^[^\s/@]+\.[a-z]{2,}(?:[/?#]\S*)?$", re.IGNORECASE)


def normalize_url(mentioned: str | None) -> str | None:
    """A fetchable URL from what the classifier reported, or None.

    The scheme is added here rather than asked for in the prompt, and the
    shape is checked here too: a model that returns prose, a bare word, or a
    repaired guess must not reach the fetcher.
    """
    if not mentioned:
        return None
    candidate = mentioned.strip().strip(".,;:!?\"\'()[]<>")
    if EXPLICIT_URL.fullmatch(candidate):
        return candidate
    if URL_SHAPE.fullmatch(candidate):
        return f"https://{candidate}"
    return None


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


async def resolve_scope(
    session_id: str, classification: ClassifyResult, message: str
) -> str | None:
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
        log_step(session_id, "gate", "refused — out of scope")
        return refusal_for(documents)

    if classification.scope is Scope.ASSISTANT:
        log_step(session_id, "gate", "allowed — about the demo itself")
        return None

    if store is None:
        log_step(session_id, "gate", "refused — no knowledge base in this session")
        return refusal_for([])

    if normalize_url(classification.mentioned_url) or EXPLICIT_URL.search(message):
        # The gate below would refuse the one thing a link is for: a page
        # cannot match a knowledge base it has not been added to yet.
        # Out-of-scope messages are already refused above, so this widens what
        # may be ingested, never what may be answered from memory.
        log_step(session_id, "gate", "allowed — the message points at a web page")
        return None

    query = classification.rag_query or ""
    if not query:
        log_step(session_id, "gate", "allowed — no query to judge relevance on")
        return None

    if classification.rag_ambiguous and documents:
        # "this file", "the document", "it" — the classifier hands these back
        # as a word like "file", which cannot be expected to match an
        # embedding however relevant the session's material is. Refusing on
        # that miss tells a visitor who just uploaded something that their own
        # document does not cover it. The agent searches again with a better
        # query, and says so plainly when nothing comes back.
        log_step(
            session_id,
            "gate",
            f"allowed — {query!r} too vague to gate on, session holds "
            f"{len(documents)} document(s)",
        )
        return None

    log_step(
        session_id, "gate", f"searching {query!r} at min_score={GATE_MIN_SCORE}"
    )
    matches = await store.search(session_id, query, limit=1, min_score=GATE_MIN_SCORE)
    if not matches:
        log_step(
            session_id,
            "gate",
            f"refused — nothing matched {query!r} in {len(documents)} document(s)",
        )
        return refusal_for(documents)
    log_step(
        session_id, "gate", f"allowed — best match scored {matches[0].score:.2f}"
    )
    return None


async def documents_held(session_id: str) -> list[str]:
    if not is_db_configured():
        return []
    return await get_vector_store().list_documents(session_id)


async def build_chat_deps(
    session_id: str, intent: str, mentioned_url: str | None = None
) -> ChatDeps:
    vector_store = get_vector_store() if is_db_configured() else None
    documents = await vector_store.list_documents(session_id) if vector_store else []
    return ChatDeps(
        session_id=session_id,
        intent=intent,
        vector_store=vector_store,
        documents=documents,
        mentioned_url=None if mentioned_url in documents else mentioned_url,
    )


CHAT_USAGE_LIMITS = UsageLimits(request_limit=5, tool_calls_limit=6)

# A session caps at max_documents_per_session and each page costs a summary
# call plus embeddings, so a handful of the most promising pages is the offer —
# not everything the site links to.
MAX_SUGGESTED_PAGES = 4


def retrieval_was_weak(deps: ChatDeps) -> bool:
    """True when the agent searched and came back with nothing.

    Read from what the search tool recorded rather than inferred from the
    answer text, so "the documents did not cover this" is a fact about
    retrieval and not a guess about prose.
    """
    return bool(deps.searches) and all(
        search.matches == 0 for search in deps.searches
    )


async def offer_more_pages(
    session_id: str, documents: Sequence[str], because: str
) -> list[Suggestion]:
    suggestions = await suggest_pages(session_id, documents)
    if suggestions:
        log_step(
            session_id,
            "suggest",
            f"{len(suggestions)} page(s) to explore — {because}",
        )
    return suggestions


async def suggest_pages(session_id: str, documents: Sequence[str]) -> list[Suggestion]:
    """Pages this session's own material links to but has not indexed.

    Only offered when retrieval has already failed, which is what keeps this
    from nagging: a visitor whose questions are being answered never sees it.
    """
    if not is_db_configured():
        return []
    held = set(documents)
    slots_left = ziza_settings.max_documents_per_session - len(held)
    if slots_left <= 0:
        return []
    suggestions: list[Suggestion] = []
    for record in await load_page_links(session_id):
        for link in record.links:
            if link.url in held:
                continue
            suggestions.append(
                Suggestion(
                    kind=SuggestionKind.ADD_PAGE,
                    label=link.text,
                    message=f"Add {link.url} to my knowledge base",
                    url=link.url,
                )
            )
    return suggestions[: min(MAX_SUGGESTED_PAGES, slots_left)]


APPROVAL_FALLBACK_PROMPT = "That needs your confirmation before I can do it."

CALL_FALLBACK_PROMPT = "That needs something from you before I can continue."

DEFERRAL_UNAVAILABLE = (
    "That action needs a follow-up step, which isn't available in this session."
)

APPROVAL_CLOSING = "Confirm to continue, or say no to leave things as they are."

CALL_CLOSING = "Tell me how you'd like to proceed and I'll carry on."


def describe_call(call: DeferredCall) -> str:
    summary = call.metadata.get("summary")
    if summary:
        return str(summary)
    return (
        APPROVAL_FALLBACK_PROMPT
        if call.kind is DeferredKind.APPROVAL
        else CALL_FALLBACK_PROMPT
    )


def describe_pause(pending: Sequence[DeferredCall]) -> str:
    closing = (
        APPROVAL_CLOSING
        if all(call.kind is DeferredKind.APPROVAL for call in pending)
        else CALL_CLOSING
    )
    return " ".join([*(describe_call(call) for call in pending), closing])


def to_pending_read(call: DeferredCall) -> PendingCallRead:
    return PendingCallRead(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        kind=call.kind,
        details=call.metadata,
    )


async def handle_pause(
    session_id: str,
    intent: str,
    requests: DeferredToolRequests,
    paused_messages: Sequence[ModelMessage],
) -> ChatResponse:
    """Park a run that stopped holding unanswered tool calls.

    A paused run's messages hold tool calls with no results, which Anthropic
    rejects on replay. They go on the paused-run record instead of the
    transcript, so the stored history stays valid for any other continuation.
    """
    paused = await record_pause(session_id, requests, intent, paused_messages)
    if paused is None:
        log_step(session_id, "paused", "dropped — no database to park the run in")
        return ChatResponse(
            session_id=session_id, response=DEFERRAL_UNAVAILABLE, intent=intent
        )
    pending = unresolved_calls(paused)
    log_step(
        session_id,
        "paused",
        "awaiting "
        + ", ".join(
            f"{call.kind.value} {call.tool_name}[{call.tool_call_id}]"
            for call in pending
        ),
    )
    return ChatResponse(
        session_id=session_id,
        response=describe_pause(pending),
        intent=intent,
        pending_calls=[to_pending_read(call) for call in pending],
    )


async def handle_output(
    session_id: str,
    intent: str,
    output: str | DeferredToolRequests,
    turn_messages: Sequence[ModelMessage],
) -> ChatResponse:
    if isinstance(output, DeferredToolRequests):
        return await handle_pause(session_id, intent, output, turn_messages)
    await append_turn(session_id, turn_messages)
    log_step(
        session_id,
        "answered",
        f"{len(output)} chars, stored {len(turn_messages)} message(s)",
    )
    return ChatResponse(session_id=session_id, response=output, intent=intent)


@dataclass
class RefusedTurn:
    intent: str
    refusal: str
    documents: list[str]


@dataclass
class ReadyTurn:
    intent: str
    history: list[ModelMessage]
    deps: ChatDeps


async def prepare_turn(request: ChatRequest) -> RefusedTurn | ReadyTurn:
    """Everything both entry points do before the agent runs.

    Shared so the scope gate, the abandoned-pause decline, and the classifier
    hint cannot drift between the buffered and streaming paths.
    """
    if await decline_abandoned(request.session_id):
        log_step(request.session_id, "abandoned", "declined the pause left open")
    history = await load_history(request.session_id)
    log_step(
        request.session_id, "history", f"replaying {len(history)} message(s)"
    )
    classification = await classify(
        request.message, last_visitor_message(history), request.session_id
    )
    intent = classification.primary.value
    refusal = await resolve_scope(request.session_id, classification, request.message)
    if refusal is not None:
        await record_refusal(request.session_id, request.message, refusal)
        log_step(request.session_id, "refused", refusal)
        return RefusedTurn(
            intent=intent,
            refusal=refusal,
            documents=await documents_held(request.session_id),
        )
    deps = await build_chat_deps(
        request.session_id, intent, normalize_url(classification.mentioned_url)
    )
    log_step(
        request.session_id,
        "deps",
        f"intent={intent} documents={deps.documents or 'none'} "
        f"url={deps.mentioned_url or 'none'}",
    )
    return ReadyTurn(intent=intent, history=history, deps=deps)


async def chat(request: ChatRequest) -> ChatResponse:
    log_step(request.session_id, "chat", f"buffered {request.message!r}")
    prepared = await prepare_turn(request)
    if isinstance(prepared, RefusedTurn):
        return ChatResponse(
            session_id=request.session_id,
            response=prepared.refusal,
            intent=prepared.intent,
            suggestions=await offer_more_pages(
                request.session_id, prepared.documents, "the gate refused"
            ),
        )
    log_step(request.session_id, "agent", f"running {ziza_settings.ziza_chat_model}")
    result = await get_chat_agent().run(
        request.message,
        deps=prepared.deps,
        message_history=prepared.history,
        event_stream_handler=log_tool_events,
        usage_limits=CHAT_USAGE_LIMITS,
    )
    answered = await handle_output(
        request.session_id, prepared.intent, result.output, result.new_messages()
    )
    # A paused run is already asking about these same pages; adding the passive
    # offer on top prompts twice for one decision.
    if not answered.pending_calls and retrieval_was_weak(prepared.deps):
        answered.suggestions = await offer_more_pages(
            request.session_id, prepared.deps.documents, "retrieval came back empty"
        )
    return answered


async def stream_chat(request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
    log_step(request.session_id, "chat", f"streaming {request.message!r}")
    prepared = await prepare_turn(request)
    if isinstance(prepared, RefusedTurn):
        yield ChatStreamEvent(type="chat.chunk", chunk=prepared.refusal)
        refused_suggestions = await offer_more_pages(
            request.session_id, prepared.documents, "the gate refused"
        )
        if refused_suggestions:
            yield ChatStreamEvent(
                type="chat.suggestions", suggestions=refused_suggestions
            )
        return
    log_step(request.session_id, "agent", f"running {ziza_settings.ziza_chat_model}")
    async with get_chat_agent().run_stream(
        request.message,
        deps=prepared.deps,
        message_history=prepared.history,
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
        # Must stay inside the context manager and after the loop: the result
        # holds a list the run mutates, and the final message only lands once
        # the stream has been consumed.
        paused_messages = list(result.new_messages()) if deferred else []
        if deferred is None:
            await append_turn(request.session_id, result.new_messages())
            log_step(
                request.session_id, "answered", f"streamed {len(streamed)} chars"
            )
    if deferred is None:
        if retrieval_was_weak(prepared.deps):
            weak_suggestions = await offer_more_pages(
                request.session_id,
                prepared.deps.documents,
                "retrieval came back empty",
            )
            if weak_suggestions:
                yield ChatStreamEvent(
                    type="chat.suggestions", suggestions=weak_suggestions
                )
        return
    parked = await handle_pause(
        request.session_id, prepared.intent, deferred, paused_messages
    )
    yield ChatStreamEvent(type="chat.chunk", chunk=parked.response)
    for pending in parked.pending_calls:
        yield ChatStreamEvent(
            type=(
                "chat.approval_required"
                if pending.kind is DeferredKind.APPROVAL
                else "chat.input_required"
            ),
            pending_call=pending,
        )


async def resume_if_ready(paused: PausedRun) -> ChatResponse:
    """Replay the paused run once every deferred call has an answer.

    Until then the run is not replayable — a tool call with no result is
    rejected on replay — so the visitor is told what is still outstanding and
    nothing is sent to the model.
    """
    still_pending = unresolved_calls(paused)
    if still_pending:
        log_step(
            paused.session_id,
            "waiting",
            f"{len(paused.calls) - len(still_pending)} of {len(paused.calls)} "
            "answered — not replaying yet",
        )
        return ChatResponse(
            session_id=paused.session_id,
            response=describe_pause(still_pending),
            intent=paused.intent,
            pending_calls=[to_pending_read(call) for call in still_pending],
        )

    results = to_results(paused)
    paused_messages = deserialize_turns([paused.messages])
    log_step(
        paused.session_id,
        "resume",
        f"replaying {len(paused_messages)} paused message(s) with "
        f"{len(results.approvals)} approval(s) and {len(results.calls)} result(s)",
    )
    deps = await build_chat_deps(paused.session_id, paused.intent)
    result = await get_chat_agent().run(
        deps=deps,
        message_history=await load_history(paused.session_id) + paused_messages,
        deferred_tool_results=results,
        event_stream_handler=log_tool_events,
        usage_limits=CHAT_USAGE_LIMITS,
    )
    # Before the tail, not after: a run that pauses again records a fresh
    # paused run in its place.
    await discard_paused(paused.session_id)
    return await handle_output(
        paused.session_id,
        paused.intent,
        result.output,
        paused_messages + list(result.new_messages()),
    )


async def resolve_approval(request: ApprovalDecisionRequest) -> ChatResponse:
    log_step(
        request.session_id,
        "decision",
        f"{request.tool_call_id} approved={request.approved}",
    )
    paused = await resolve_call(
        request.session_id,
        request.tool_call_id,
        DeferredKind.APPROVAL,
        CallResolution(approved=request.approved),
    )
    return await resume_if_ready(paused)


@dataclass
class FailedLink:
    url: str
    reason: str


@dataclass
class LinkIngestOutcome:
    indexed: list[KnowledgeIngestResponse]
    failed: list[FailedLink]


async def ingest_each(
    session_id: str, urls: Sequence[str]
) -> LinkIngestOutcome:
    """Index a batch of pages, keeping the ones that work.

    One unreachable link must not lose the rest, so each failure is recorded
    against its own URL and reported back rather than raised.
    """
    outcome = LinkIngestOutcome(indexed=[], failed=[])
    log_step(session_id, "ingest", f"{len(urls)} url(s) queued")
    # Re-ingesting one URL would index its chunks twice for a slot it
    # already occupies.
    for target in dict.fromkeys(urls):
        try:
            outcome.indexed.append(
                await ingest_url(session_id, target)
            )
        except (
            SessionLimitError,
            UnsupportedDocumentError,
            UnsafeUrlError,
            httpx.HTTPError,
        ) as failure:
            logger.warning("Could not index %s: %s", target, failure)
            outcome.failed.append(FailedLink(url=target, reason=str(failure)))
    log_step(
        session_id,
        "←ingest",
        f"{len(outcome.indexed)} indexed, {len(outcome.failed)} failed",
    )
    return outcome


async def ingest_link_selection(
    session_id: str,
    base_url: str,
    selected_links: Sequence[str],
    index_base_url: bool = True,
) -> str:
    """The deferred call's external executor.

    The tool that raised CallDeferred is never re-entered, so the indexing it
    described happens here, and the report goes back to the model as the
    tool's return value.
    """
    targets = [base_url, *selected_links] if index_base_url else list(selected_links)
    log_step(
        session_id,
        "execute",
        f"indexing {len(targets)} page(s) (base page included={index_base_url})",
    )
    if not targets:
        return "The visitor chose not to add any of the linked pages."
    outcome = await ingest_each(session_id, targets)
    landed = "; ".join(
        f"{result.source} ({result.chunks_ingested} passages)"
        for result in outcome.indexed
    )
    report = [
        f"Indexed and now searchable: {landed}."
        if outcome.indexed
        else "Nothing could be indexed."
    ]
    if outcome.failed:
        lost = "; ".join(
            f"{failure.url} ({failure.reason})" for failure in outcome.failed
        )
        report.append(f"Could not index: {lost}.")
    return " ".join(report)


def offered_links(call: DeferredCall) -> set[str]:
    offered = call.metadata.get("links") or []
    return {
        str(link["url"])
        for link in offered
        if isinstance(link, dict) and "url" in link
    }


async def resolve_link_selection(request: LinkSelectionRequest) -> ChatResponse:
    log_step(
        request.session_id,
        "selection",
        f"{request.tool_call_id} picked {len(request.selected_links)} link(s)",
    )
    call = await require_call(
        request.session_id, request.tool_call_id, DeferredKind.CALL
    )
    unoffered = sorted(set(request.selected_links) - offered_links(call))
    if unoffered:
        # The selection decides what this endpoint fetches, so it is checked
        # against what was offered rather than trusted as URLs to go and read.
        raise UnofferedLinkError(
            f"Not offered for this call: {', '.join(unoffered)}."
        )
    base_url = call.metadata.get("url")
    if not isinstance(base_url, str):
        raise UnofferedLinkError(f"{request.tool_call_id} has no page to index.")

    outcome = await ingest_link_selection(
        request.session_id,
        base_url,
        request.selected_links,
        index_base_url=not call.metadata.get("page_already_indexed", False),
    )
    resolved = await resolve_call(
        request.session_id,
        request.tool_call_id,
        DeferredKind.CALL,
        CallResolution(content=outcome),
    )
    return await resume_if_ready(resolved)


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
    log_step(session_id, "file", f"{filename} ({content_type}, {len(data)} bytes)")
    await assert_capacity(session_id, document_name)
    document = load_document(data, filename, content_type)
    log_step(
        session_id,
        "extract",
        f"{len(document.texts)} text block(s), {len(document.images)} image(s)",
    )

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
    if images:
        log_step(session_id, "vision", f"captioning {len(images)} image(s)")
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
    log_step(
        session_id,
        "indexed",
        f"{document_name} -> {chunk_count} chunk(s), searchable={searchable}",
    )
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
    log_step(session_id, "url", f"fetching {url}")
    # The submitted URL is the document's identity, not the URL it redirects
    # to, so re-submitting the same link never consumes a second slot.
    await assert_capacity(session_id, url)
    page = await fetch_page(url)
    log_step(
        session_id,
        "←url",
        f"{page.content_type} {len(page.body)} bytes from {page.url}",
    )

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

    # Taken from the body already fetched — offering the links must not cost
    # a second request to the same page.
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
        log_step(session_id, "summary", summary.summary)
    except Exception as failure:
        # The page text is already indexed; losing the overview is not fatal.
        logger.warning("Could not summarise %s: %s", page.url, failure)

    store = get_vector_store()
    chunk_count = await store.add_sections(session_id, sections)
    searchable = await store.wait_until_searchable(session_id)
    log_step(
        session_id,
        "indexed",
        f"{url} -> {chunk_count} chunk(s), searchable={searchable}",
    )
    return KnowledgeIngestResponse(
        session_id=session_id,
        source=source,
        chunks_ingested=chunk_count,
        pages_summarised=summarised,
        searchable=searchable,
        documents_used=await store.count_documents(session_id),
        documents_allowed=ziza_settings.max_documents_per_session,
    )



