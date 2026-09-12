# Building with pydantic-ai

How the `ziza_chat` module is built, in the order you would build it. Every
snippet is real code from this repository, so you can open the file next to the
step and see it in context.

The end product: an assistant that classifies each message, retrieves from a
per-session document store, answers only from that material, and describes
uploaded images so they can be searched as text.

**Contents**

1. [Install and configure](#1-install-and-configure)
2. [Define the output shape](#2-define-the-output-shape)
3. [Your first agent](#3-your-first-agent)
4. [Dependencies](#4-dependencies)
5. [Tools](#5-tools)
6. [Dynamic instructions](#6-dynamic-instructions)
7. [Images and other media](#7-images-and-other-media)
8. [Running agents](#8-running-agents)
9. [Seeing what the agent did](#9-seeing-what-the-agent-did)
10. [Conversation memory](#10-conversation-memory)
11. [Bounding the context](#11-bounding-the-context)
12. [Rolling summarisation](#12-rolling-summarisation)
13. [Prompt caching](#13-prompt-caching)
14. [Human in the loop](#14-human-in-the-loop)
15. [Follow-ups and the scope gate](#15-follow-ups-and-the-scope-gate)
16. [Testing without calling a model](#16-testing-without-calling-a-model)
17. [Evaluating quality](#17-evaluating-quality)
18. [Putting it in production](#18-putting-it-in-production)
19. [Gotchas](#19-gotchas)

---

## 1. Install and configure

```bash
uv add "pydantic-ai>=2.5.0" python-dotenv
```

Model names live in settings so they can be swapped without touching code
(`app/ziza_chat/config.py`):

```python
class ZizaChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ziza_chat_model: str = "anthropic:claude-sonnet-5"
    ziza_classifier_model: str = "anthropic:claude-haiku-4-5"
    ziza_vision_model: str = "anthropic:claude-haiku-4-5"
```

The `"provider:model"` string form is worth preserving: swapping providers
becomes an env var rather than a code change.

> **The first thing that will bite you.** pydantic-settings loads `.env` into a
> *settings object*. pydantic-ai reads `ANTHROPIC_API_KEY` from the *process
> environment*. These are not the same place, and nothing connects them — so
> a key sitting in `.env` produces `UserError: Set the ANTHROPIC_API_KEY
> environment variable` while every other setting resolves fine.
>
> Fix it once, at the top of `app/main.py`, before anything constructs an agent:
>
> ```python
> from dotenv import load_dotenv
> load_dotenv()
> ```
>
> Any script that bypasses `main.py` needs its own `load_dotenv()`.

---

## 2. Define the output shape

Decide what you want *back* before you write the agent. A pydantic model is the
contract, and pydantic-ai makes the model produce data matching it — no parsing,
no "respond with JSON" in the prompt (`app/ziza_chat/agents/outputs.py`):

```python
class Scope(str, Enum):
    KNOWLEDGE_BASE = "knowledge base"
    ASSISTANT = "assistant"
    OUT_OF_SCOPE = "out of scope"


class ClassifyResult(BaseModel):
    scope: Scope = Field(
        default=Scope.KNOWLEDGE_BASE,
        description="Where an answer would have to come from. ...",
    )
    intents: list[Intent] = Field(
        min_length=1,
        description="All intents that apply, ordered from most to least dominant.",
    )
    needs_rag: bool = Field(description="True if ... retrieved knowledge would help.")
    rag_query: str | None = Field(default=None, description="...")

    @property
    def primary(self) -> Intent:
        return self.intents[0] if self.intents else Intent.QUESTION
```

Three things doing real work here:

- **`Field(description=...)` is prompt text.** It is sent to the model as part
  of the output schema and changes behaviour. Treat it as API surface, not
  documentation — this is not a comment you can tidy away.
- **Validators are enforcement.** `min_length=1` means an empty `intents` list
  is rejected and the model is asked again. Constraints you would normally
  write as defensive code become retries.
- **Enums constrain the vocabulary.** The model returns `"question"`, and you
  get `Intent.QUESTION`.

Give every field a sensible default where you can. A required field the model
omits is a failed run; a defaulted one degrades. Note `scope` defaults to
`KNOWLEDGE_BASE` — the restrictive value, so a malformed response cannot
accidentally widen what the assistant will answer.

---

## 3. Your first agent

`app/ziza_chat/agents/classifier.py`:

```python
@lru_cache
def get_classifier_agent() -> Agent[None, ClassifyResult]:
    return Agent[None, ClassifyResult](
        ziza_settings.ziza_classifier_model,
        output_type=ClassifyResult,
        instructions=CLASSIFIER_PROMPT,
        model_settings=ModelSettings(temperature=0, max_tokens=1024),
    )
```

Small file, five decisions:

**Build it in a function, not at import time.** Constructing an `Agent`
instantiates its provider, which requires an API key. At module level that means
importing your app — in a test, a script, a CI lint step — fails without one.

**`@lru_cache` makes it a singleton.** The agent is stateless and reusable;
rebuilding per request wastes work.

**Parameterise the constructor**, `Agent[None, ClassifyResult](...)`, rather
than annotating the variable. The two type parameters are `[DepsType,
OutputType]`; `None` means this agent takes no dependencies. This also sidesteps
a PyCharm false positive — it doesn't fully support the TypeVar defaults
pydantic-ai uses and will infer `Agent[object, str]` from a bare call.

**`instructions=`, not `system_prompt=`.** Instructions are not replayed from
prior runs' message history. Once you add conversation memory, a `system_prompt`
from an earlier turn can leak back in; instructions always reflect the current
code.

**`temperature=0` for decisions.** Classification has one right answer, and
determinism makes evals meaningful — a score change should come from your prompt
change, not from sampling luck. Leave temperature alone for conversational
agents, where variety is desirable.

> `temperature` is rejected outright (HTTP 400) by current-generation Anthropic
> models. It works on Haiku 4.5, which is why the classifier can use it. Do not
> copy the pattern to an agent on a newer model — see
> [Gotchas](#19-gotchas).

---

## 4. Dependencies

Tools need request-scoped things: a database handle, a session id, the current
user. That is what `deps_type` is for (`app/ziza_chat/deps.py`):

```python
@dataclass
class ChatDeps:
    session_id: str
    intent: str = "unknown"
    documents: list[str] = field(default_factory=list)
    vector_store: MongoVectorStore | None = None
```

Declare it on the agent and pass an instance per run:

```python
agent = Agent[ChatDeps, str](model, deps_type=ChatDeps, ...)
result = await agent.run(message, deps=ChatDeps(session_id="abc", ...))
```

Tools reach it through `context.deps`. Keep it a plain dataclass with no
behaviour — it is a bag of handles, and making it optional-friendly
(`MongoVectorStore | None`) lets tools degrade instead of crashing when a
dependency is unavailable.

---

## 5. Tools

A tool is a plain function. Pass the function itself — pydantic-ai builds the
schema from the signature and infers whether it wants context:

```python
agent = Agent[ChatDeps, str](
    model,
    deps_type=ChatDeps,
    instructions=SYSTEM_PROMPT,
    tools=[search_knowledge_base, current_datetime],
)
```

You only need the `Tool(...)` wrapper to set options (`name`, `max_retries`,
`prepare`). Wrapping without options adds nothing and, with generics involved,
gives type checkers more to get wrong.

Two shapes, from `app/ziza_chat/tools/common.py`:

```python
def current_datetime() -> str:
    """Get the current date and time (UTC).

    Use this whenever an answer depends on today's date or the current time —
    e.g. "what day is it", relative dates ("next Friday"), or judging how
    recent something is. Never guess the date from prior knowledge.
    """
    return datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M:%S UTC")


async def search_knowledge_base(context: RunContext[ChatDeps], query: str) -> str:
    """Search this session's knowledge base for passages relevant to the query.

    Call this whenever the answer might depend on a document the visitor added.
    ...
    """
    store = context.deps.vector_store
    if store is None:
        return "The knowledge base is not available in this session."
    retrieved = await store.search(context.deps.session_id, query)
    return format_retrieved_chunks(query, retrieved)
```

The rules that matter:

**The docstring is the tool description sent to the model.** It is how the model
decides *when* to call the tool and *what* to pass. This is the single most
common thing to get wrong — a tool that "isn't being called" usually has a
docstring that doesn't say when to call it. Write for the model: when to reach
for it, what the argument should look like, what comes back.

**`RunContext[DepsT]` as the first parameter is the signal.** With it, the tool
gets dependencies; without it, it doesn't. Nothing else changes.

**Return a string the model can read.** Not a dump of objects — a formatted,
labelled result. Ours tags each passage with its source and score so the model
can cite them.

**Failures are results, not exceptions.** "No passages matched" is an answer;
raising would make the model retry something that will fail identically.

---

## 6. Dynamic instructions

Static instructions are a constant. `@agent.instructions` adds text computed per
run — the place to inject request state (`app/ziza_chat/agents/chat.py`):

```python
@agent.instructions
def add_intent(context: RunContext[ChatDeps]) -> str:
    return (
        f"A fast classifier labelled this message: {context.deps.intent}. "
        "Treat it as a hint about tone and whether retrieval is likely to "
        "help — trust the message itself when the two disagree."
    )


@agent.instructions
def add_session_documents(context: RunContext[ChatDeps]) -> str:
    documents = context.deps.documents
    if not documents:
        return "This session's knowledge base is empty."
    listed = "\n".join(f"- {document}" for document in documents)
    return f"This session's knowledge base holds {len(documents)} document(s):\n{listed}"
```

Stack as many as you like; they are concatenated. The second one fixes a real
bug: without it, the agent had no way to know what the session held and would
claim the knowledge base was empty whenever a search happened to miss.

Note the function's *return value* is used, not its docstring — unlike tools.

---

## 7. Images and other media

Pass a list instead of a string, mixing text and media
(`app/ziza_chat/agents/vision.py`):

```python
from pydantic_ai.messages import BinaryImage, UserContent

async def describe_image(data: bytes, media_type: str) -> ImageDescription:
    prompt: list[UserContent] = [
        "Describe this image for a search index.",
        BinaryImage(data, media_type=media_type),
    ]
    result = await get_vision_agent().run(prompt)
    return result.output
```

`BinaryImage` is the narrowed subclass of `BinaryContent` — its `media_type` only
accepts image types, so passing a PDF is a type error rather than a runtime
surprise. Also available: `ImageUrl`, `DocumentUrl`, `AudioUrl`, `VideoUrl`, and
`BinaryContent` for anything else. Note `media_type` is keyword-only.

Structured output pays off doubly here. Rather than a prose caption, we ask for
fields designed for retrieval:

```python
class ImageDescription(BaseModel):
    summary: str = Field(description="What the image is and what it conveys...")
    visible_text: str = Field(default="", description="Text visible in the image...")
    entities: list[str] = Field(default_factory=list, description="People, products...")
    data_points: list[str] = Field(default_factory=list, description="Concrete values...")
```

A prose caption ("a bar chart of quarterly revenue") matches almost no real
question. Transcribed text and extracted figures do.

---

## 8. Running agents

```python
result = await agent.run(message, deps=deps)
result.output          # typed as your output_type
result.usage           # input_tokens / output_tokens — a property, not a method

async with agent.run_stream(message, deps=deps) as result:
    async for chunk in result.stream_text(delta=True):
        yield chunk
```

Bound what a run may consume — important once an agent has tools and could loop:

```python
from pydantic_ai.usage import UsageLimits

await agent.run(
    message,
    deps=deps,
    usage_limits=UsageLimits(request_limit=5, tool_calls_limit=6),
)
```

Other knobs: `retries=` (on the agent or a run) caps output-validation retries;
`ModelSettings(max_tokens=...)` caps response length; `max_concurrency=` caps
parallel tool execution.

---

## 9. Seeing what the agent did

Do not add logging inside each tool. `event_stream_handler` receives the agent's
event stream, so one handler covers every tool, present and future
(`app/ziza_chat/tool_logging.py`):

```python
async def log_tool_events(
    context: RunContext[ChatDeps], events: AsyncIterable[AgentStreamEvent]
) -> None:
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            logger.info("tool call   %s(%s)", event.part.tool_name, summarize(event.part.args))
        elif isinstance(event, FunctionToolResultEvent):
            outcome = "retry" if isinstance(event.part, RetryPromptPart) else "return"
            logger.info("tool %-6s %s -> %s", outcome, event.part.tool_name,
                        summarize(event.part.content))
```

Attach it to `run`, `run_stream`, or `run_sync`:

```python
result = await agent.run(message, deps=deps, event_stream_handler=log_tool_events)
```

Output:

```
tool call   search_knowledge_base({"query": "production release approval"})
tool return search_knowledge_base -> [source: handbook.txt | relevance: 0.68] ...
```

Watch for `tool retry` — that is the model being told its arguments were invalid
and trying again. It explains round-trips you would otherwise not account for.
Truncate long values; retrieved passages will bury everything else.

---

## 10. Conversation memory

Every request above builds its context from scratch, so the agent cannot see the
previous turn. History in pydantic-ai is a `list[ModelMessage]` — the full
transcript including tool calls and their results, not just the text — and you
hand it back on the next run:

```python
result = await agent.run(message, deps=deps, message_history=previous)
result.all_messages()   # previous + this run
result.new_messages()   # only what this run added
```

Persist with `ModelMessagesTypeAdapter` (`pydantic_ai.messages`). Use
`mode="json"` so every value is a JSON primitive — it round-trips through
`validate_python` and is safe to hand to MongoDB:

```python
def serialize_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return list(ModelMessagesTypeAdapter.dump_python(list(messages), mode="json"))
```

Do not hand-roll this. The part types are a discriminated union and the adapter
is the only thing that knows how to rebuild them.

**Choosing `instructions=` in §3 pays off here.** Instructions are not replayed
from history, so the prompt the agent runs under is always the one in your code.
A `system_prompt` from an earlier turn would leak back in.

**Store one document per turn, not one per session.** Tool results carry
retrieved passages, so a session document grows towards MongoDB's 16MB limit,
and two concurrent requests rewriting it lose a turn between them
(`app/ziza_chat/history_store/`).

**Validate turn by turn on the way back out.** A pydantic-ai upgrade can change
the message schema; validating the whole transcript as one batch means old turns
take every live conversation down with them. Per-turn, one unreadable turn costs
only its own context.

**Record what the gate answered.** Out-of-scope messages never reach the agent,
so there is no run to persist — and without a synthesised turn the transcript has
a hole that the next message ("why not?") refers back to.

> **The streaming trap.** A `StreamedRunResult` holds a list the run mutates,
> and the final message only lands once the stream is consumed. `append_turn`
> must sit *inside* the `async with` and *after* the `async for`, or you persist
> a turn with no answer in it.

---

## 11. Bounding the context

Replaying everything means resending every previously retrieved passage forever.
`ProcessHistory` trims before each model request:

```python
from pydantic_ai.capabilities import ProcessHistory

agent = Agent[ChatDeps, str](..., capabilities=[ProcessHistory(trim_to_recent_turns)])
```

`history_processors=` is deprecated and warns; `capabilities=[ProcessHistory(...)]`
is the current form.

Two constraints make the naive version wrong.

**You cannot cut at an arbitrary message.** A tool result whose tool call was
trimmed away is rejected outright:

```
tool_use ids were found without tool_result blocks immediately after
```

So only ever cut at a turn boundary — a `ModelRequest` carrying a
`UserPromptPart`.

**Trimming fights caching.** Dropping the oldest turn on every request rewrites
the prompt prefix each time, and prefix caching (§13) then never reads. Quantise
the cut instead, so it moves in whole steps and the prefix stays byte-identical
in between:

```python
def turns_to_drop(total_turns: int) -> int:
    if total_turns <= MAX_PROMPT_TURNS:
        return 0
    excess_turns = total_turns - MAX_PROMPT_TURNS
    return min(
        ceil(excess_turns / TRIM_STEP_TURNS) * TRIM_STEP_TURNS,
        total_turns - 1,
    )
```

You eat one cold turn per step rather than one per turn. The window sawtooths
between 6 and 10 turns — that is the price of holding the cut still.

**Compute the offset from the absolute turn count, not from a sliding tail.**
Two sliding windows composed (a store that loads "the last 20" plus a trimmer
that keeps "the last 10") slide by one turn again past turn 20, silently undoing
the alignment. Derive the offset once, in the store, from `count()`.

---

## 12. Rolling summarisation

Trimming forgets. Past the window the agent genuinely cannot see turn 1, and for
a session that ran long that is real information loss. A rolling summary is what
makes the forgetting graceful.

Four decisions carry it.

**Fold, don't re-read.** Summarise the previous summary plus only the turns that
just dropped out, so cost is bounded by the step size rather than by conversation
length (`app/ziza_chat/agents/conversation_summary.py`).

**Refresh on the step boundary from §11.** That runs it once every few turns
instead of every turn, and keeps the summary block stable in the prompt between
refreshes — a summary that changed every turn would undo the caching work.

**Run it off the response path.** `ProcessHistory` fires before *every* model
request, including after each tool call, so summarising there puts a model call
in the visitor's latency budget. Refresh after the turn is stored, in a tracked
background task, and log-and-drop on failure.

**Carry source labels through the schema.** The assistant has to keep citing
documents correctly after the passages are gone, so make the labels a required
field rather than something the summariser may paraphrase away.

> **The summary is an injection path.** It is *derived from* retrieved document
> text and then fed back in — but it no longer looks like tool output. Keep it in
> the user channel behind an explicit header, never in `instructions`, which would
> hand laundered document content operator authority. Tell the summariser it is
> describing a transcript and must report that a request was made rather than
> reproduce it.

---

## 13. Prompt caching

Caching is a prefix match over `tools` → `system` → `messages`: any byte change
invalidates everything after it. pydantic-ai exposes Anthropic's controls through
provider-prefixed settings:

```python
from pydantic_ai.models.anthropic import AnthropicModelSettings

model_settings=AnthropicModelSettings(
    anthropic_cache_instructions=True,
    anthropic_cache_tool_definitions=True,
    anthropic_cache=True,
)
```

Measured over four turns of this agent, at Sonnet rates:

| Turn | Uncached | Cached | |
|---|---|---|---|
| 1 (cold) | $0.0060 | $0.0074 | +25% — the write premium |
| 2 | $0.0065 | $0.0012 | −81% |
| 3 (intent changed) | $0.0071 | $0.0029 | −59% |
| 4 | $0.0072 | $0.0009 | −88% |
| **total** | **$0.0267** | **$0.0125** | **−53%** |

**Dynamic instructions do not defeat it.** This was the surprise: the `@agent.instructions`
functions from §6 change per request, and a varying system block should invalidate
every message after it. pydantic-ai sorts instruction parts static-first and puts
the breakpoint after the last static one, so a changed intent hint rewrites only
what follows. Row 3 still reads 1,727 tokens from cache.

**Verify rather than assume.** `RunUsage` exposes `cache_read_tokens` and
`cache_write_tokens`. If reads are zero across turns, something in the prefix is
varying.

**Provider portability survives.** Every provider's settings class carries the
same note in its source — *"ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN
MERGE THEM WITH OTHER MODELS"* — so these keys are inert on other providers
rather than an error. `CachePoint` is the portable alternative: honoured by
Anthropic and Bedrock, translated by OpenRouter, filtered out by OpenAI (which
caches automatically anyway), ignored by Google.

---

## 14. Human in the loop

Some tool calls should not be the model's decision alone. The intuitive model is
a blocking callback — the tool pauses, asks, continues — and it cannot work that
way, because the human is behind an HTTP boundary and may answer minutes later on
a different worker.

So the run **ends**. Put `DeferredToolRequests` in the output type, and a gated
tool's call becomes the run's output instead of an answer:

```python
CHAT_OUTPUT_SPEC: OutputSpec[str | DeferredToolRequests] = [str, DeferredToolRequests]

agent = Agent[ChatDeps, str | DeferredToolRequests](..., output_type=CHAT_OUTPUT_SPEC)
```

Gate a whole tool with `requires_approval=True`, or conditionally from inside it:

```python
if not context.tool_call_approved:
    raise ApprovalRequired(metadata={"summary": "Delete 2 document(s)."})
```

`RunContext.tool_call_approved` is what stops an infinite pause loop — on the
resumed run it is `True`, so the tool proceeds. The `metadata` arrives in
`DeferredToolRequests.metadata`, keyed by `tool_call_id`, which is how you show a
human a readable description instead of raw arguments.

Resume with the decision injected as the tool's outcome:

```python
results = DeferredToolResults()
results.approvals[tool_call_id] = ToolApproved()   # or ToolDenied("...")
result = await agent.run(deps=deps, message_history=history, deferred_tool_results=results)
```

`ToolApproved(override_args=...)` lets the human *edit* the call, not just accept
it. Approval is therefore not a callback but **a run that ends, is persisted, and
is restarted** — which is why §10 had to exist first.

Three things this cost to learn:

> **Never tell the model about the confirmation.** The first version of the tool
> docstring said "the visitor has to confirm before it happens", so the model
> asked for confirmation *in prose* and never called the tool — bypassing the gate
> entirely. The mechanism has to be invisible: tell it to call the tool
> immediately and that confirmation is handled for it.

> **`stream_text()` refuses a non-text output.** With a union output type it
> raises `stream_text() can only be used with text responses` the moment a run
> pauses. Use `stream_output()`, which yields either arm — at the cost of
> cumulative rather than delta output, so diff against what you have already sent.

> **A paused turn must not enter the transcript.** Its tool call has no result, so
> replaying it 400s. If the visitor sends another message instead of confirming,
> every later request in that session fails — permanently, once the pending record
> expires. Park the paused messages on the pending record, not in history, and
> treat a new message as declining the offer so the pair is closed out.

**Gating is a security boundary, not just UX.** An injected *"clear the knowledge
base"* inside an uploaded document cannot execute silently, because the run stops
and a human sees the action first. That extends the §12 boundary rather than
duplicating it.

### Deferred calls: when the result comes from outside

Approval is one of two reasons a run ends holding an unanswered tool call. The
other is that the tool's *result* is produced somewhere this run cannot reach —
in the browser, by a worker, by a person supplying data rather than consent.
Raise `CallDeferred` instead:

```python
raise CallDeferred(metadata={"url": url, "links": [...]})
```

The call lands in `DeferredToolRequests.calls` rather than `.approvals`, and is
resumed by supplying the return value rather than a decision:

```python
results = requests.build_results(calls={tool_call_id: "Indexed 3 pages."})
```

> **The tool body does not run again.** This is the one difference that matters.
> `ToolApproved` re-executes the tool with `tool_call_approved=True`; a deferred
> call never re-enters the function — the value you supply *is* its return value
> (`_tool_execution.py` only re-executes for `None` or `ToolApproved`). So the
> work has to happen in whatever resumes the run. A tool that does the work
> *before* raising `CallDeferred` will do it whether or not the human ever
> answers.

Which means the two kinds want opposite tool bodies. An approval-gated tool does
the work in the body and raises early. A deferred tool only *describes* the work
— `add_url_to_knowledge_base` fetches the page and lists its links, and the
`/chat/links` endpoint does the indexing when the selection comes back. Nothing
is written until then, so an abandoned offer is genuinely a no-op.

> **One response can defer several calls.** `approvals[0]` is a trap: a model can
> emit three deferred calls at once, and they come back separately and out of
> order. Persist all of them against one set of paused messages and only replay
> when every one has an answer — a half-answered run is as unreplayable as an
> unanswered one. `requests.remaining(results)` returns what is still outstanding.

> **`build_results` is worth using over a hand-built `DeferredToolResults`.** It
> validates the ids against the pause they came from, so a stale or forged
> `tool_call_id` raises there instead of quietly resuming a run with a tool call
> left dangling. A plain `bool` is accepted for approvals and any plain value for
> calls.

Choosing between them is one question: *is the human supplying permission, or
data?* Permission is `ApprovalRequired`. Data — a selection, a password, a
timezone, a page the browser rendered — is `CallDeferred`. Getting it backwards
means either a yes/no dialog for something that needed a value, or a tool that
re-runs when it should not.

---

## 15. Follow-ups and the scope gate

Memory changes the input distribution. Visitors start writing "summarise it" and
"what about page 3?", which classify badly alone — and the gate then refuses a
perfectly valid follow-up. Measured on the real classifier:

| Message | Previous | `rag_query` |
|---|---|---|
| `summarise it` | — | `'<UNKNOWN>'` |
| `summarise it` | `what does the handbook say about releases?` | `'handbook releases'` |
| `what about page 3?` | — | `'page 3'` |
| `what about page 3?` | `explain the onboarding document` | `'onboarding document page 3'` |

Pass the previous message as context — and **only** the previous *visitor*
message. Not the transcript, not retrieved passages, and not the rolling summary,
all of which contain document text. The classifier never seeing retrieved content
is what makes the gate injection-resistant (§16); widening its input is the
easiest way to lose that property by accident.

---

## 16. Testing without calling a model

Two lines in `tests/conftest.py` make the whole suite safe and offline:

```python
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False
os.environ.setdefault("ANTHROPIC_API_KEY", "test-placeholder-key")
```

The first turns any real request into an error — so a test that accidentally
calls a provider fails loudly instead of spending money. The second satisfies
agent *construction*, which needs a key even when no call is made.

`TestModel` synthesises schema-valid output, proving your wiring:

```python
@pytest.mark.anyio
async def test_returns_valid_classification() -> None:
    agent = get_classifier_agent()
    with agent.override(model=TestModel()):
        result = await agent.run("who is Sarah?")
    assert isinstance(result.output, ClassifyResult)
```

Or force an exact response to test how your code handles it:

```python
forced_output = TestModel(
    custom_output_args={
        "intents": ["question", "greeting"],
        "needs_rag": True,
        "rag_query": "Sarah",
    }
)
with agent.override(model=forced_output):
    result = await agent.run("Hey, who is Sarah?")

assert result.output.intents == [Intent.QUESTION, Intent.GREETING]
assert result.output.primary is Intent.QUESTION
```

`agent.override()` is a context manager, so the swap is scoped and cannot leak
between tests.

`pytest.mark.anyio` needs a backend fixture (anyio ships with pytest support, so
`pytest-asyncio` is not required):

```python
@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
```

---

## 17. Evaluating quality

Tests prove the plumbing. They say nothing about whether the classifier is
*right*. That is `pydantic-evals` (`evals/classifier_evals.py`):

```python
@dataclass                       # evaluators are dataclasses, not plain classes
class PrimaryIntentEntailed(Evaluator[str, ClassifyResult]):
    threshold: float = 0.5

    def evaluate(self, ctx: EvaluatorContext[str, ClassifyResult]) -> EvaluationReason:
        probability = intent_entailment(ctx.inputs, ctx.output.primary)
        return EvaluationReason(
            value=probability >= self.threshold,
            reason=f"P({ctx.output.primary.value!r}) = {probability:.2f}",
        )


dataset: Dataset[str, ClassifyResult] = Dataset(
    name="intent-classifier",    # `name` is required
    cases=CASES,
    evaluators=[PrimaryIntentEntailed(), MeanIntentEntailment()],
)
report = dataset.evaluate_sync(classify_task)
report.print(include_input=True, include_reasons=True)
```

Run it:

```bash
uv run --group evals python -m evals.classifier_evals
```

The grader here is a local NLI cross-encoder, not another LLM call — it scores
whether the message entails "This is a question." That keeps grading free and
offline. Evals live in `evals/`, outside `tests/`, because they call real models
and cost money; their dependencies sit in a separate group so torch never enters
the app image.

---

## 18. Putting it in production

**Enforce boundaries in code, not in the prompt.** This is the lesson that cost
the most to learn here. A system prompt saying "never answer general questions"
holds *most* of the time, which is worse than it sounds: it looks fixed while
still leaking. The durable version, in `app/ziza_chat/service.py`:

```python
async def chat(request: ChatRequest) -> ChatResponse:
    classification = await classify(request.message)
    refusal = await resolve_scope(request.session_id, classification)
    if refusal is not None:
        return ChatResponse(..., response=refusal)     # agent never runs
    ...
```

Out-of-scope messages never reach the chat agent, so there is nothing to argue
with, reframe as fiction, or override from inside an uploaded document. Note
the classifier only ever sees the user's message — never retrieved content — so
an injected instruction in a document cannot reach the decision. Keep the prompt
rule too, as the default for cases code cannot judge, but do not rely on it.

**Choose the model per job and measure.** Vision captioning on Opus cost ~$65
per 1,000 images; Haiku extracted the same facts for ~$5. Measure before
assuming a bigger model is needed:

```python
result = await agent.run(prompt)
usage = result.usage
cost = (usage.input_tokens * INPUT_RATE + usage.output_tokens * OUTPUT_RATE) / 1e6
```

**Cache work that costs a model call.** Image captions are keyed by a hash of
the image bytes, so re-uploading the same file costs nothing.

**Bound the fan-out.** An image-heavy PDF is one upload and potentially hundreds
of model calls. Cap the count and the concurrency (`max_images_per_document`,
`max_concurrent_captions`), and log when you truncate.

**Deployment checklist:**

- Provider keys reach the *container environment*, not just a settings file.
  Agents are built lazily, so a missing key does not stop the app booting or
  failing its healthcheck — it fails every request behind a green deploy. Fail
  the deploy on a missing secret and log a startup error.
- Fail loudly at startup for anything a request will need.
- Pre-download local models at image build time; otherwise the first request
  pays for the download and a restart repeats it.

---

## 19. Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `UserError: Set the ANTHROPIC_API_KEY environment variable` while `.env` clearly has it | pydantic-settings loads `.env` into an object; pydantic-ai reads `os.environ` | `load_dotenv()` before any agent is constructed |
| HTTP 400 `temperature is deprecated for this model` | Current-generation Anthropic models reject sampling parameters | Drop `temperature`; it is still valid on Haiku 4.5 |
| Everything fails without a key at import time | Agent built at module level | Build inside an `@lru_cache` function |
| A tool is never called | Its docstring doesn't say when to use it | The docstring is the model's API documentation — write it for the model |
| PyCharm: `Expected Agent[None, X], got Agent[object, str]` | Incomplete support for PEP 696 TypeVar defaults | Parameterise the constructor: `Agent[None, X](...)`; verify with mypy |
| PyCharm: "unexpected argument" on `BinaryImage(...)` | Cannot read `@pydantic_dataclass` generated `__init__` | `# noinspection PyArgumentList` on the line directly above |
| `'RunUsage' object is not callable` | `result.usage` is a property in 2.5 | Drop the parentheses |
| Comment cleanup silently changes behaviour | Tool docstrings and `Field(description=...)` are prompt text | Never strip them; they are API surface |
| The prompt boundary "mostly" holds | A prompt is a default, not a control | Enforce in code before the agent runs |
| `history_processors` warns as deprecated | Superseded in 2.x | `capabilities=[ProcessHistory(fn)]` |
| `stream_text() can only be used with text responses` | The run ended in `DeferredToolRequests`, not text | `stream_output()`, which yields either arm of the union |
| `tool_use ids were found without tool_result blocks` | Replaying a paused approval, or a cut that split a turn | Keep paused messages off the transcript; only cut at user-turn boundaries |
| An approval-gated tool is never called; the model asks in prose instead | Its docstring mentioned that the visitor must confirm | Never tell the model about the gate — say to call it immediately |
| Cache hit rate is zero on a long session | The trim boundary slides one turn at a time, rewriting the prefix | Quantise the cut; derive the offset from the absolute turn count |
| PyCharm: `No overload of 'Agent' matches` on `output_type=[...]` | `OutputSpec` is a recursive `TypeAliasType` PyCharm cannot expand | Annotate a typed constant; verify with mypy, `# noinspection PyTypeChecker` |
| `CollectionWasNotInitialized` constructing a Beanie `Document` in a test | `__init__` reads settings that only exist after `init_beanie` | Use a plain stand-in object in unit tests |

---

## Where things live

| Path | Purpose |
|---|---|
| `app/ziza_chat/agents/outputs.py` | Structured output models |
| `app/ziza_chat/agents/classifier.py` | Fast classifier, structured output |
| `app/ziza_chat/agents/chat.py` | Main agent: tools, dynamic instructions |
| `app/ziza_chat/agents/vision.py` | Multimodal image captioning |
| `app/ziza_chat/tools/common.py` | Tool implementations |
| `app/ziza_chat/deps.py` | Per-request dependencies |
| `app/ziza_chat/tool_logging.py` | `event_stream_handler` |
| `app/ziza_chat/service.py` | Orchestration and the scope gate |
| `tests/conftest.py` | Offline test harness |
| `app/ziza_chat/history_store/` | Conversation memory: turns, trimming, summary, repair |
| `app/ziza_chat/hitl/` | Paused runs: approvals and deferred calls |
| `app/ziza_chat/tools/knowledge.py` | The approval-gated and deferred tools |
| `app/ziza_chat/agents/conversation_summary.py` | Rolling summariser |
| `evals/classifier_evals.py` | Quality evaluation |
