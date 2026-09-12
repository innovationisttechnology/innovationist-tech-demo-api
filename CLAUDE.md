# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install        # uv sync
make dev            # uvicorn --reload on :8181  (make dev-debug for --log-level debug)
make check          # lint + mypy + test — run this before considering work done
make format         # ruff format + ruff check --fix
make test           # pytest
make mypy           # mypy app tests  (strict mode)
make docker-up      # api + atlas-local mongo + mongo-express via docker compose
```

Targeted test runs (the Makefile has no arg passthrough — call pytest directly):

```bash
.venv/bin/python -m pytest tests/test_ziza_scope_gate.py
.venv/bin/python -m pytest tests/test_ziza_scope_gate.py::test_name
.venv/bin/python -m pytest -m "not integration"   # what CI runs by default
```

Integration tests need a real replica set and are **skipped** unless `TEST_MONGO_URI` is set:

```bash
docker compose up -d mongo
TEST_MONGO_URI="mongodb://admin:adminpassword@localhost:27018/innovationist_test?authSource=admin&directConnection=true" \
  .venv/bin/python -m pytest -m integration
```

Classifier evals (needs the `evals` dependency group and a real `ANTHROPIC_API_KEY` — makes live model calls):

```bash
uv run --group evals python -m evals.classifier_evals
```

## Architecture

FastAPI + Beanie (MongoDB ODM) + pydantic-ai. Python 3.13, managed with `uv`. Everything mounts under `/api`
via `app/routes/routes.py`; `app/main.py` owns the lifespan, CORS, and the exception handlers.

Domain packages are `router → service → repo → Beanie Document` (see README for the layout and the steps to add
a domain). Two domains exist:

- **`content_sync`** (`/api/flags`) — session-scoped feature flags with live SSE fan-out.
- **`ziza_chat`** (`/api/ziza`) — a RAG chat demo over documents the visitor uploads.

### Cross-cutting invariants

- **The database is optional.** `is_db_configured()` gates everything DB-backed; without `MONGO_URI` the app
  boots, skips the change-stream watcher and vector index, and DB-dependent endpoints degrade (503 for ziza
  knowledge routes) rather than crash. Unit tests rely on this — `conftest.py` forces `mongo_uri = None`.
- **Two settings objects.** `app.core.config.settings` for the app, `app.ziza_chat.config.ziza_settings` for
  models/limits. Read config from these, never `os.environ`.
- **`load_dotenv()` in `app/main.py` is load-bearing.** pydantic-settings never writes to `os.environ`, but
  pydantic-ai reads `ANTHROPIC_API_KEY` from there. Any entrypoint that bypasses `main.py` (scripts, evals)
  must call `load_dotenv()` itself.
- **`utc_now()`**, not `datetime.now()`. **`find_one_or_raise()`** for lookup-or-404 in repos.
- **Everything is keyed by `session_id`** — an anonymous browser session, not a user. Vector chunks, captions,
  transcripts, and flags all carry it, and all expire on `session_ttl_seconds` TTL indexes so demo data
  self-cleans.
- **New Beanie models must be registered** in `get_document_models()` in `app/core/db/db_config.py`.
- **Index changes go through `app/core/db/index_migrations.py`.** `create_index` cannot alter an existing
  index (changing a TTL raises `IndexOptionsConflict` and fails startup), and Beanie never drops indexes it
  stopped declaring. Reconciliation runs from the models' own declarations *before* `init_beanie`; retired
  index names go in `SUPERSEDED_INDEXES`.

### content_sync: writes never broadcast directly

Services write to Mongo and return; the SSE fan-out comes exclusively from `watch_sync_flags`, a change-stream
watcher started in the lifespan that pushes into per-session queues held by `connection_manager`. This means a
**replica set is required** (hence the `mongodb/mongodb-atlas-local` image, not plain `mongo`) and deletes are
**soft** — a hard delete would give the change stream no `session_id`/`key` to route on, so `deleted: true`
tombstones are emitted as `flag.deleted` and reaped by a TTL index.

### ziza_chat: the scope gate is enforcement, the prompt is not

Request path: `classify` (fast Haiku classifier → `ClassifyResult`) → `resolve_scope` → chat agent.

`resolve_scope` in `service.py` is the actual boundary of the demo. It refuses in code — out-of-scope
classification, or a knowledge-base question where nothing clears `GATE_MIN_SCORE` — before the chat agent
ever runs. This is deliberate: the classifier only ever sees the visitor's message, never retrieved content, so
prompt injection inside an uploaded document cannot reach it. The chat agent's system prompt says the same
things, but a prompt is a default a model can be argued out of. **Changes to what the demo will answer belong
in the gate, not only in the prompt** — and gate refusals are still written to history via `refusal_messages()`
so the transcript has no holes.

Agents live in `app/ziza_chat/agents/` and are `@lru_cache`d factories (`get_chat_agent()`, etc.) — construct
them through those, and override with `TestModel`/`FunctionModel` in tests rather than mocking HTTP.

Ingestion (`ingest_file` / `ingest_url` / `ingest_knowledge`) is capacity-checked by `assert_capacity` *before*
any extraction, fetch, or captioning so a rejected upload costs nothing; it raises `SessionLimitError`, which
`main.py` maps to a 429. Images are captioned by a vision model into searchable text (cached by content hash,
session-scoped); HTML pages get an extra summary chunk. `clear_knowledge` must delete chunks, captions, **and**
transcripts together — each one holds the visitor's content.

Retrieval (`vector_store/store.py`) prefers Atlas `$vectorSearch` and falls back to ranking in Python on
`OperationFailure`, so it works on a non-Atlas Mongo. Atlas returns `(1 + cosine) / 2`; that is un-normalized
back to raw cosine so `MIN_SCORE` means one thing on both paths. Embeddings are local fastembed/ONNX
(`bge-small-en-v1.5`, torch-free on purpose) and silently truncate past 512 tokens — `chunking.py` enforces
that in tokens, not characters. The Dockerfile pre-downloads the model into `/opt/fastembed_cache`.

Chat history is one `ConversationTurn` document per turn (not one growing per-session document — tool returns
carry retrieved passages and would walk toward the 16MB limit). `deserialize` validates turn by turn so a
pydantic-ai schema change costs one turn, not the whole conversation.

## Conventions

- Ports are offset from the older sibling project so both can run side by side: API 8181, Mongo 27018,
  mongo-express 8281.
- CORS origins are explicit; `CORS_ORIGINS="*"` raises at startup.
- Tests are hermetic by default: `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` in `conftest.py`, so a test
  that reaches a real provider fails loudly. Async tests use `@pytest.mark.anyio`.
- CI (`.github/workflows/deploy.yml`) runs ruff → mypy → unit tests → integration tests, and deploys to EC2
  only from `main`.

## Code style

- **Names are the documentation.** Use descriptive, spelled-out names — `index` not `i`, `session_id` not
  `sid`, `failure` not `e`. If a function's name and signature already say what it does, it needs no docstring.
- **Do not write comments that restate the code.** No narrating obvious steps, parameters, or loops; no
  docstring whose only content is the function name in a sentence. If the comment would be true of any code
  that looks like this, delete it.
- **Write a comment only when it carries what the code cannot** — a non-obvious constraint, an external
  limitation, or why a surprising approach was chosen. The existing comments in this repo are the model: the
  512-token silent truncation in `embeddings.py`, the un-normalization of Atlas cosine scores, why deletes are
  soft in `content_sync`, why `append_turn` runs inside the `run_stream` context manager. Every one of them
  explains a *why* that would otherwise be relearned the hard way. Keep them short.
- **Never remove text that is load-bearing rather than explanatory**, even though it looks like a comment:
  - type-checker and linter directives (`# type: ignore`, `# noqa`)
  - docstrings and `Field(description=...)` text that is sent to a model — pydantic-ai tool docstrings
    (`search_knowledge_base`, `current_datetime`) and the descriptions on `ClassifyResult` are prompt content
    that changes model behavior, not documentation
  - license headers, encoding declarations, and pragmas
