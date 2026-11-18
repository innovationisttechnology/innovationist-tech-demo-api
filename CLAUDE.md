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
.venv/bin/python -m pytest tests/test_content_sync.py
.venv/bin/python -m pytest tests/test_content_sync.py::test_name
.venv/bin/python -m pytest -m "not integration"   # what CI runs by default
```

Integration tests need a real replica set and are **skipped** unless `TEST_MONGO_URI` is set:

```bash
docker compose up -d mongo
TEST_MONGO_URI="mongodb://admin:adminpassword@localhost:27018/innovationist_test?authSource=admin&directConnection=true" \
  .venv/bin/python -m pytest -m integration
```

## Architecture

FastAPI + Beanie (MongoDB ODM). Python 3.13, managed with `uv`. Everything mounts under `/api`
via `app/routes/routes.py`; `app/main.py` owns the lifespan, CORS, and the exception handlers.

Domain packages are `router → service → repo → Beanie Document` (see README for the layout and the steps to add
a domain). One domain exists:

- **`content_sync`** (`/api/flags`) — session-scoped feature flags with live SSE fan-out.

The demo's AI chatbot no longer lives here; it runs on ziza-ai-api, which the demo UI's server calls directly.

### Cross-cutting invariants

- **The database is optional at boot.** `is_db_configured()` gates DB setup; without `MONGO_URI` the app
  boots and skips Beanie init and the change-stream watcher. Unit tests rely on this — `conftest.py` forces
  `mongo_uri = None`. `/api/flags` itself still needs a database; `RequiresDatabase` (`app/core/db`) turns a
  missing one into a 503 for routes that declare it.
- **Read config from `app.core.config.settings`**, never `os.environ`.
- **`utc_now()`**, not `datetime.now()`. **`find_one_or_raise()`** for lookup-or-404 in repos.
- **Everything is keyed by `session_id`** — an anonymous browser session, not a user. Flags carry it and
  expire on a `session_ttl_seconds` TTL index so demo data self-cleans.
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

## Conventions

- Ports are offset from the older sibling project so both can run side by side: API 8181, Mongo 27018,
  mongo-express 8281.
- CORS origins are explicit; `CORS_ORIGINS="*"` raises at startup.
- Tests are hermetic by default: unit tests run with no database. Async tests use `@pytest.mark.anyio`.
- CI (`.github/workflows/deploy.yml`) runs ruff → mypy → unit tests → integration tests, and deploys to EC2
  only from `main`.

## Code style

- **Names are the documentation.** Use descriptive, spelled-out names — `index` not `i`, `session_id` not
  `sid`, `failure` not `e`. If a function's name and signature already say what it does, it needs no docstring.
- **Do not write comments that restate the code.** No narrating obvious steps, parameters, or loops; no
  docstring whose only content is the function name in a sentence. If the comment would be true of any code
  that looks like this, delete it.
- **Write a comment only when it carries what the code cannot** — a non-obvious constraint, an external
  limitation, or why a surprising approach was chosen. The existing comments in this repo are the model: why
  deletes are soft in `content_sync`, why index reconciliation must precede `init_beanie`, why the lifespan
  swallows an already-failed watcher on shutdown, why CORS origins are stripped of trailing slashes. Every one
  of them explains a *why* that would otherwise be relearned the hard way. Keep them short.
- **Never remove text that is load-bearing rather than explanatory**, even though it looks like a comment:
  - type-checker and linter directives (`# type: ignore`, `# noqa`)
  - license headers, encoding declarations, and pragmas
