# Innovationist Tech Demo API

FastAPI service scaffolded with a layered, domain-oriented architecture.

## Stack

- **FastAPI** + **Uvicorn** (ASGI)
- **Pydantic Settings** for typed configuration
- **Beanie** (MongoDB ODM) — optional, enabled via `MONGO_URI`
- **Ruff** + **mypy (strict)** + **pytest** + **pre-commit**

## Getting started

```bash
make install        # uv sync
cp .env.example .env
make dev            # run with auto-reload
```

Then visit:
- http://localhost:8181/            — root status
- http://localhost:8181/api/health  — health check
- http://localhost:8181/docs        — OpenAPI docs

## Commands

```bash
make dev            # run with auto-reload
make start          # production run (0.0.0.0:8181)
make format         # ruff format + ruff check --fix
make lint           # ruff check
make mypy           # mypy type checking
make test           # pytest
make check          # lint + mypy + test
```

## Architecture

Request flow: **router → service → repo → Beanie Document**

```
app/
├── main.py              # FastAPI app + async lifespan
├── core/
│   ├── config.py        # pydantic-settings Settings
│   ├── db/              # Beanie init + document registry
│   ├── exceptions/      # error model + handlers
│   └── utils/           # time, repo helpers, logging
├── routes/routes.py     # central router, mounted at /api
└── <domain>/            # one package per domain
    ├── models.py        # Beanie Document (schema + indexes)
    ├── schemas.py       # Pydantic request/response models
    ├── repo.py          # database operations (no business logic)
    ├── service.py       # business logic and orchestration
    └── router.py        # HTTP endpoints + dependency injection
```

### Adding a domain

1. Create `app/<domain>/` as a package (`__init__.py`) with `models.py`, `schemas.py`, `repo.py`, `service.py`, `router.py`.
2. Register the router in `app/routes/routes.py`.
3. Register any Beanie models in `get_document_models()` in `app/core/db/db_config.py`.

## Conventions

- `utc_now()` — always use this (not `datetime.utcnow()` / `datetime.now()`).
- `find_one_or_raise()` — use for lookup-or-404 patterns in repos.
- Config is read via `app.core.config.settings` (pydantic-settings), not `os.environ`.
- New domains are proper packages, not bare `.py` files.
