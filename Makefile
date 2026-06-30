SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python

.PHONY: install dev dev-debug start format lint test mypy check

install:
	uv sync

dev:
	$(PY) -m uvicorn app.main:app --reload --port 8181

dev-debug:
	$(PY) -m uvicorn app.main:app --reload --port 8181 --log-level debug

start:
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8181

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest

mypy:
	$(PY) -m mypy app tests

check: lint mypy test
