"""End-to-end tests against a real MongoDB replica set.

Marked `integration` (see conftest): skipped unless TEST_MONGO_URI points at a
running replica set. CI starts one via `docker compose up -d mongo`.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration


def _session() -> str:
    """A unique session per test so runs don't collide in the shared test DB."""
    return f"it-{uuid.uuid4().hex[:8]}"


def test_health_reports_connected() -> None:
    with TestClient(app) as client:
        body = client.get("/api/health").json()
        assert body["database"] == "connected"


def test_flag_crud_roundtrip() -> None:
    session_id = _session()
    with TestClient(app) as client:
        base = f"/api/flags?session_id={session_id}"

        created = client.post(
            base, json={"key": "banner", "value": "hello", "enabled": True}
        )
        assert created.status_code == 201
        assert created.json()["key"] == "banner"

        # Duplicate active key is rejected.
        dup = client.post(
            base, json={"key": "banner", "value": "again", "enabled": True}
        )
        assert dup.status_code == 409

        listed = client.get(base).json()
        assert [flag["key"] for flag in listed] == ["banner"]

        toggled = client.put(
            f"/api/flags/banner?session_id={session_id}", json={"enabled": False}
        )
        assert toggled.status_code == 200
        assert toggled.json()["enabled"] is False

        assert (
            client.delete(f"/api/flags/banner?session_id={session_id}").status_code
            == 204
        )
        assert client.get(base).json() == []

        # Soft delete uses a partial unique index, so the key can be re-added.
        readded = client.post(
            base, json={"key": "banner", "value": "fresh", "enabled": True}
        )
        assert readded.status_code == 201


def test_sessions_are_isolated() -> None:
    session_a, session_b = _session(), _session()
    with TestClient(app) as client:
        client.post(
            f"/api/flags?session_id={session_a}",
            json={"key": "only-a", "value": "x", "enabled": True},
        )
        # A different session sees none of session A's flags.
        assert client.get(f"/api/flags?session_id={session_b}").json() == []
