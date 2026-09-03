import os

import pydantic_ai.models
import pytest

from app.core.config import settings

# Set in CI (and optionally locally) to a running MongoDB replica set. Integration
# tests run against this; when it's unset they are skipped, so the default test
# run needs no infrastructure.
_TEST_MONGO_URI = os.environ.get("TEST_MONGO_URI")

pydantic_ai.models.ALLOW_MODEL_REQUESTS = False

# Agent construction requires a provider key even though tests never call the
# provider (models are overridden with TestModel); a placeholder satisfies
# that check without granting real access.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-placeholder-key")


@pytest.fixture
def anyio_backend() -> str:
    """Run @pytest.mark.anyio tests on asyncio only (trio isn't installed)."""
    return "asyncio"


@pytest.fixture(autouse=True)
def configure_database(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the app at the right database for each test tier.

    - Unit tests run hermetically with no database: booting the app via
      TestClient starts in "no database" mode (the change-stream watcher is
      skipped).
    - Tests marked `integration` run against the real MongoDB replica set given
      by TEST_MONGO_URI, using a dedicated `innovationist_test` database. They
      are skipped when TEST_MONGO_URI is unset.
    """
    if request.node.get_closest_marker("integration"):
        if not _TEST_MONGO_URI:
            pytest.skip("integration test requires TEST_MONGO_URI")
        monkeypatch.setattr(settings, "mongo_uri", _TEST_MONGO_URI)
        monkeypatch.setattr(settings, "mongo_db_name", "innovationist_test")
    else:
        monkeypatch.setattr(settings, "mongo_uri", None)
