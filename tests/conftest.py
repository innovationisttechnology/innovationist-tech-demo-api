import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def disable_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic — never touch a live MongoDB.

    Booting the app via TestClient runs its lifespan, which would otherwise call
    init_db() and connect to MONGO_URI. Forcing it unset means the app starts in
    its "no database" mode: the health endpoint reports `not_configured` and the
    change-stream watcher is skipped. Real DB behaviour (change streams need a
    replica set) is covered by the manual/integration check instead.
    """
    monkeypatch.setattr(settings, "mongo_uri", None)
