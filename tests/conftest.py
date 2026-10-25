import pytest


@pytest.fixture(autouse=True)
def default_weights(monkeypatch):
    """Pin the weights so a developer's own .env cannot change test results.

    Real environment variables beat the .env file, so these shadow it. Tests that exercise
    the overrides delete or replace these variables themselves.
    """
    for name, value in {"INDUSTRY": "0.40", "SIZE": "0.25", "SOURCE": "0.20", "RECENCY": "0.15"}.items():
        monkeypatch.setenv(f"WEIGHT_{name}", value)
