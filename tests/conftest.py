import os
from types import SimpleNamespace

import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch, tmp_path):
    """Cut every test off from the machine it's running on.

    Two leaks to close. Environment variables: matched case-insensitively
    against the real environment, because pydantic-settings reads `api_key` and
    `API_KEY` alike, so deleting only the uppercase spelling leaves the other
    one live. The field list comes from the model, so a setting added later is
    covered without anyone remembering to update this. And the .env file, which
    Settings reads by relative path, so running from a scratch directory means
    there isn't one to find.

    This exists because `API_AUTH_TOKEN` in a developer's own environment made
    the happy-path test return 401.
    """
    fields = {name.lower() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.lower() in fields:
            monkeypatch.delenv(name, raising=False)

    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def text_block(text):
    return SimpleNamespace(type="text", text=text)


class FakeCohereClient:
    """Stands in for cohere.AsyncClientV2.

    Give it a list of content blocks to reply with, or an exception to raise.
    """

    def __init__(self, blocks=None, raises=None, finish_reason="COMPLETE"):
        self.blocks = blocks
        self.raises = raises
        self.finish_reason = finish_reason
        self.calls = []

    async def chat(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": list(messages), **kwargs})
        if self.raises:
            raise self.raises
        return SimpleNamespace(
            message=SimpleNamespace(role="assistant", content=self.blocks),
            finish_reason=self.finish_reason,
        )


@pytest.fixture
def settings():
    return Settings(cohere_api_key="test-key", cohere_model="test-model")
