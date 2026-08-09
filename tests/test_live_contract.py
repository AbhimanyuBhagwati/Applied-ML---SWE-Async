"""Contract tests against the real Cohere and Wikipedia APIs.

Skipped unless RUN_LIVE_TESTS=1 and a key is present, so the ordinary suite
stays offline and free. Everything else in this repo mocks both providers,
which proves the code behaves as written but not that its assumptions still
match the APIs. These are the ones that would have caught, for instance,
`tool_choice="NONE"` being rejected by this model despite being documented.

    RUN_LIVE_TESTS=1 uv run pytest tests/test_live_contract.py

A trial key allows 20 calls a minute, so running these back to back will
eventually fail with a 429 from Cohere. That is the key's limit, not a defect.
"""

import os

import cohere
import httpx
import pytest

from app.chat_service import ChatService
from app.config import Settings
from app.wikipedia import WIKIPEDIA_TOOL, WikipediaClient

# Read at import, before conftest's isolation fixture strips the environment
# and moves off the directory holding .env. That isolation is right for every
# other test and exactly wrong for these.
# Any non-empty value would be true, so "0" and "false" would have turned
# these on and started spending quota.
LIVE = os.getenv("RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}
REAL = Settings() if LIVE else None

live = pytest.mark.skipif(
    not (LIVE and REAL and REAL.cohere_api_key),
    reason="set RUN_LIVE_TESTS=1, and COHERE_API_KEY in .env or the environment",
)


@pytest.fixture
def settings():
    return REAL


@live
@pytest.mark.asyncio
async def test_the_model_asks_for_the_tool_and_answers_from_it(settings):
    """The whole of task 2 against the real thing."""
    async with (
        httpx.AsyncClient() as http,
        cohere.AsyncClientV2(
            api_key=settings.cohere_api_key, timeout=settings.request_timeout_seconds
        ) as client,
    ):
        service = ChatService(client, WikipediaClient(settings, http), settings)
        result = await service.chat("Who was the second person to walk on the moon?")

    assert result.tool_calls, "the model should have searched"
    assert "Aldrin" in result.text
    assert result.citations, "a grounded answer should cite something"
    for citation in result.citations:
        assert result.text[citation["start"] : citation["end"]] == citation["text"]


@live
@pytest.mark.asyncio
async def test_a_question_that_needs_no_lookup_doesnt_search(settings):
    async with (
        httpx.AsyncClient() as http,
        cohere.AsyncClientV2(
            api_key=settings.cohere_api_key, timeout=settings.request_timeout_seconds
        ) as client,
    ):
        service = ChatService(client, WikipediaClient(settings, http), settings)
        result = await service.chat("Write one short line about rain. Do not search.")

    assert result.tool_calls == []


@live
@pytest.mark.asyncio
async def test_withholding_the_tool_is_accepted_by_the_api(settings):
    """The final round passes tools=None, and the API has to accept that.

    `tool_choice="NONE"` is the documented alternative and is rejected by this
    model with 400, which no mocked test could have told us.
    """
    async with cohere.AsyncClientV2(
        api_key=settings.cohere_api_key, timeout=settings.request_timeout_seconds
    ) as client:
        response = await client.chat(
            model=settings.cohere_model,
            messages=[{"role": "user", "content": "Say OK."}],
            tools=None,
            max_tokens=settings.max_output_tokens,
        )

    assert response.finish_reason not in {"ERROR", "TIMEOUT"}


@live
@pytest.mark.asyncio
async def test_the_tool_schema_is_accepted(settings):
    """A schema the API rejects would fail every request in production."""
    async with cohere.AsyncClientV2(
        api_key=settings.cohere_api_key, timeout=settings.request_timeout_seconds
    ) as client:
        response = await client.chat(
            model=settings.cohere_model,
            messages=[{"role": "user", "content": "Who was Ada Lovelace?"}],
            tools=[WIKIPEDIA_TOOL],
            max_tokens=settings.max_output_tokens,
        )

    assert response.finish_reason not in {"ERROR", "TIMEOUT"}


@pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_TESTS=1 to run")
@pytest.mark.asyncio
async def test_wikipedia_still_returns_the_shape_we_parse():
    """No key needed for this one. Guards against a MediaWiki schema change."""
    settings = Settings(_env_file=None, cohere_api_key="unused")
    async with httpx.AsyncClient() as http:
        results = await WikipediaClient(settings, http).search("Apollo 11", limit=2)

    assert results, "a real search should return something"
    for result in results:
        assert result["title"]
        assert result["url"].startswith("https://en.wikipedia.org/wiki/")
        assert result["extract"], "prop=extracts should have filled this in"
        assert "<span" not in result["snippet"]
