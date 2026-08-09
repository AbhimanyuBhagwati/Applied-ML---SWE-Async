import json
import os
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings, get_settings
from app.wikipedia import WikipediaClient


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

    This exists because a setting in a developer's own environment was
    changing test results.
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


def tool_call(name, arguments, call_id="call_0"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeCohereClient:
    """Stands in for cohere.AsyncClientV2.

    Give it content blocks to reply with, or an exception to raise. `script`
    takes a list of turns for the tool loop: each is either a list of tool
    calls or a list of content blocks, replayed one per chat() call.
    """

    def __init__(
        self,
        blocks=None,
        raises=None,
        finish_reason="COMPLETE",
        script=None,
        citations=None,
        tool_finish_reason="TOOL_CALL",
    ):
        # Separate from finish_reason so a test can script a tool-calling turn
        # that Cohere also marked as failed.
        self.tool_finish_reason = tool_finish_reason
        self.blocks = blocks
        self.raises = raises
        self.finish_reason = finish_reason
        self.script = list(script) if script else None
        self.citations = citations
        self.calls = []

    async def chat(self, *, model, messages, tools=None, **kwargs):
        self.calls.append(
            {"model": model, "messages": list(messages), "tools": tools, **kwargs}
        )
        if self.raises:
            raise self.raises

        blocks = self.blocks
        if tools is None and self.script is not None:
            # No tool on offer, so scripted tool turns are skipped to the next
            # text turn, the way a real model would simply answer.
            while self.script and getattr(self.script[0][0], "type", None) == "function":
                self.script.pop(0)

        if self.script is not None:
            step = self.script.pop(0)
            if step and getattr(step[0], "type", None) == "function":
                return SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=step,
                        tool_plan="I will look this up.",
                        citations=None,
                    ),
                    finish_reason=self.tool_finish_reason,
                )
            blocks = step

        return SimpleNamespace(
            message=SimpleNamespace(
                role="assistant",
                content=blocks,
                tool_calls=None,
                tool_plan=None,
                citations=self.citations,
            ),
            finish_reason=self.finish_reason,
        )


@pytest.fixture
def settings():
    return Settings(cohere_api_key="test-key", cohere_model="test-model")


# Two hits, in the shape list=search actually returns, markup and all.
SEARCH_PAYLOAD = {
    "query": {
        "searchinfo": {"totalhits": 5060},
        "search": [
            {
                "ns": 0,
                "title": "Buzz Aldrin",
                "pageid": 4079,
                "wordcount": 12000,
                "timestamp": "2026-01-02T03:04:05Z",
                "snippet": (
                    '<span class="searchmatch">Buzz</span> Aldrin was the '
                    "second person to walk on the Moon &amp; the first to "
                    "hold a religious ceremony there."
                ),
            },
            {
                "ns": 0,
                "title": "Apollo 11",
                "pageid": 1892,
                "wordcount": 20000,
                "timestamp": "2026-01-03T00:00:00Z",
                "snippet": "Apollo 11 was the spaceflight that first landed humans.",
            },
        ],
    }
}


# The second call, prop=extracts, keyed by the pageids above.
EXTRACTS_PAYLOAD = {
    "query": {
        "pages": [
            {
                "pageid": 4079,
                "title": "Buzz Aldrin",
                "extract": (
                    "Buzz Aldrin is an American former astronaut. He was the "
                    "second person to walk on the Moon, after Neil Armstrong."
                ),
            },
            {
                "pageid": 1892,
                "title": "Apollo 11",
                "extract": "Apollo 11 first landed humans on the Moon in 1969.",
            },
        ]
    }
}


@pytest.fixture
def wikipedia_http():
    """httpx client answering both MediaWiki calls from the fixtures above."""
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json=SEARCH_PAYLOAD)
        return httpx.Response(200, json=EXTRACTS_PAYLOAD)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client.recorded_requests = requests
    # Only the search calls, so a test asserting on search parameters doesn't
    # accidentally read the extracts request that follows it.
    client.searches = lambda: [
        r for r in requests if r.url.params.get("list") == "search"
    ]
    return client


@pytest.fixture
def wikipedia(settings, wikipedia_http):
    return WikipediaClient(settings, wikipedia_http)
