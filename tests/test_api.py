import json

import cohere
import pytest
from fastapi.testclient import TestClient

from app.chat_service import ChatService
from app.config import get_settings
from app.main import app, get_chat_service
from tests.conftest import FakeCohereClient, text_block


@pytest.fixture
def make_client(monkeypatch, wikipedia):
    """Boots the app with a fake Cohere client swapped in.

    Both env vars are pinned because Settings also reads .env, and these tests
    shouldn't care what a developer happens to have in theirs.
    """

    def build(fake, api_key="test-key", **env):
        monkeypatch.setenv("COHERE_API_KEY", api_key)
        monkeypatch.setenv("COHERE_MODEL", "test-model")
        for name, value in env.items():
            monkeypatch.setenv(name.upper(), str(value))
        get_settings.cache_clear()
        app.dependency_overrides[get_chat_service] = lambda: ChatService(
            fake, wikipedia, app.state.settings
        )
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_health_when_configured(make_client):
    with make_client(FakeCohereClient([])) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "test-model",
        "cohere_api_key_configured": True,
    }


def test_health_reports_unready_without_a_key(make_client):
    """Liveness alone would keep traffic coming to an instance that can't work."""
    with make_client(FakeCohereClient([]), api_key="") as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["cohere_api_key_configured"] is False


def test_chat_returns_the_model_reply(make_client):
    fake = FakeCohereClient([text_block("Buzz Aldrin.")])

    with make_client(fake) as client:
        response = client.post(
            "/chat", json={"query": "Who was the second person to walk on the moon?"}
        )

    assert response.status_code == 200
    assert response.json() == {
        "query": "Who was the second person to walk on the moon?",
        "response": "Buzz Aldrin.",
        "finish_reason": "COMPLETE",
        "tool_plan": None,
        "tool_calls": [],
        "citations": [],
    }
    assert fake.calls[0]["messages"][-1]["content"].startswith("Who was the second")


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {},
        {"query": "x" * 8001},
        {"query": "   "},
        {"query": "\n\t "},
        {"query": "hi", "temperature": 0.9},
    ],
    ids=["empty", "missing", "long", "spaces", "whitespace", "unknown-field"],
)
def test_bad_queries_are_rejected(make_client, payload):
    with make_client(FakeCohereClient([])) as client:
        assert client.post("/chat", json=payload).status_code == 422


def test_surrounding_whitespace_is_stripped(make_client):
    """Cohere would reject a padded query, so don't spend a call finding out."""
    fake = FakeCohereClient([text_block("Buzz Aldrin.")])

    with make_client(fake) as client:
        body = client.post("/chat", json={"query": "  who walked second?\n"}).json()

    assert fake.calls[0]["messages"][-1]["content"] == "who walked second?"
    assert body["query"] == "who walked second?"


@pytest.mark.parametrize(
    "upstream,expected",
    [(429, 429), (401, 503), (403, 503), (400, 502), (500, 502), (None, 502)],
    ids=[
        "rate-limit",
        "bad-key",
        "forbidden",
        "bad-request",
        "server-error",
        "network",
    ],
)
def test_upstream_failures_map_to_sensible_statuses(make_client, upstream, expected):
    error = (
        cohere.core.ApiError(status_code=upstream, body="secret internal detail")
        if upstream
        else ConnectionError("dns is down")
    )

    with make_client(FakeCohereClient(raises=error)) as client:
        response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == expected
    # Whatever went wrong upstream, the caller gets a fixed string.
    assert "secret internal detail" not in response.text
    assert "dns is down" not in response.text


def test_rate_limits_pass_through_retry_after(make_client):
    error = cohere.core.ApiError(
        status_code=429, body="slow down", headers={"Retry-After": "30"}
    )

    with make_client(FakeCohereClient(raises=error)) as client:
        response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "30"


def test_failed_generation_is_not_a_200(make_client):
    fake = FakeCohereClient([text_block("half an ans")], finish_reason="ERROR")

    with make_client(fake) as client:
        response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == 502


def test_output_tokens_are_capped(make_client):
    """Otherwise one call can run as long as the model feels like."""
    fake = FakeCohereClient([text_block("Buzz Aldrin.")])

    with make_client(fake, max_output_tokens=256) as client:
        client.post("/chat", json={"query": "hi"})

    assert fake.calls[0]["max_tokens"] == 256


def test_missing_api_key_is_a_503(make_client):
    with make_client(FakeCohereClient([]), api_key="") as client:
        response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == 503
    assert "COHERE_API_KEY" in response.json()["detail"]


def test_route_uses_the_service_built_at_startup(monkeypatch):
    """No dependency override, so this exercises the real wiring.

    The other tests replace get_chat_service outright, which means none of them
    would notice if the route stopped reading app.state.
    """
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("COHERE_MODEL", "test-model")
    get_settings.cache_clear()

    with TestClient(app) as client:
        assert isinstance(client.app.state.chat_service, ChatService)
        # Swap the service the lifespan built, then let the request find it the
        # normal way instead of being handed a fake.
        client.app.state.chat_service = ChatService(
            FakeCohereClient([text_block("Buzz Aldrin.")]),
            client.app.state.wikipedia,
            client.app.state.settings,
        )
        response = client.post("/chat", json={"query": "hi"})

    get_settings.cache_clear()
    assert response.status_code == 200
    assert response.json()["response"] == "Buzz Aldrin."


class RecordingClient:
    """Captures how main.py builds the SDK client, and whether it closes it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True


def test_client_is_built_with_a_timeout_and_a_retry_cap(monkeypatch):
    """Left alone the SDK allows 300s and 2 retries, so both are passed."""
    built = []

    def record(**kwargs):
        built.append(RecordingClient(**kwargs))
        return built[-1]

    monkeypatch.setattr(cohere, "AsyncClientV2", record)
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("MAX_RETRIES", "3")
    get_settings.cache_clear()

    with TestClient(app):
        pass

    assert built[0].kwargs["timeout"] == 12.5
    assert built[0].kwargs["max_retries"] == 3
    # Closed on the way out, or the connection pool leaks on every reload.
    assert built[0].closed is True


@pytest.mark.parametrize(
    "query,expected",
    [("a", 200), ("a" * 8000, 200), ("a" * 8001, 422), ("", 422)],
    ids=["one-char", "exactly-8000", "8001", "zero"],
)
def test_query_length_boundaries(make_client, query, expected):
    """Pins both ends of the range, not just the middle."""
    with make_client(FakeCohereClient([text_block("ok")])) as client:
        assert client.post("/chat", json={"query": query}).status_code == expected


def test_the_endpoint_serialises_a_real_tool_round_trip(make_client):
    """Every other API test used a fake that never calls a tool.

    So nothing here ever serialised a populated tool_calls list, and the
    service returning dataclasses while the response model wanted pydantic
    models was a 500 that only showed up against the live API.
    """
    from tests.conftest import tool_call

    fake = FakeCohereClient(
        script=[
            [tool_call("search_wikipedia", {"query": "Apollo 11", "limit": 2})],
            [text_block("Buzz Aldrin was second.")],
        ]
    )

    with make_client(fake) as client:
        response = client.post("/chat", json={"query": "who was second?"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Buzz Aldrin was second."
    assert body["tool_plan"] == "I will look this up."

    call = body["tool_calls"][0]
    assert call["name"] == "search_wikipedia"
    assert call["arguments"] == {"query": "Apollo 11", "limit": 2}
    assert call["result_count"] == 2
    assert call["results"][0]["title"] == "Buzz Aldrin"
    assert call["results"][0]["url"] == "https://en.wikipedia.org/wiki/Buzz_Aldrin"


def test_the_placeholder_user_agent_warns_at_startup(make_client, caplog):
    with caplog.at_level("WARNING"), make_client(FakeCohereClient([])):
        pass

    assert "WIKIPEDIA_USER_AGENT is unset" in caplog.text


def test_the_response_does_not_repeat_whole_articles(make_client):
    """tool_calls was 8KB for a 227 byte answer, and the budget allows eight
    searches. The caller is auditing the grounding, not re-reading Wikipedia."""
    from tests.conftest import tool_call

    fake = FakeCohereClient(
        script=[
            [tool_call("search_wikipedia", {"query": "moon"})],
            [text_block("Buzz Aldrin was second.")],
        ]
    )

    with make_client(fake) as client:
        body = client.post("/chat", json={"query": "who?"}).json()

    for result in body["tool_calls"][0]["results"]:
        # Title, url and snippet survive whole; those are the audit trail.
        assert result["title"]
        assert result["url"]
        assert len(result["extract"]) <= 203

    assert len(json.dumps(body["tool_calls"])) < 2000


def test_a_request_that_runs_too_long_is_a_504(make_client):
    """A gateway timeout, not a 502: nothing upstream refused us."""
    import asyncio

    class SlowClient(FakeCohereClient):
        async def chat(self, **kwargs):
            await asyncio.sleep(5)

    with make_client(SlowClient(), tool_loop_seconds=0.05) as client:
        response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == 504
    assert response.json()["detail"] == "The request took too long."


def test_an_sdk_timeout_is_a_504_not_a_502(make_client):
    """Nothing refused us, it just took too long, same as our own ceiling."""
    import httpx

    with make_client(FakeCohereClient(raises=httpx.ReadTimeout("read timed out"))) as c:
        response = c.post("/chat", json={"query": "hi"})

    assert response.status_code == 504
    assert "read timed out" not in response.text


@pytest.mark.parametrize("status", [408, 504], ids=["request-timeout", "gateway-timeout"])
def test_upstream_saying_it_timed_out_is_a_504(make_client, status):
    """Same thing to a caller as our own ceiling, so not a generic 502."""
    error = cohere.core.ApiError(status_code=status, body="upstream timed out")

    with make_client(FakeCohereClient(raises=error)) as client:
        response = client.post("/chat", json={"query": "hi"})

    assert response.status_code == 504


# --- history (task 3) ---


def test_history_starts_empty(make_client):
    with make_client(FakeCohereClient([])) as client:
        assert client.get("/history").json() == {
            "total": 0,
            "limit": 50,
            "offset": 0,
            "entries": [],
        }


def test_the_history_holds_what_the_caller_was_given(make_client):
    """Round trip through the real endpoints, grounding included."""
    from tests.conftest import tool_call

    fake = FakeCohereClient(
        script=[
            [tool_call("search_wikipedia", {"query": "moon"})],
            [text_block("Buzz Aldrin was second.")],
        ]
    )

    with make_client(fake) as client:
        answered = client.post("/chat", json={"query": "who was second?"}).json()
        entry = client.get("/history").json()["entries"][0]

    for field in ("query", "response", "finish_reason", "tool_plan", "citations"):
        assert entry[field] == answered[field], field
    assert entry["tool_calls"] == answered["tool_calls"]
    assert entry["id"] and entry["created_at"]


def test_history_is_newest_first(make_client):
    with make_client(FakeCohereClient([text_block("a")])) as client:
        for i in range(3):
            client.post("/chat", json={"query": f"Q{i}"})
        body = client.get("/history").json()

    assert [e["query"] for e in body["entries"]] == ["Q2", "Q1", "Q0"]
    assert body["total"] == 3


def test_the_query_is_recorded_as_it_was_used(make_client):
    """Stripped, so the record matches what the model was actually asked."""
    with make_client(FakeCohereClient([text_block("hi")])) as client:
        client.post("/chat", json={"query": "  who was second?\n"})
        entry = client.get("/history").json()["entries"][0]

    assert entry["query"] == "who was second?"


def test_a_rejected_query_is_not_history(make_client):
    """It never reached the model, so there is no response to pair it with."""
    with make_client(FakeCohereClient([text_block("hi")])) as client:
        assert client.post("/chat", json={"query": ""}).status_code == 422
        assert client.get("/history").json()["total"] == 0


def test_a_failed_answer_is_not_history(make_client):
    error = cohere.core.ApiError(status_code=500, body="upstream is unwell")

    with make_client(FakeCohereClient(raises=error)) as client:
        assert client.post("/chat", json={"query": "who?"}).status_code == 502
        assert client.get("/history").json()["total"] == 0


def test_history_pages(make_client):
    with make_client(FakeCohereClient([text_block("a")])) as client:
        for i in range(5):
            client.post("/chat", json={"query": f"Q{i}"})
        first = client.get("/history", params={"limit": 2}).json()
        second = client.get("/history", params={"limit": 2, "offset": 2}).json()
        last = client.get("/history", params={"limit": 2, "offset": 4}).json()

    assert [e["query"] for e in first["entries"]] == ["Q4", "Q3"]
    assert [e["query"] for e in second["entries"]] == ["Q2", "Q1"]
    assert [e["query"] for e in last["entries"]] == ["Q0"]
    # Every page carries the whole total, so a caller can tell there is more.
    assert first["total"] == second["total"] == last["total"] == 5


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"limit": -1}, {"offset": -1}, {"limit": "many"}],
    ids=["zero", "over-max", "negative", "negative-offset", "not-a-number"],
)
def test_bad_paging_is_a_422(make_client, params):
    """The same bounding as the query length, for the same reason."""
    with make_client(FakeCohereClient([])) as client:
        assert client.get("/history", params=params).status_code == 422


def test_reading_history_calls_nobody(make_client, wikipedia_http):
    """It's entirely local. Neither Cohere nor Wikipedia should hear about it."""
    fake = FakeCohereClient([text_block("hi")])

    with make_client(fake) as client:
        client.post("/chat", json={"query": "who?"})
        before = len(fake.calls), len(wikipedia_http.recorded_requests)
        client.get("/history")
        client.get("/history", params={"limit": 1})

    assert (len(fake.calls), len(wikipedia_http.recorded_requests)) == before


def test_history_survives_a_restart_through_the_endpoints(
    tmp_path, monkeypatch, wikipedia
):
    """Two app lifespans over one database file."""
    from app.chat_service import ChatService
    from app.config import get_settings
    from app.main import app, get_chat_service

    database = str(tmp_path / "history.db")

    def boot(fake):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        monkeypatch.setenv("HISTORY_DATABASE_PATH", database)
        get_settings.cache_clear()
        app.dependency_overrides[get_chat_service] = lambda: ChatService(
            fake, wikipedia, app.state.settings
        )
        return TestClient(app)

    with boot(FakeCohereClient([text_block("Buzz Aldrin.")])) as client:
        client.post("/chat", json={"query": "who was second?"})

    with boot(FakeCohereClient([])) as client:
        body = client.get("/history").json()

    app.dependency_overrides.clear()
    get_settings.cache_clear()

    assert body["total"] == 1
    assert body["entries"][0]["query"] == "who was second?"
