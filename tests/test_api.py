import cohere
import pytest
from fastapi.testclient import TestClient

from app.chat_service import ChatService
from app.config import get_settings
from app.main import app, get_chat_service
from tests.conftest import FakeCohereClient, text_block


@pytest.fixture
def make_client(monkeypatch):
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
            fake, app.state.settings
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
    }
    assert fake.calls[0]["messages"][0]["content"].startswith("Who was the second")


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

    assert fake.calls[0]["messages"][0]["content"] == "who walked second?"
    assert body["query"] == "who walked second?"


@pytest.mark.parametrize(
    "upstream,expected",
    [(429, 429), (401, 503), (403, 503), (400, 502), (500, 502), (None, 502)],
    ids=["rate-limit", "bad-key", "forbidden", "bad-request", "server-error", "network"],
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


def test_auth_is_off_unless_a_token_is_configured(make_client):
    with make_client(FakeCohereClient([text_block("hi")])) as client:
        assert client.post("/chat", json={"query": "hi"}).status_code == 200


def test_auth_is_enforced_once_configured(make_client):
    fake = FakeCohereClient([text_block("hi")])

    with make_client(fake, api_auth_token="s3cret") as client:
        missing = client.post("/chat", json={"query": "hi"})
        wrong = client.post(
            "/chat", json={"query": "hi"}, headers={"Authorization": "Bearer nope"}
        )
        right = client.post(
            "/chat", json={"query": "hi"}, headers={"Authorization": "Bearer s3cret"}
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert right.status_code == 200
    # Only the authorised call should have reached Cohere.
    assert len(fake.calls) == 1


def test_rate_limit_is_off_by_default(make_client):
    fake = FakeCohereClient([text_block("hi")])

    with make_client(fake) as client:
        codes = [
            client.post("/chat", json={"query": "hi"}).status_code for _ in range(6)
        ]

    assert codes == [200] * 6


def test_rate_limit_returns_429_with_retry_after(make_client):
    fake = FakeCohereClient([text_block("hi")])

    with make_client(fake, rate_limit_per_minute=2) as client:
        codes = [
            client.post("/chat", json={"query": "hi"}).status_code for _ in range(4)
        ]
        blocked = client.post("/chat", json={"query": "hi"})

    assert codes == [200, 200, 429, 429]
    assert int(blocked.headers["Retry-After"]) > 0
    # Blocked calls must not reach Cohere, which is the whole point.
    assert len(fake.calls) == 2


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
            FakeCohereClient([text_block("Buzz Aldrin.")]), client.app.state.settings
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


@pytest.mark.asyncio
async def test_the_sdk_honours_the_timeout_and_retries_we_pass():
    """Guards the assumption the code rests on.

    Reaches into SDK internals deliberately. The alternative is trusting that
    `timeout=` does something, and an earlier version of this file was built on
    a wrong belief about exactly that.
    """
    async with cohere.AsyncClientV2(api_key="x", timeout=12.5, max_retries=3) as client:
        inner = client._client_wrapper.httpx_client

        assert inner.httpx_client.timeout.read == 12.5
        assert inner.base_max_retries == 3


@pytest.mark.asyncio
async def test_the_sdk_client_closes_itself():
    """The other assumption: __aexit__ shuts down the httpx client it owns."""
    client = cohere.AsyncClientV2(api_key="x")
    inner = client._client_wrapper.httpx_client.httpx_client
    assert not inner.is_closed

    async with client:
        pass

    assert inner.is_closed


@pytest.mark.parametrize(
    "query,expected",
    [("a", 200), ("a" * 8000, 200), ("a" * 8001, 422), ("", 422)],
    ids=["one-char", "exactly-8000", "8001", "zero"],
)
def test_query_length_boundaries(make_client, query, expected):
    """Pins both ends of the range, not just the middle."""
    with make_client(FakeCohereClient([text_block("ok")])) as client:
        assert client.post("/chat", json={"query": query}).status_code == expected


def test_the_limit_counts_every_attempt_not_just_the_ones_that_reach_cohere(make_client):
    """Documented policy, so it gets a test rather than an accident.

    A caller who spends their budget on malformed bodies has spent it. The
    alternative leaves the service floodable with requests that cost it work.
    """
    fake = FakeCohereClient([text_block("ok")])

    with make_client(fake, rate_limit_per_minute=2) as client:
        first = client.post("/chat", json={"query": ""})
        second = client.post("/chat", json={"nope": 1})
        third = client.post("/chat", json={"query": "a real one"})

    assert [first.status_code, second.status_code] == [422, 422]
    assert third.status_code == 429
    assert fake.calls == []


def test_a_bad_token_costs_budget_too(make_client):
    """Checked before auth, so a bad-token flood isn't free."""
    with make_client(FakeCohereClient([]), rate_limit_per_minute=1, api_auth_token="s") as c:
        first = c.post("/chat", json={"query": "hi"}, headers={"Authorization": "Bearer x"})
        second = c.post("/chat", json={"query": "hi"}, headers={"Authorization": "Bearer s"})

    assert first.status_code == 401
    assert second.status_code == 429
