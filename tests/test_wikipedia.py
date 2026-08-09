from datetime import UTC

import httpx
import pytest

from app.wikipedia import WikipediaError


@pytest.mark.asyncio
async def test_search_sends_the_parameters_the_spec_wants(wikipedia, wikipedia_http):
    await wikipedia.search("Nelson Mandela", limit=2)

    params = wikipedia_http.searches()[-1].url.params
    assert params["action"] == "query"
    assert params["list"] == "search"
    assert params["srsearch"] == "Nelson Mandela"
    assert params["srlimit"] == "2"
    assert params["format"] == "json"


@pytest.mark.asyncio
async def test_search_identifies_itself(wikipedia, wikipedia_http):
    """Wikimedia's etiquette guide asks clients to send a real User-Agent."""
    await wikipedia.search("anything")

    assert all(
        "cohere-chat" in r.headers["user-agent"] for r in wikipedia_http.recorded_requests
    )


@pytest.mark.asyncio
async def test_search_returns_the_fields_the_model_needs(wikipedia):
    results = await wikipedia.search("second person on the moon")

    assert [r["title"] for r in results] == ["Buzz Aldrin", "Apollo 11"]
    assert results[0]["url"] == "https://en.wikipedia.org/wiki/Buzz_Aldrin"
    assert results[0]["wordcount"] == 12000
    assert results[0]["last_edited"] == "2026-01-02T03:04:05Z"


@pytest.mark.asyncio
async def test_snippets_lose_their_markup(wikipedia):
    """list=search wraps matches in <span class="searchmatch"> and escapes text."""
    results = await wikipedia.search("buzz")

    snippet = results[0]["snippet"]
    assert "<span" not in snippet and "searchmatch" not in snippet
    assert "&amp;" not in snippet and "Moon & the first" in snippet
    assert snippet.startswith("Buzz Aldrin was the second person")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "title,expected",
    [("Buzz Aldrin", "Buzz_Aldrin"), ("Antarctica/Q&A", "Antarctica/Q%26A")],
    ids=["spaces", "ampersand"],
)
async def test_article_urls_are_encoded(wikipedia, wikipedia_http, title, expected):
    """A raw title with & or ? in it produces a broken link."""
    assert wikipedia._article_url(title).endswith(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,expected", [(99, "5"), (0, "1"), ("two", "3")])
async def test_limit_is_clamped(wikipedia, wikipedia_http, limit, expected):
    """The model picks this, so it can be anything at all."""
    await wikipedia.search("moon", limit=limit)

    assert wikipedia_http.searches()[-1].url.params["srlimit"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["   ", None])
async def test_an_empty_query_never_leaves_the_process(wikipedia, wikipedia_http, query):
    """srsearch is required; a nosrsearch error isn't worth a round trip."""
    with pytest.raises(WikipediaError):
        await wikipedia.search(query)

    assert wikipedia_http.recorded_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="upstream is unwell"),
        httpx.Response(429, text="slow down"),
        httpx.Response(200, text="this is not json"),
    ],
    ids=["server-error", "rate-limited", "not-json"],
)
async def test_bad_responses_become_wikipedia_errors(settings, response):
    from app.wikipedia import WikipediaClient

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: response))
    )

    with pytest.raises(WikipediaError):
        await client.search("anything")


@pytest.mark.asyncio
async def test_a_network_failure_becomes_a_wikipedia_error(settings):
    from app.wikipedia import WikipediaClient

    def boom(request):
        raise httpx.ConnectError("wikipedia unreachable")

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(boom))
    )

    with pytest.raises(WikipediaError) as caught:
        await client.search("anything")

    # The transport exception names the host and path. A misconfigured api_url
    # would leak an internal hostname straight into the response.
    assert "unreachable" not in str(caught.value)
    assert str(caught.value) == "Could not reach Wikipedia."


@pytest.mark.asyncio
async def test_extracts_are_attached_to_the_matching_hits(wikipedia):
    """The snippet says why it matched; the extract says what it contains."""
    results = await wikipedia.search("second person on the moon")

    assert "second person to walk on the Moon" in results[0]["extract"]
    assert results[1]["extract"].startswith("Apollo 11 first landed humans")


@pytest.mark.asyncio
async def test_a_failed_extract_call_doesnt_lose_the_search(settings):
    """The search worked. Thin results beat no results."""
    from app.wikipedia import WikipediaClient
    from tests.conftest import SEARCH_PAYLOAD

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json=SEARCH_PAYLOAD)
        return httpx.Response(500, text="extracts are down")

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    results = await client.search("moon")

    assert [r["title"] for r in results] == ["Buzz Aldrin", "Apollo 11"]
    assert all(r["extract"] == "" for r in results)
    assert results[0]["snippet"].startswith("Buzz Aldrin was the second")


@pytest.mark.asyncio
async def test_no_hits_means_no_second_request(settings):
    """Nothing to fetch extracts for, so don't ask."""
    from app.wikipedia import WikipediaClient

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"query": {"search": []}})

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    assert await client.search("qwertyuiop") == []
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_long_extracts_are_truncated_on_a_word_boundary(settings):
    from app.wikipedia import WikipediaClient
    from tests.conftest import SEARCH_PAYLOAD

    settings.wikipedia_extract_chars = 40

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json=SEARCH_PAYLOAD)
        return httpx.Response(
            200,
            json={"query": {"pages": [{"pageid": 4079, "extract": "word " * 100}]}},
        )

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    extract = (await client.search("moon"))[0]["extract"]

    assert extract.endswith("...")
    assert len(extract) <= 43
    assert "wor..." not in extract


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        "a string",
        {"query": None},
        {"query": "not a dict"},
        {"query": {"search": None}},
        {"query": {}},
    ],
    ids=[
        "top-list",
        "top-string",
        "query-null",
        "query-wrong-type",
        "search-null",
        "no-key",
    ],
)
async def test_a_broken_response_raises(settings, payload):
    """Accepting either an error or empty results would enforce nothing."""
    from app.wikipedia import WikipediaClient

    client = WikipediaClient(
        settings,
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ),
    )

    with pytest.raises(WikipediaError):
        await client.search("anything")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"query": {"search": []}}], ids=["search-empty"])
async def test_a_genuinely_empty_result_is_an_empty_list(settings, payload):
    """The spec says a real empty result is query.search: [].

    Anything else missing means an invalid response or a schema change, and
    saying "nothing matched" would be a claim about the world we can't make.
    """
    from app.wikipedia import WikipediaClient

    client = WikipediaClient(
        settings,
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ),
    )

    assert await client.search("anything") == []


@pytest.mark.asyncio
async def test_junk_entries_in_a_valid_list_are_skipped(settings):
    """The list is the right type, the things in it are not."""
    from app.wikipedia import WikipediaClient

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(
                200,
                json={
                    "query": {"search": [None, "nope", {"title": "Real", "pageid": 1}]}
                },
            )
        return httpx.Response(200, json={"query": {"pages": []}})

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    assert [r["title"] for r in await client.search("anything")] == ["Real"]


@pytest.mark.asyncio
async def test_extracts_are_batched_because_the_api_caps_them_at_20(settings):
    """TextExtracts returns at most 20 per request and drops the rest silently.

    With srlimit up to 50, results 21 onwards would come back with no extract.
    """
    from app.wikipedia import EXTRACT_BATCH, WikipediaClient

    settings.wikipedia_max_results = 50
    hits = [
        {"title": f"Article {i}", "pageid": i, "snippet": "s", "wordcount": 1}
        for i in range(1, 26)
    ]
    seen_batches = []

    def handler(request):
        params = request.url.params
        if params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": hits}})
        ids = params["pageids"].split("|")
        seen_batches.append(len(ids))
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [{"pageid": int(i), "extract": f"Body of {i}"} for i in ids]
                }
            },
        )

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    results = await client.search("many", limit=25)

    assert seen_batches == [EXTRACT_BATCH, 5]
    assert all(r["extract"] for r in results), "every hit should have its extract"


@pytest.mark.asyncio
async def test_extract_length_is_bounded_upstream_too(settings):
    """exchars caps the response at the API, so we don't download and discard."""
    from app.wikipedia import MAX_EXCHARS, WikipediaClient
    from tests.conftest import EXTRACTS_PAYLOAD, SEARCH_PAYLOAD

    settings.wikipedia_extract_chars = 9000  # over the API's own ceiling
    seen = {}

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json=SEARCH_PAYLOAD)
        seen.update(request.url.params)
        return httpx.Response(200, json=EXTRACTS_PAYLOAD)

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await client.search("moon")

    assert int(seen["exchars"]) == MAX_EXCHARS
    assert int(seen["exlimit"]) == 20


@pytest.mark.asyncio
async def test_a_hit_with_no_title_is_dropped(settings):
    """Otherwise it becomes an article called "" linking to the front page."""
    from app.wikipedia import WikipediaClient

    payload = {
        "query": {
            "search": [
                {"title": "", "pageid": 1, "snippet": "s"},
                {"title": None, "pageid": 2, "snippet": "s"},
                {"pageid": 3, "snippet": "s"},
                {"title": "Real Article", "pageid": 4, "snippet": "s"},
            ]
        }
    }

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"query": {"pages": []}})

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    results = await client.search("anything")

    assert [r["title"] for r in results] == ["Real Article"]
    assert not any(r["url"] == settings.wikipedia_article_url for r in results)


@pytest.mark.asyncio
async def test_wikipedia_log_lines_are_flattened_and_bounded(settings, caplog):
    from app.wikipedia import WikipediaClient

    def boom(request):
        raise httpx.ConnectError("line one\nline two\r\n" + "x" * 500)

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(boom))
    )

    with caplog.at_level("WARNING"), pytest.raises(WikipediaError):
        await client.search("anything")

    logged = caplog.records[0].getMessage()
    assert "\n" not in logged and "\r" not in logged
    assert len(logged) < 300


@pytest.mark.asyncio
async def test_a_429_is_actually_honoured_not_just_reported(settings):
    """Telling the model to wait isn't enforcement; it asks again next round."""
    from app.wikipedia import WikipediaClient

    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(429, text="slow", headers={"Retry-After": "30"})

    client = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(WikipediaError, match="rate limiting"):
        await client.search("first")
    with pytest.raises(WikipediaError, match="rate limiting"):
        await client.search("second")

    # The second call never left the process.
    assert len(attempts) == 1


def make_client(settings, handler):
    from app.wikipedia import WikipediaClient

    return WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.mark.asyncio
async def test_the_longest_outstanding_backoff_wins(settings):
    """Taking the newest let a 1 second 429 cancel a 5 minute one."""
    waits = iter(["300", "1"])
    client = make_client(
        settings,
        lambda r: httpx.Response(429, headers={"Retry-After": next(waits, "1")}),
    )

    with pytest.raises(WikipediaError):
        await client.search("a")
    blocked_after_long = client._blocked_until
    client._blocked_until = 0.0  # let one more request through
    with pytest.raises(WikipediaError):
        await client.search("b")

    assert client._blocked_until < blocked_after_long
    client._blocked_until = blocked_after_long
    with pytest.raises(WikipediaError):
        await client.search("c")
    assert client._blocked_until == blocked_after_long


@pytest.mark.asyncio
async def test_an_http_date_retry_after_is_understood(settings):
    """RFC 9110 allows a date as well as seconds, and both turn up."""
    from datetime import datetime, timedelta
    from email.utils import format_datetime

    when = format_datetime(datetime.now(UTC) + timedelta(seconds=120))
    client = make_client(
        settings, lambda r: httpx.Response(429, headers={"Retry-After": when})
    )

    with pytest.raises(WikipediaError) as caught:
        await client.search("a")

    # Roughly two minutes, not the fixed fallback.
    assert "119 seconds" in str(caught.value) or "120 seconds" in str(caught.value)


@pytest.mark.asyncio
async def test_mediawiki_throttling_arrives_as_a_200(settings):
    """MediaWiki reports it in the body, so the status check never sees it."""
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(
            200, json={"error": {"code": "ratelimited", "info": "slow"}}
        )

    client = make_client(settings, handler)

    with pytest.raises(WikipediaError, match="rate limiting"):
        await client.search("a")
    with pytest.raises(WikipediaError, match="rate limiting"):
        await client.search("b")

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_a_schema_change_in_extracts_keeps_the_snippets(settings):
    """The README promises snippet-only degradation; parsing sat outside the try."""
    from tests.conftest import SEARCH_PAYLOAD

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json=SEARCH_PAYLOAD)
        return httpx.Response(200, json={"query": {"pages": "schema changed"}})

    results = await make_client(settings, handler).search("moon")

    assert [r["title"] for r in results] == ["Buzz Aldrin", "Apollo 11"]
    assert results[0]["snippet"].startswith("Buzz Aldrin was the second")
    assert all(r["extract"] == "" for r in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected",
    [
        ({"code": "search-error", "info": "internal host said no"}, "search-error"),
        ({"info": "no code at all"}, "unknown"),
        ("just a string", "refused"),
    ],
    ids=["with-code", "no-code", "not-an-object"],
)
async def test_a_mediawiki_error_arrives_as_a_200(settings, error, expected):
    """The status check never sees it, and `info` can quote config back."""
    client = make_client(settings, lambda r: httpx.Response(200, json={"error": error}))

    with pytest.raises(WikipediaError, match=expected) as caught:
        await client.search("anything")

    assert "internal host" not in str(caught.value)


@pytest.mark.asyncio
async def test_the_cooldown_gates_every_request_not_just_the_first(settings):
    """A 429 on the search must stop the extract call that follows it."""
    from tests.conftest import SEARCH_PAYLOAD

    seen = []

    def handler(request):
        seen.append(request.url.params.get("list") or "extracts")
        if len(seen) == 1:
            return httpx.Response(200, json=SEARCH_PAYLOAD)
        return httpx.Response(429, headers={"Retry-After": "60"})

    client = make_client(settings, handler)

    # First search succeeds, its extract call is refused and installs a block.
    await client.search("first")
    with pytest.raises(WikipediaError, match="rate limiting"):
        await client.search("second")

    # The second search never reached the transport.
    assert seen == ["search", "extracts"]


@pytest.mark.asyncio
async def test_a_503_backs_off_too(settings):
    """Wikimedia documents Retry-After on 503 as well as 429."""
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, headers={"Retry-After": "90"})

    client = make_client(settings, handler)

    with pytest.raises(WikipediaError, match="rate limiting"):
        await client.search("a")
    with pytest.raises(WikipediaError, match="rate limiting"):
        await client.search("b")

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_a_200_throttle_uses_its_own_retry_after(settings):
    """It arrives with a header like any other throttle response."""
    client = make_client(
        settings,
        lambda r: httpx.Response(
            200,
            json={"error": {"code": "maxlag", "info": "lagged"}},
            headers={"Retry-After": "120"},
        ),
    )

    with pytest.raises(WikipediaError) as caught:
        await client.search("a")

    assert "120 seconds" in str(caught.value)


@pytest.mark.parametrize(
    "header",
    ["²", "Mon, 01 Jan 999999999 00:00:00 GMT", "1e5", None],
    ids=["superscript-two", "impossible-year", "exponent", "absent"],
)
def test_a_malformed_retry_after_never_escapes(header):
    """Parsed directly: httpx encodes outgoing headers as ASCII, so some of
    these can't be built into a mock response even though a real server can
    send the bytes and httpx will decode them as latin-1.

    "superscript two" passes isdigit() and then float() rejects it, and an
    implausible year overflows rather than parsing. Both escaped raw before.
    """
    from app.wikipedia import DEFAULT_BACKOFF, MAX_BACKOFF, _retry_seconds

    seconds = _retry_seconds(header)

    assert isinstance(seconds, float)
    assert 0 < seconds <= MAX_BACKOFF
    assert seconds == DEFAULT_BACKOFF


@pytest.mark.asyncio
async def test_more_hits_than_asked_for_are_discarded(settings):
    """A custom or broken endpoint ignoring srlimit would otherwise walk past
    the result, request and context bounds all at once."""
    hits = [{"title": f"A{i}", "pageid": i, "snippet": "s"} for i in range(1, 46)]
    batches = []

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": hits}})
        batches.append(len(request.url.params["pageids"].split("|")))
        return httpx.Response(200, json={"query": {"pages": []}})

    results = await make_client(settings, handler).search("many", limit=5)

    assert len(results) == 5
    assert batches == [5]


def test_a_naive_http_date_is_treated_as_utc():
    """Some servers omit the zone; assuming local time would shift the wait."""
    from datetime import datetime, timedelta

    from app.wikipedia import _retry_seconds

    when = (datetime.now(UTC) + timedelta(seconds=90)).strftime("%a, %d %b %Y %H:%M:%S")

    assert 80 < _retry_seconds(when) <= 95


@pytest.mark.asyncio
async def test_concurrent_requests_are_capped(settings):
    """Wikimedia asks clients to keep to a handful at a time.

    The handler awaits on purpose. A synchronous mock completes each request
    before the next is scheduled, so nothing ever overlaps and the test proves
    nothing.
    """
    import asyncio

    from app.wikipedia import MAX_CONCURRENCY

    live = 0
    peak = 0

    async def handler(request):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return httpx.Response(200, json={"query": {"search": []}})

    client = make_client(settings, handler)
    await asyncio.gather(*(client.search(f"q{i}") for i in range(12)))

    assert peak <= MAX_CONCURRENCY, f"{peak} at once, limit is {MAX_CONCURRENCY}"


@pytest.mark.asyncio
async def test_queued_callers_recheck_the_cooldown(settings):
    """A 429 mid-queue has to stop the callers already waiting for a permit.

    They passed the cooldown check while the coast was clear. Checking only
    before the semaphore let all twelve drain through after the block was
    installed; the fix is to recheck after taking a permit.
    """
    import asyncio

    reached = []

    async def handler(request):
        reached.append(request)
        await asyncio.sleep(0.01)
        return httpx.Response(429, headers={"Retry-After": "60"})

    client = make_client(settings, handler)
    results = await asyncio.gather(
        *(client.search(f"q{i}") for i in range(12)), return_exceptions=True
    )

    assert all(isinstance(r, WikipediaError) for r in results)
    # Only the ones already in flight when the 429 landed get through.
    assert len(reached) <= 3, f"{len(reached)} of 12 reached Wikipedia"
