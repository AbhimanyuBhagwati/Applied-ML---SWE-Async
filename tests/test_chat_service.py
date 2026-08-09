import asyncio
import json
from types import SimpleNamespace

import cohere
import httpx
import pytest
from cohere.core import ApiError

from app.chat_service import ChatError, ChatService
from tests.conftest import FakeCohereClient, text_block


@pytest.mark.asyncio
async def test_sends_the_query_and_returns_the_reply(settings, wikipedia):
    client = FakeCohereClient([text_block("Buzz Aldrin.")])
    service = ChatService(client, wikipedia, settings)

    result = await service.chat("Who walked on the moon second?")

    assert result.text == "Buzz Aldrin."
    assert result.finish_reason == "COMPLETE"
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "Who walked on the moon second?",
    }
    # The Wikipedia tool is offered on every call; whether it's used is the
    # model's decision.
    assert [t["function"]["name"] for t in client.calls[0]["tools"]] == [
        "search_wikipedia"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocks",
    [[], [SimpleNamespace(type="thinking", text="hmm")]],
    ids=["empty", "reasoning-only"],
)
async def test_a_reply_with_no_text_is_an_error(settings, wikipedia, blocks):
    """Turned up live: the model burned the budget reasoning and said nothing.

    The call succeeded and the answer was empty, which is no use to a caller,
    so it fails here instead of returning 200 with "".
    """
    client = FakeCohereClient(blocks, finish_reason="MAX_TOKENS")

    with pytest.raises(ChatError, match="no text"):
        await ChatService(client, wikipedia, settings).chat("hi")


@pytest.mark.asyncio
async def test_non_text_blocks_are_skipped(settings, wikipedia):
    client = FakeCohereClient(
        [SimpleNamespace(type="thinking", text="hmm"), text_block("Buzz Aldrin.")]
    )

    result = await ChatService(client, wikipedia, settings).chat("hi")

    assert result.text == "Buzz Aldrin."


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["ERROR", "TIMEOUT"])
async def test_failed_finish_reasons_are_errors(settings, wikipedia, reason):
    """A response object exists, but the generation didn't work."""
    client = FakeCohereClient([text_block("partial")], finish_reason=reason)

    with pytest.raises(ChatError, match=reason):
        await ChatService(client, wikipedia, settings).chat("hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["COMPLETE", "MAX_TOKENS"])
async def test_ordinary_finish_reasons_are_fine(settings, wikipedia, reason):
    """MAX_TOKENS is a truncated answer, not a failure. Still a 200."""
    client = FakeCohereClient([text_block("Buzz Aldrin.")], finish_reason=reason)

    result = await ChatService(client, wikipedia, settings).chat("hi")

    assert result.finish_reason == reason


@pytest.mark.asyncio
async def test_api_errors_carry_the_status_but_not_the_body(settings, wikipedia):
    client = FakeCohereClient(
        raises=cohere.core.ApiError(
            status_code=401, body="invalid api token for org acme-corp"
        )
    )

    with pytest.raises(ChatError) as caught:
        await ChatService(client, wikipedia, settings).chat("hi")

    assert caught.value.upstream_status == 401
    # The body can name orgs, keys or internal hosts. It belongs in the log.
    assert "acme-corp" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["Retry-After", "retry-after"])
async def test_retry_after_is_picked_up_whatever_the_casing(settings, wikipedia, header):
    client = FakeCohereClient(
        raises=cohere.core.ApiError(
            status_code=429, body="slow down", headers={header: "30"}
        )
    )

    with pytest.raises(ChatError) as caught:
        await ChatService(client, wikipedia, settings).chat("hi")

    assert caught.value.upstream_status == 429
    assert caught.value.retry_after == "30"


@pytest.mark.asyncio
async def test_network_errors_have_no_upstream_status(settings, wikipedia):
    client = FakeCohereClient(raises=httpx.ConnectError("no route to host"))

    with pytest.raises(ChatError) as caught:
        await ChatService(client, wikipedia, settings).chat("hi")

    assert caught.value.upstream_status is None
    assert "no route to host" not in str(caught.value)


@pytest.mark.asyncio
async def test_logged_bodies_are_flattened_and_truncated(settings, wikipedia, caplog):
    """A body with newlines could otherwise forge extra log lines."""
    body = "line one\nWARNING forged entry\r\nmore\n" + "x" * 500
    client = FakeCohereClient(raises=cohere.core.ApiError(status_code=400, body=body))

    with caplog.at_level("WARNING"), pytest.raises(ChatError):
        await ChatService(client, wikipedia, settings).chat("hi")

    logged = caplog.records[0].getMessage()
    assert "\n" not in logged and "\r" not in logged
    assert len(logged) < 300
    assert logged.endswith("...")


# --- the tool loop (task 2) ---


def searched(query, limit=3, call_id="call_0"):
    from tests.conftest import tool_call

    return tool_call("search_wikipedia", {"query": query, "limit": limit}, call_id)


@pytest.mark.asyncio
async def test_search_results_are_fed_back_into_a_second_call(settings, wikipedia):
    """The whole of task 2: model asks, we search, model answers on the results."""
    client = FakeCohereClient(
        script=[
            [searched("Apollo 11 crew")],
            [text_block("Buzz Aldrin was the second person to walk on the Moon.")],
        ]
    )

    result = await ChatService(client, wikipedia, settings).chat("who was second?")

    assert "Buzz Aldrin" in result.text
    assert len(client.calls) == 2

    # Second call carries the assistant's request and then the tool results.
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_0"
    document = json.loads(tool_message["content"][0]["document"]["data"])
    assert document["title"] == "Buzz Aldrin"
    assert document["url"] == "https://en.wikipedia.org/wiki/Buzz_Aldrin"


@pytest.mark.asyncio
async def test_the_search_is_reported_back_to_the_caller(settings, wikipedia):
    client = FakeCohereClient(
        script=[[searched("Apollo 11 crew")], [text_block("Buzz Aldrin.")]]
    )

    result = await ChatService(client, wikipedia, settings).chat("who was second?")

    assert result.tool_plan == "I will look this up."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_wikipedia"
    assert result.tool_calls[0].arguments == {"query": "Apollo 11 crew", "limit": 3}
    assert result.tool_calls[0].result_count == 2
    assert result.tool_calls[0].results[0]["title"] == "Buzz Aldrin"


@pytest.mark.asyncio
async def test_several_searches_in_one_turn(settings, wikipedia):
    """The model can ask for more than one at a time."""
    client = FakeCohereClient(
        script=[
            [
                searched("Apollo 11", call_id="a"),
                searched("Neil Armstrong", call_id="b"),
            ],
            [text_block("Both were on Apollo 11.")],
        ]
    )

    result = await ChatService(client, wikipedia, settings).chat("tell me about them")

    assert [c.arguments["query"] for c in result.tool_calls] == [
        "Apollo 11",
        "Neil Armstrong",
    ]
    tool_ids = [
        m["tool_call_id"]
        for m in client.calls[1]["messages"]
        if m.__class__ is dict and m.get("role") == "tool"
    ]
    assert tool_ids == ["a", "b"]


@pytest.mark.asyncio
async def test_several_rounds_of_searching(settings, wikipedia):
    """One search wasn't enough, so it goes again before answering."""
    client = FakeCohereClient(
        script=[
            [searched("moon landing")],
            [searched("Buzz Aldrin")],
            [text_block("Buzz Aldrin.")],
        ]
    )

    result = await ChatService(client, wikipedia, settings).chat("who was second?")

    assert result.text == "Buzz Aldrin."
    assert len(result.tool_calls) == 2
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_the_last_call_withdraws_the_tools(settings, wikipedia):
    """Once the rounds are used up the model is made to answer.

    Leaving the tool on offer meant a model that asked again got a 502 instead
    of the answer it had everything it needed to write.
    """
    settings.max_tool_rounds = 2
    client = FakeCohereClient(
        script=[[searched("a")], [searched("b")], [text_block("Buzz Aldrin.")]]
    )

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.text == "Buzz Aldrin."
    assert [call["tools"] is not None for call in client.calls] == [True, True, False]


@pytest.mark.asyncio
async def test_a_model_that_ignores_the_withdrawal_is_cut_off(settings, wikipedia):
    """No tools offered and it asks anyway: nothing left to try but stop.

    The fake is told to disregard the withdrawal, because a well-behaved one
    can't produce this and the loop would otherwise run forever.
    """
    settings.max_tool_rounds = 1

    class Stubborn(FakeCohereClient):
        async def chat(self, **kwargs):
            kwargs["tools"] = [object()]  # pretend the tool was offered
            return await super().chat(**kwargs)

    client = Stubborn(script=[[searched("a")], [searched("b")]])

    with pytest.raises(ChatError, match="would not stop"):
        await ChatService(client, wikipedia, settings).chat("who?")


@pytest.mark.asyncio
async def test_one_round_still_shows_the_model_what_it_found(settings, wikipedia):
    """The off-by-one made this impossible: search, then no chance to use it."""
    settings.max_tool_rounds = 1
    client = FakeCohereClient(
        script=[[searched("Apollo 11")], [text_block("Buzz Aldrin was second.")]]
    )

    result = await ChatService(client, wikipedia, settings).chat("who was second?")

    assert result.text == "Buzz Aldrin was second."
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_a_wikipedia_outage_is_handed_to_the_model_not_the_caller(settings):
    """The model can retry with other terms, or say it doesn't know."""
    from app.wikipedia import WikipediaClient

    def boom(request):
        raise httpx.ConnectError("wikipedia unreachable")

    broken = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(boom))
    )
    client = FakeCohereClient(
        script=[[searched("moon")], [text_block("I could not reach Wikipedia.")]]
    )

    result = await ChatService(client, broken, settings).chat("who was second?")

    assert result.text == "I could not reach Wikipedia."
    assert "error" in result.tool_calls[0].results[0]
    document = json.loads(
        client.calls[1]["messages"][-1]["content"][0]["document"]["data"]
    )
    assert "error" in document


@pytest.mark.asyncio
async def test_no_hits_still_tells_the_model_something(settings):
    """An empty content list is rejected by the API, so it can't just be []."""
    from app.wikipedia import WikipediaClient

    empty = WikipediaClient(
        settings,
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"query": {"search": []}})
            )
        ),
    )
    client = FakeCohereClient(
        script=[[searched("qwertyuiop")], [text_block("Nothing found.")]]
    )

    result = await ChatService(client, empty, settings).chat("who?")

    assert result.tool_calls[0].result_count == 0
    content = client.calls[1]["messages"][-1]["content"]
    assert len(content) == 1
    assert "No Wikipedia articles matched" in content[0]["document"]["data"]


@pytest.mark.asyncio
async def test_unparsable_tool_arguments_dont_crash(settings, wikipedia):
    from tests.conftest import tool_call

    bad = tool_call("search_wikipedia", {})
    bad.function.arguments = "{not json"
    client = FakeCohereClient(script=[[bad], [text_block("Nothing to go on.")]])

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.tool_calls[0].arguments == {}
    # An empty query never reaches Wikipedia, so it comes back as an error.
    assert "error" in result.tool_calls[0].results[0]


@pytest.mark.asyncio
async def test_a_tool_we_dont_have_is_reported_not_raised(settings, wikipedia):
    from tests.conftest import tool_call

    client = FakeCohereClient(
        script=[
            [tool_call("delete_wikipedia", {"query": "x"})],
            [text_block("I can't do that.")],
        ]
    )

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert "No such tool" in result.tool_calls[0].results[0]["error"]


@pytest.mark.asyncio
async def test_the_model_is_given_the_whole_extract(settings):
    """It has to answer from these, so it gets them in full.

    Trimming is the HTTP layer's job. Doing it here would mean the model reads
    the same 200 characters the caller does, which is not enough to answer.
    """
    from app.wikipedia import WikipediaClient

    payload = {"query": {"search": [{"title": "T", "pageid": 1, "snippet": "s"}]}}

    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200, json={"query": {"pages": [{"pageid": 1, "extract": "w" * 1200}]}}
        )

    wiki = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    client = FakeCohereClient(script=[[searched("x")], [text_block("ok")]])

    result = await ChatService(client, wiki, settings).chat("who?")

    document = json.loads(
        client.calls[1]["messages"][-1]["content"][0]["document"]["data"]
    )
    assert len(document["extract"]) == 1200
    assert len(result.tool_calls[0].results[0]["extract"]) == 1200


def test_preview_trims_the_extract_and_leaves_the_rest():
    """The extract is the bulk; the title, url and snippet are the audit trail."""
    from app.chat_service import PREVIEW_CHARS, preview

    trimmed = preview(
        {"title": "T", "url": "u", "snippet": "s" * 300, "extract": "e" * 1200}
    )

    assert trimmed["title"] == "T"
    assert trimmed["url"] == "u"
    assert trimmed["snippet"] == "s" * 300
    assert len(trimmed["extract"]) <= PREVIEW_CHARS + 3
    assert trimmed["extract"].endswith("...")


@pytest.mark.asyncio
async def test_citations_come_back_as_article_urls(settings, wikipedia):
    """Not as the whole document, which is what an earlier version did."""
    source = SimpleNamespace(
        document={
            "title": "Buzz Aldrin",
            "url": "https://en.wikipedia.org/wiki/Buzz_Aldrin",
        },
        tool_output=None,
    )
    citation = SimpleNamespace(start=0, end=11, text="Buzz Aldrin", sources=[source])
    client = FakeCohereClient([text_block("Buzz Aldrin.")], citations=[citation])

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.citations == [
        {
            "start": 0,
            "end": 11,
            "text": "Buzz Aldrin",
            "sources": ["https://en.wikipedia.org/wiki/Buzz_Aldrin"],
        }
    ]


# Cohere doesn't return citation sources in one predictable shape, so each of
# these is a form seen or plausible from the API.
@pytest.mark.parametrize(
    "source,expected",
    [
        (
            SimpleNamespace(document=None, tool_output={"url": "https://w/Buzz"}),
            "https://w/Buzz",
        ),
        (
            SimpleNamespace(
                document={"data": json.dumps({"title": "Apollo 11"})}, tool_output=None
            ),
            "Apollo 11",
        ),
        (
            SimpleNamespace(document={"data": "{not json"}, tool_output=None),
            "wikipedia",
        ),
        (SimpleNamespace(document=None, tool_output=None), "wikipedia"),
    ],
    ids=["tool_output-url", "wrapped-title", "unparsable-data", "nothing-at-all"],
)
def test_citation_sources_never_dump_the_document(source, expected):
    """An earlier version fell back to str(source), which emitted whole articles."""
    from app.chat_service import _source_label

    assert _source_label(source) == expected


@pytest.mark.asyncio
async def test_a_response_asking_for_too_many_searches_is_capped(settings, wikipedia):
    """25 tool calls used to mean 25 searches, so the round cap bounded nothing.

    Every call still gets a tool message back, because the next request is
    rejected otherwise. The ones over the cap get a refusal instead of a search.
    """
    from tests.conftest import tool_call

    settings.max_tool_calls_per_round = 3
    settings.max_total_searches = 100
    many = [tool_call("search_wikipedia", {"query": f"q{i}"}, f"c{i}") for i in range(25)]
    client = FakeCohereClient(script=[many, [text_block("Answer.")]])
    wikipedia._http.recorded_requests.clear()

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert len(result.tool_calls) == 25
    searched_count = sum(1 for c in result.tool_calls if c.result_count > 0)
    assert searched_count == 3
    assert "Not searched" in result.tool_calls[3].results[0]["error"]

    # The protocol still holds: one tool message per call id, in order.
    tool_messages = [
        m
        for m in client.calls[1]["messages"]
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert [m["tool_call_id"] for m in tool_messages] == [f"c{i}" for i in range(25)]


@pytest.mark.asyncio
async def test_wikipedia_errors_reaching_the_caller_carry_no_configuration(settings):
    """These land in tool_calls[].results, which is part of the response."""
    from app.wikipedia import WikipediaClient

    def boom(request):
        raise httpx.ConnectError(
            "connection refused to wiki.internal.corp:8080/w/api.php"
        )

    broken = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(boom))
    )
    client = FakeCohereClient(script=[[searched("moon")], [text_block("No luck.")]])

    result = await ChatService(client, broken, settings).chat("who?")

    error = result.tool_calls[0].results[0]["error"]
    assert "wiki.internal.corp" not in error
    assert "api.php" not in error


def test_citation_offsets_are_rebased_onto_the_joined_answer():
    """Cohere's offsets are per content block; the caller gets one string.

    Without rebasing, a citation from the second block points at whatever
    happens to sit at that offset in the first.
    """
    from app.chat_service import _extract_citations, _flatten

    message = SimpleNamespace(
        content=[
            text_block("Neil Armstrong was first."),
            text_block("Buzz Aldrin was second."),
        ],
        citations=[
            SimpleNamespace(
                start=0, end=14, text="Neil Armstrong", sources=[], content_index=0
            ),
            SimpleNamespace(
                start=0, end=11, text="Buzz Aldrin", sources=[], content_index=1
            ),
        ],
    )

    text, offsets, lead = _flatten(message)
    citations = _extract_citations(message, offsets, lead, text)

    for citation in citations:
        assert text[citation["start"] : citation["end"]] == citation["text"]


def test_empty_blocks_are_skipped_without_shifting_the_offsets():
    """A blank block contributes nothing, so it must not move later citations."""
    from app.chat_service import _extract_citations, _flatten

    message = SimpleNamespace(
        content=[
            text_block("Neil Armstrong was first."),
            text_block(""),
            SimpleNamespace(type="thinking", text="hmm"),
            text_block("Buzz Aldrin was second."),
        ],
        citations=[
            SimpleNamespace(
                start=0, end=11, text="Buzz Aldrin", sources=[], content_index=3
            )
        ],
    )

    text, offsets, lead = _flatten(message)
    citation = _extract_citations(message, offsets, lead, text)[0]

    assert text == "Neil Armstrong was first.\nBuzz Aldrin was second."
    assert text[citation["start"] : citation["end"]] == "Buzz Aldrin"


@pytest.mark.parametrize("kind", ["THINKING_CONTENT", "PLAN"])
def test_citations_about_text_the_caller_never_sees_are_dropped(kind):
    """Those blocks aren't in the response, so their offsets mean nothing here."""
    from app.chat_service import _extract_citations, _flatten

    message = SimpleNamespace(
        content=[text_block("Buzz Aldrin was second.")],
        citations=[
            SimpleNamespace(start=0, end=11, text="reasoning", sources=[], type=kind),
            SimpleNamespace(
                start=0, end=11, text="Buzz Aldrin", sources=[], type="TEXT_CONTENT"
            ),
        ],
    )

    text, offsets, lead = _flatten(message)
    citations = _extract_citations(message, offsets, lead, text)

    assert [c["text"] for c in citations] == ["Buzz Aldrin"]


@pytest.mark.asyncio
async def test_the_total_search_budget_stops_the_loop(settings, wikipedia):
    """Per-round caps multiply out; this is the ceiling for the whole request."""
    settings.max_tool_rounds = 10
    settings.max_tool_calls_per_round = 20
    settings.max_total_searches = 3
    client = FakeCohereClient(
        script=[[searched(f"q{i}")] for i in range(3)] + [[text_block("Answer.")]]
    )

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.text == "Answer."
    assert len([c for c in result.tool_calls if c.result_count > 0]) == 3
    # Once the budget is gone the tool is withheld, so the model answers.
    assert client.calls[-1]["tools"] is None


@pytest.mark.asyncio
async def test_the_time_budget_stops_the_loop(settings, wikipedia):
    """Driven rather than waited on: the fake answers in microseconds.

    This is the soft check between searches, which degrades to an answer from
    what's already gathered. The hard ceiling is the timeout around the loop.
    """
    readings = iter([0.0])  # first sets the deadline, then it's long past

    def clock():
        return next(readings, 100.0)

    settings.tool_loop_seconds = 30.0
    client = FakeCohereClient(script=[[text_block("Answering from memory.")]])
    messages = [{"role": "user", "content": "who?"}]

    result = await ChatService(client, wikipedia, settings)._loop(messages, clock)

    assert result.text == "Answering from memory."
    # Out of time before the first call, so it was never offered the tool.
    assert client.calls[0]["tools"] is None


@pytest.mark.asyncio
async def test_the_hard_ceiling_cancels_work_in_flight(settings, wikipedia):
    """The soft check only runs between searches; this one interrupts."""
    import asyncio

    settings.tool_loop_seconds = 0.05

    class SlowClient(FakeCohereClient):
        async def chat(self, **kwargs):
            await asyncio.sleep(5)
            raise AssertionError("should have been cancelled")

    with pytest.raises(ChatError, match="too long") as caught:
        await ChatService(SlowClient(), wikipedia, settings).chat("who?")

    assert caught.value.timed_out is True


@pytest.mark.asyncio
async def test_a_tool_call_with_no_id_cannot_be_answered(settings, wikipedia):
    """Every tool result names the call it answers; without one, no way back."""
    from tests.conftest import tool_call

    call = tool_call("search_wikipedia", {"query": "moon"})
    call.id = None
    client = FakeCohereClient(script=[[call], [text_block("never reached")]])

    with pytest.raises(ChatError, match="no id"):
        await ChatService(client, wikipedia, settings).chat("who?")


@pytest.mark.asyncio
async def test_a_malformed_sdk_tool_call_doesnt_crash(settings, wikipedia):
    from tests.conftest import tool_call

    call = tool_call("search_wikipedia", {})
    call.function = None
    client = FakeCohereClient(script=[[call], [text_block("Could not search.")]])

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.tool_calls[0].name == "unknown"
    assert result.text == "Could not search."


@pytest.mark.parametrize(
    "citation,why",
    [
        (
            SimpleNamespace(start=0, end=5, text="x", sources=[], content_index=7),
            "points at a block that isn't in the answer",
        ),
        (
            SimpleNamespace(start=None, end=5, text="x", sources=[], content_index=0),
            "has no start",
        ),
        (
            SimpleNamespace(start=0, end=None, text="x", sources=[], content_index=0),
            "has no end",
        ),
    ],
    ids=["unknown-block", "no-start", "no-end"],
)
def test_a_citation_that_cannot_be_placed_is_dropped(citation, why):
    """Better to omit it than to hand back offsets into unrelated words."""
    from app.chat_service import _extract_citations, _flatten

    message = SimpleNamespace(
        content=[text_block("Buzz Aldrin was second.")], citations=[citation]
    )

    text, offsets, lead = _flatten(message)

    assert _extract_citations(message, offsets, lead, text) == [], why


@pytest.mark.asyncio
async def test_a_failed_generation_does_not_get_its_tool_calls_run(settings, wikipedia):
    """Running them first spends real requests on a response we then reject."""
    from tests.conftest import tool_call

    client = FakeCohereClient(
        script=[[tool_call("search_wikipedia", {"query": "x"})]],
        tool_finish_reason="ERROR",
    )
    wikipedia._http.recorded_requests.clear()

    with pytest.raises(ChatError, match="ERROR"):
        await ChatService(client, wikipedia, settings).chat("who?")

    assert wikipedia._http.recorded_requests == []


def test_a_citation_always_matches_the_text_it_points_at():
    """Partially overlapping the stripped whitespace moved the offsets but
    left Cohere's original text, so answer[start:end] != text."""
    from app.chat_service import _extract_citations, _flatten

    message = SimpleNamespace(
        content=[text_block("   Buzz Aldrin was second.")],
        # Starts inside the whitespace and runs into the words.
        citations=[SimpleNamespace(start=1, end=14, text="  Buzz Aldrin", sources=[])],
    )

    text, offsets, lead = _flatten(message)
    citation = _extract_citations(message, offsets, lead, text)[0]

    assert text[citation["start"] : citation["end"]] == citation["text"]


@pytest.mark.asyncio
async def test_a_failed_lookup_is_not_counted_as_a_result(settings):
    """The error document isn't a search hit."""
    broken = __import__("app.wikipedia", fromlist=["WikipediaClient"]).WikipediaClient(
        settings,
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: (_ for _ in ()).throw(httpx.ConnectError("down"))
            )
        ),
    )
    client = FakeCohereClient(script=[[searched("moon")], [text_block("No luck.")]])

    result = await ChatService(client, broken, settings).chat("who?")

    assert result.tool_calls[0].result_count == 0
    assert result.tool_calls[0].arguments == {"query": "moon", "limit": 3}


@pytest.mark.asyncio
async def test_a_refused_call_still_reports_what_was_asked(settings, wikipedia):
    """Blanking the arguments hid which query went unanswered."""
    from tests.conftest import tool_call

    settings.max_tool_calls_per_round = 1
    calls = [
        tool_call("search_wikipedia", {"query": "first"}, "a"),
        tool_call("search_wikipedia", {"query": "second"}, "b"),
    ]
    client = FakeCohereClient(script=[calls, [text_block("ok")]])

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.tool_calls[1].arguments == {"query": "second"}
    assert result.tool_calls[1].result_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"query": "moon", "limit": 3}', {"query": "moon", "limit": 3}),
        ('{"query": 123}', {}),
        ('{"query": "moon", "limit": NaN}', {"query": "moon"}),
        (
            '{"query": "moon", "limit": ' + "9" * 40 + "}",
            {"query": "moon", "limit": 1000},
        ),
    ],
    ids=["good", "wrong-type", "nan", "huge"],
)
async def test_only_primitives_survive_argument_parsing(
    settings, wikipedia, raw, expected
):
    """Filtering field names isn't enough: the values are the model's too.

    A deeply nested value under `query` blew the stack in asdict(), and
    json.loads accepts NaN, which then breaks JSON rendering on the way out.
    """
    from tests.conftest import tool_call

    call = tool_call("search_wikipedia", {})
    call.function.arguments = raw
    client = FakeCohereClient(script=[[call], [text_block("ok")]])

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.tool_calls[0].arguments == expected


@pytest.mark.asyncio
async def test_a_deeply_nested_query_value_is_dropped(settings, wikipedia):
    """asdict() recurses through whatever the model put in there."""
    import json as _json

    from tests.conftest import tool_call

    node = deep = {}
    for _ in range(3000):
        node["x"] = {}
        node = node["x"]

    call = tool_call("search_wikipedia", {})
    call.function.arguments = _json.dumps({"query": deep})
    client = FakeCohereClient(script=[[call], [text_block("ok")]])

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.tool_calls[0].arguments == {}


@pytest.mark.asyncio
async def test_a_lone_surrogate_in_the_query_is_dropped(settings, wikipedia):
    """json.loads accepts "\\ud800"; rendering the response then can't encode it."""
    from tests.conftest import tool_call

    call = tool_call("search_wikipedia", {})
    call.function.arguments = '{"query": "\\ud800"}'
    client = FakeCohereClient(script=[[call], [text_block("ok")]])

    result = await ChatService(client, wikipedia, settings).chat("who?")

    assert result.tool_calls[0].arguments == {}
    __import__("json").dumps(result.tool_calls[0].arguments).encode("utf-8")


@pytest.mark.asyncio
async def test_the_soft_budget_is_reachable_through_chat(settings, wikipedia):
    """Through chat(), not _loop, which is the gap that hid this.

    The soft deadline and the hard asyncio.timeout used to start together with
    the same duration, so the timeout always won and the graceful path could
    never run. The soft one now stops short by one upstream call.
    """
    settings.tool_loop_seconds = 1.0
    settings.request_timeout_seconds = 0.5  # so the soft deadline lands at 0.5
    settings.max_tool_rounds = 10

    class Slow(FakeCohereClient):
        async def chat(self, **kwargs):
            await asyncio.sleep(0.3)
            return await super().chat(**kwargs)

    # A model that would keep searching all day if it were let.
    client = Slow(
        script=[[searched(f"q{i}")] for i in range(5)] + [[text_block("From memory.")]]
    )

    result = await ChatService(client, wikipedia, settings).chat("who?")

    # Without the reserved headroom the soft deadline equals the hard one, so
    # the tool is never withheld and the whole request dies on the 504.
    assert result.text == "From memory."
    assert client.calls[-1]["tools"] is None


@pytest.mark.asyncio
async def test_api_errors_are_caught_however_the_module_was_imported(settings, wikipedia):
    """cohere.core is a lazy attribute of the package.

    `except cohere.core.ApiError` only resolved because a module-level
    annotation happened to touch it first. Any routine refactor, deferred
    annotations for instance, turned the except clause itself into an
    AttributeError and bypassed the entire status mapping.
    """
    import app.chat_service as module

    assert module.ApiError is cohere.core.ApiError

    client = FakeCohereClient(raises=ApiError(status_code=429, body="slow down"))

    with pytest.raises(ChatError) as caught:
        await ChatService(client, wikipedia, settings).chat("hi")

    assert caught.value.upstream_status == 429


@pytest.mark.asyncio
async def test_searches_in_a_round_run_together(settings):
    """Four serial searches is eight round trips against one deadline."""
    from app.wikipedia import WikipediaClient
    from tests.conftest import SEARCH_PAYLOAD, tool_call

    live = 0
    peak = 0

    async def handler(request):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return httpx.Response(200, json=SEARCH_PAYLOAD)

    wiki = WikipediaClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    settings.max_tool_calls_per_round = 4
    calls = [tool_call("search_wikipedia", {"query": f"q{i}"}, f"c{i}") for i in range(4)]
    client = FakeCohereClient(script=[calls, [text_block("done")]])

    result = await ChatService(client, wiki, settings).chat("who?")

    assert len(result.tool_calls) == 4
    assert peak > 1, "the searches ran one after another"
    # Order still matches the calls, because gather preserves it.
    assert [c.arguments["query"] for c in result.tool_calls] == ["q0", "q1", "q2", "q3"]
