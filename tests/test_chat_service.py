from types import SimpleNamespace

import cohere
import httpx
import pytest

from app.chat_service import ChatError, ChatService
from tests.conftest import FakeCohereClient, text_block


@pytest.mark.asyncio
async def test_sends_the_query_and_returns_the_reply(settings):
    client = FakeCohereClient([text_block("Buzz Aldrin.")])
    service = ChatService(client, settings)

    result = await service.chat("Who walked on the moon second?")

    assert result.text == "Buzz Aldrin."
    assert result.finish_reason == "COMPLETE"
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["messages"] == [
        {"role": "user", "content": "Who walked on the moon second?"}
    ]


@pytest.mark.asyncio
async def test_multiple_content_blocks_are_joined(settings):
    client = FakeCohereClient([text_block("First line."), text_block("Second line.")])

    result = await ChatService(client, settings).chat("hi")

    assert result.text == "First line.\nSecond line."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocks",
    [[], None, [SimpleNamespace(type="thinking", text="hmm")]],
    ids=["empty-list", "missing", "reasoning-only"],
)
async def test_a_reply_with_no_text_is_an_error(settings, blocks):
    """Turned up live: the model burned the budget reasoning and said nothing.

    The call succeeded and the answer was empty, which is no use to a caller,
    so it fails here instead of returning 200 with "".
    """
    client = FakeCohereClient(blocks, finish_reason="MAX_TOKENS")

    with pytest.raises(ChatError, match="no text"):
        await ChatService(client, settings).chat("hi")


@pytest.mark.asyncio
async def test_non_text_blocks_are_skipped(settings):
    client = FakeCohereClient(
        [SimpleNamespace(type="thinking", text="hmm"), text_block("Buzz Aldrin.")]
    )

    result = await ChatService(client, settings).chat("hi")

    assert result.text == "Buzz Aldrin."


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["ERROR", "TIMEOUT"])
async def test_failed_finish_reasons_are_errors(settings, reason):
    """A response object exists, but the generation didn't work."""
    client = FakeCohereClient([text_block("partial")], finish_reason=reason)

    with pytest.raises(ChatError, match=reason):
        await ChatService(client, settings).chat("hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["COMPLETE", "MAX_TOKENS", "STOP_SEQUENCE"])
async def test_ordinary_finish_reasons_are_fine(settings, reason):
    """MAX_TOKENS is a truncated answer, not a failure. Still a 200."""
    client = FakeCohereClient([text_block("Buzz Aldrin.")], finish_reason=reason)

    result = await ChatService(client, settings).chat("hi")

    assert result.finish_reason == reason


@pytest.mark.asyncio
async def test_api_errors_carry_the_status_but_not_the_body(settings):
    client = FakeCohereClient(
        raises=cohere.core.ApiError(
            status_code=401, body="invalid api token for org acme-corp"
        )
    )

    with pytest.raises(ChatError) as caught:
        await ChatService(client, settings).chat("hi")

    assert caught.value.upstream_status == 401
    # The body can name orgs, keys or internal hosts. It belongs in the log.
    assert "acme-corp" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["Retry-After", "retry-after"])
async def test_retry_after_is_picked_up_whatever_the_casing(settings, header):
    client = FakeCohereClient(
        raises=cohere.core.ApiError(
            status_code=429, body="slow down", headers={header: "30"}
        )
    )

    with pytest.raises(ChatError) as caught:
        await ChatService(client, settings).chat("hi")

    assert caught.value.upstream_status == 429
    assert caught.value.retry_after == "30"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers", [{"X-Request-Id": "abc"}, {}, None], ids=["other-headers", "empty", "none"]
)
async def test_no_retry_after_header(settings, headers):
    client = FakeCohereClient(
        raises=cohere.core.ApiError(status_code=429, body="slow down", headers=headers)
    )

    with pytest.raises(ChatError) as caught:
        await ChatService(client, settings).chat("hi")

    assert caught.value.retry_after is None


@pytest.mark.asyncio
async def test_network_errors_have_no_upstream_status(settings):
    client = FakeCohereClient(raises=httpx.ConnectError("no route to host"))

    with pytest.raises(ChatError) as caught:
        await ChatService(client, settings).chat("hi")

    assert caught.value.upstream_status is None
    assert "no route to host" not in str(caught.value)


@pytest.mark.asyncio
async def test_logged_bodies_are_flattened_and_truncated(settings, caplog):
    """A body with newlines could otherwise forge extra log lines."""
    body = "line one\nWARNING forged entry\r\nmore\n" + "x" * 500
    client = FakeCohereClient(raises=cohere.core.ApiError(status_code=400, body=body))

    with caplog.at_level("WARNING"), pytest.raises(ChatError):
        await ChatService(client, settings).chat("hi")

    logged = caplog.records[0].getMessage()
    assert "\n" not in logged and "\r" not in logged
    assert len(logged) < 300
    assert logged.endswith("...")
