"""Talking to the Cohere Chat API."""

import logging
from dataclasses import dataclass
from typing import Any

import cohere
from cohere.core import ApiError

from app.config import Settings

log = logging.getLogger(__name__)

# ApiError is imported by name rather than reached through cohere.core.
# The SDK resolves its submodules lazily, so `cohere.core` only exists
# once something has touched it. Reaching for it inside an except clause
# raises AttributeError from the clause itself, and every API error then
# goes uncaught instead of being mapped to a status.

# Provider bodies can reflect the input back, name organisations, or run to
# pages of diagnostics. Enough to identify the failure, not a transcript.
LOGGED_BODY_CHARS = 200

# The SDK's own list. MAX_TOKENS and STOP_SEQUENCE are ordinary endings, these
# two mean the generation did not work.
FAILED_FINISH_REASONS = {"ERROR", "TIMEOUT"}


class ChatError(RuntimeError):
    """The upstream call failed.

    `upstream_status` is Cohere's HTTP status when we got a response at all, and
    None when we never got that far. The HTTP layer decides what to turn that
    into. Messages here are for logs, so they must not carry response bodies.
    """

    def __init__(
        self,
        message: str,
        upstream_status: int | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status
        self.retry_after = retry_after


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None = None


class ChatService:
    def __init__(self, client: cohere.AsyncClientV2, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def chat(self, query: str) -> ChatResult:
        try:
            response = await self._client.chat(
                model=self._settings.cohere_model,
                messages=[{"role": "user", "content": query}],
                max_tokens=self._settings.max_output_tokens,
            )
        except ApiError as exc:
            # The body carries provider diagnostics, so it goes to the log and
            # not into the exception the HTTP layer will render.
            log.warning("Cohere returned %s: %s", exc.status_code, _for_log(exc.body))
            raise ChatError(
                f"Cohere returned {exc.status_code}",
                upstream_status=exc.status_code,
                retry_after=_retry_after(exc.headers),
            ) from exc
        except Exception as exc:
            # Deliberately broad. Underneath the SDK is httpx, so a DNS failure,
            # a dropped connection or a timeout all surface as their own types.
            # Same treatment as a provider body. A transport exception can be
            # multiline and can quote a URL or host back at you.
            log.warning("Cohere request failed: %s", _for_log(exc))
            raise ChatError("could not reach Cohere") from exc

        if response.finish_reason in FAILED_FINISH_REASONS:
            log.warning("Cohere finished with %s", response.finish_reason)
            raise ChatError(f"generation failed with {response.finish_reason}")

        text = _extract_text(response.message)
        if not text:
            # Seen live: the model spent the whole token budget on reasoning and
            # left nothing in the text blocks, finishing MAX_TOKENS. The call
            # succeeded and the answer is empty, which is no use to the caller,
            # so it's a failure here rather than a 200 carrying "".
            log.warning("Cohere returned no text, finish_reason=%s", response.finish_reason)
            raise ChatError(f"no text in response, finish_reason={response.finish_reason}")

        return ChatResult(text=text, finish_reason=response.finish_reason)


def _for_log(body: Any) -> str:
    """Make a provider body safe to put in a log line.

    Truncated, because a log still ends up shipped somewhere. Flattened,
    because a body containing newlines can otherwise forge extra log entries,
    and the body is one of the few places a caller's own input comes back.
    """
    text = str(body)[:LOGGED_BODY_CHARS]
    flattened = " ".join(text.split())
    return flattened + "..." if len(str(body)) > LOGGED_BODY_CHARS else flattened


def _retry_after(headers: dict[str, str] | None) -> str | None:
    """Find Retry-After in a plain dict, where the casing isn't guaranteed."""
    for name, value in (headers or {}).items():
        if name.lower() == "retry-after":
            return value
    return None


def _extract_text(message: Any) -> str:
    """Flatten the assistant message's content blocks into one string.

    v2 returns content as a list of blocks rather than a plain string, and it
    can be empty or missing if the model produced nothing.
    """
    blocks = getattr(message, "content", None) or []
    parts = [
        block.text
        for block in blocks
        if getattr(block, "type", "text") == "text" and getattr(block, "text", None)
    ]
    return "\n".join(parts).strip()
