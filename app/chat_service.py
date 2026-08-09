"""Talking to the Cohere Chat API, including the Wikipedia tool loop."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cohere
import httpx
from cohere.core import ApiError

from app.config import Settings
from app.wikipedia import (
    DEFAULT_RESULTS,
    WIKIPEDIA_TOOL,
    WikipediaClient,
    WikipediaError,
)

log = logging.getLogger(__name__)

# ApiError is imported by name rather than reached through cohere.core: the
# SDK resolves submodules lazily, so the dotted form in an except clause works
# only while something else happens to have imported it first.

# Only str and int survive parsing. Keeping the model's own values would let a
# deeply nested object blow the stack in asdict(), and a NaN through
# json.loads() break JSON rendering on the way out.
MAX_QUERY_CHARS = 500
# Separate from PREVIEW_CHARS: unrelated concerns shouldn't share a number.
ERROR_CHARS = 200
MAX_LIMIT = 1_000

SYSTEM_PROMPT = (
    "You are a research assistant. For questions about real-world facts, "
    "search Wikipedia rather than answering from memory, and base your answer "
    "on what comes back. If the search returns nothing useful, say so instead "
    "of guessing. Keep answers short."
)

# How much of each result the caller gets back. The model needs the whole
# extract to answer from; the caller is checking what grounded the answer, and
# the title, url and snippet already say that. Sending the extracts too made
# tool_calls 8KB for a 227 byte answer, and the budget allows eight searches.
PREVIEW_CHARS = 200

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
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status
        self.retry_after = retry_after
        self.timed_out = timed_out


@dataclass
class Budget:
    """What one /chat request may spend on searching.

    Per-round caps multiply rather than bound, so these apply to the request as
    a whole. Context size isn't tracked: it can't be known before fetching, so
    it's bounded statically by searches x results x extract_chars instead.
    """

    searches_left: int
    deadline: float
    # Injected so tests can drive it. Patching time.monotonic globally would
    # also move asyncio's loop clock, and with it the hard timeout.
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def start(
        cls, settings: Settings, clock: Callable[[], float] = time.monotonic
    ) -> "Budget":
        # Short of the hard ceiling by one upstream call. Given the whole
        # window, this deadline and the asyncio.timeout around the loop start
        # together and expire together, so the timeout always wins and the
        # graceful path, stop searching and answer from what we have, can
        # never run.
        reserved = min(settings.request_timeout_seconds, settings.tool_loop_seconds / 2)
        return cls(
            searches_left=settings.max_total_searches,
            deadline=clock() + settings.tool_loop_seconds - reserved,
            clock=clock,
        )

    def spent(self) -> str | None:
        """Why we should stop searching, or None to carry on."""
        if self.clock() >= self.deadline:
            return "the time budget ran out"
        if self.searches_left <= 0:
            return "the search budget ran out"
        return None

    def charge(self) -> None:
        self.searches_left -= 1


@dataclass
class ToolCallRecord:
    """A search the model asked for, and what it got back."""

    name: str
    arguments: dict[str, Any]
    result_count: int
    results: list[dict[str, Any]]


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None = None
    tool_plan: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        client: cohere.AsyncClientV2,
        wikipedia: WikipediaClient,
        settings: Settings,
    ) -> None:
        self._client = client
        self._wikipedia = wikipedia
        self._settings = settings

    async def chat(self, query: str) -> ChatResult:
        """Answer the query, letting the model search Wikipedia first.

        The loop is: ask the model, and if it wants a search, run it, hand the
        results back and ask again. Most questions take one round; the cap
        stops a model that keeps asking forever.
        """
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            async with asyncio.timeout(self._settings.tool_loop_seconds):
                return await self._loop(messages)
        except TimeoutError as exc:
            log.warning("Tool loop hit the %ss ceiling", self._settings.tool_loop_seconds)
            raise ChatError("the request took too long", timed_out=True) from exc

    async def _loop(
        self, messages: list[Any], clock: Callable[[], float] = time.monotonic
    ) -> ChatResult:
        tool_calls: list[ToolCallRecord] = []
        tool_plan: str | None = None
        rounds = self._settings.max_tool_rounds
        budget = Budget.start(self._settings, clock)
        searched = 0

        # Rounds count searches, not Cohere calls: there is always one more
        # call after the last round, to turn the results into an answer.
        while True:
            # Withheld on the last call so the model answers rather than
            # asking for a search we would refuse. tool_choice="NONE" is the
            # documented alternative; this model returns 400 for it.
            last = searched >= rounds or budget.spent() is not None
            response = await self._call_cohere(messages, allow_tools=not last)
            message = response.message

            # First: a failed generation's tool calls aren't worth running.
            if response.finish_reason in FAILED_FINISH_REASONS:
                log.warning("Cohere finished with %s", response.finish_reason)
                raise ChatError(f"generation failed with {response.finish_reason}")

            if not getattr(message, "tool_calls", None):
                return self._final_result(response, tool_plan, tool_calls)

            if last:
                # Asked for a tool when none were offered. Nothing sensible is
                # left to do, and continuing would loop forever.
                log.warning("Model asked for a tool after tools were withdrawn")
                raise ChatError("model would not stop asking for searches")

            searched += 1

            tool_plan = getattr(message, "tool_plan", None) or tool_plan
            messages.append(message)

            # Every tool call needs a tool message back or the next request
            # is rejected, so calls we decline still get an answer saying so.
            allowed = self._settings.max_tool_calls_per_round
            plan: list[tuple[str, Any, str | None]] = []
            for index, call in enumerate(message.tool_calls):
                call_id = getattr(call, "id", None)
                if not isinstance(call_id, str) or not call_id:
                    # Every tool result has to name the call it answers, so
                    # there is no way to continue the conversation without it.
                    raise ChatError("Cohere sent a tool call with no id")

                if index >= allowed:
                    refusal = f"at most {allowed} searches per turn"
                else:
                    refusal = budget.spent()
                if refusal is None:
                    budget.charge()
                plan.append((call_id, call, refusal))

            # The searches in a round are independent, so they run together.
            # Serially, four calls is eight sequential round trips against the
            # same deadline. The client's own semaphore bounds the fan-out.
            done = await asyncio.gather(
                *(self._run_search(call) for _, call, refusal in plan if refusal is None)
            )
            results = iter(done)

            for call_id, call, refusal in plan:
                if refusal is None:
                    record = next(results)
                else:
                    log.warning("Not searching: %s", refusal)
                    record = ToolCallRecord(
                        name=_tool_name(call),
                        # Kept, so the response says which query went unrun.
                        arguments=_tool_arguments(call),
                        result_count=0,
                        results=[
                            {
                                "error": f"Not searched: {refusal}. Answer with "
                                "what you already have."
                            }
                        ],
                    )

                tool_calls.append(record)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _as_documents(record.results),
                    }
                )

    def _final_result(
        self, response: Any, tool_plan: str | None, tool_calls: list[ToolCallRecord]
    ) -> ChatResult:
        # finish_reason is checked by the caller, before any tool calls run.
        text, offsets, lead = _flatten(response.message)
        if not text:
            # A successful call with no text is still no answer.
            log.warning(
                "Cohere returned no text, finish_reason=%s", response.finish_reason
            )
            raise ChatError(
                f"no text in response, finish_reason={response.finish_reason}"
            )

        return ChatResult(
            text=text,
            finish_reason=response.finish_reason,
            tool_plan=tool_plan,
            tool_calls=tool_calls,
            citations=_extract_citations(response.message, offsets, lead, text),
        )

    async def _run_search(self, call: Any) -> ToolCallRecord:
        name = _tool_name(call)
        arguments = _tool_arguments(call)

        try:
            if name != WIKIPEDIA_TOOL["function"]["name"]:
                raise WikipediaError(f"No such tool: {name}")
            results = await self._wikipedia.search(
                query=arguments.get("query", ""),
                limit=arguments.get("limit", DEFAULT_RESULTS),
            )
        except WikipediaError as exc:
            # Don't fail the request. Tell the model the search failed and let
            # it try different terms or answer without it.
            log.warning("Search failed: %s", _for_log(exc))
            results = [{"error": str(exc)[:ERROR_CHARS]}]
            return ToolCallRecord(
                name=name, arguments=arguments, result_count=0, results=results
            )

        return ToolCallRecord(
            name=name,
            arguments=arguments,
            result_count=len(results),
            # Held in full: this is what the model was given, and it's what the
            # context budget is charged for. The HTTP layer trims it.
            results=results,
        )

    async def _call_cohere(self, messages: list[Any], allow_tools: bool = True) -> Any:
        try:
            return await self._client.chat(
                model=self._settings.cohere_model,
                messages=messages,
                tools=[WIKIPEDIA_TOOL] if allow_tools else None,
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
        except httpx.TimeoutException as exc:
            # Same thing from the caller's side as our own ceiling: nothing
            # refused us, it just took too long.
            log.warning("Cohere timed out: %s", _for_log(exc))
            raise ChatError("Cohere timed out", timed_out=True) from exc
        except Exception as exc:
            # Deliberately broad. Underneath the SDK is httpx, so a DNS failure
            # or a dropped connection surfaces as its own type.
            log.warning("Cohere request failed: %s", _for_log(exc))
            raise ChatError("could not reach Cohere") from exc


def preview(result: dict[str, Any]) -> dict[str, Any]:
    """A search hit cut down to what a caller needs to audit the answer.

    Titles, urls and snippets stay whole. The extract is what the model read,
    and repeating it all makes the response many times the size of the answer.
    """
    extract = result.get("extract")
    if not isinstance(extract, str) or len(extract) <= PREVIEW_CHARS:
        return result
    return {**result, "extract": extract[:PREVIEW_CHARS].rstrip() + "..."}


def _as_documents(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap tool results as document blocks so the model can cite them."""
    # An empty content list is rejected, so an empty result set still has to
    # say something.
    if not results:
        results = [{"note": "No Wikipedia articles matched that search."}]
    return [
        {"type": "document", "document": {"data": json.dumps(result)}}
        for result in results
    ]


def _extract_citations(
    message: Any, offsets: dict[int, int], lead: int, answer: str
) -> list[dict[str, Any]]:
    """The spans the model cited, rebased onto the joined answer.

    Cohere's offsets are into one content block, named by `content_index`. A
    citation that can't be placed is dropped: THINKING_CONTENT and PLAN refer
    to blocks that aren't in the response.
    """
    citations = []
    for citation in getattr(message, "citations", None) or []:
        kind = getattr(citation, "type", None)
        if kind is not None and kind != "TEXT_CONTENT":
            continue

        index = getattr(citation, "content_index", 0) or 0
        if index not in offsets:
            continue

        start = getattr(citation, "start", None)
        end = getattr(citation, "end", None)
        if not isinstance(start, int) or not isinstance(end, int):
            continue

        shift = offsets[index] - lead
        start = min(max(start + shift, 0), len(answer))
        end = min(max(end + shift, 0), len(answer))
        if start >= end:
            continue

        citations.append(
            {
                "start": start,
                "end": end,
                # From the answer, not from Cohere, so that
                # answer[start:end] == text always holds.
                "text": answer[start:end],
                "sources": [
                    _source_label(source)
                    for source in getattr(citation, "sources", None) or []
                ],
            }
        )
    return citations


def _source_label(source: Any) -> str:
    """The article URL or title behind a citation source.

    Not one predictable shape: sometimes `document`, sometimes `tool_output`,
    fields either present or still wrapped in the JSON we sent as "data". Never
    falls back to str(source), which would dump the whole article.
    """
    for holder in (
        getattr(source, "document", None),
        getattr(source, "tool_output", None),
    ):
        if not isinstance(holder, dict):
            continue
        if holder.get("url") or holder.get("title"):
            return str(holder.get("url") or holder["title"])
        raw = holder.get("data")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and (parsed.get("url") or parsed.get("title")):
                return str(parsed.get("url") or parsed["title"])
    return "wikipedia"


def _for_log(body: Any) -> str:
    """Make a provider body safe for a log line.

    Truncated because logs get shipped elsewhere, flattened because newlines in
    a body can forge extra entries.
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


def _tool_arguments(call: Any) -> dict[str, Any]:
    """The model's arguments, parsed and reduced to the schema's own fields.

    ValueError, not JSONDecodeError: json also raises it for an integer literal
    longer than int() will parse. Extra fields are dropped because a deeply
    nested object blows the stack in asdict() on the way out.
    """
    raw = getattr(getattr(call, "function", None), "arguments", None)
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        log.warning("Unparsable arguments for %s", _tool_name(call))
        return {}
    if not isinstance(parsed, dict):
        return {}

    arguments: dict[str, Any] = {}
    query = parsed.get("query")
    if isinstance(query, str) and _encodable(query):
        arguments["query"] = query[:MAX_QUERY_CHARS]
    limit = parsed.get("limit")
    # bool is an int subclass, and json.loads accepts NaN and Infinity, so
    # neither survives on type alone.
    if isinstance(limit, int) and not isinstance(limit, bool):
        arguments["limit"] = max(-MAX_LIMIT, min(limit, MAX_LIMIT))
    return arguments


def _encodable(value: str) -> bool:
    """Whether the string can survive being rendered as JSON.

    json.loads accepts a lone surrogate like "\ud800", which then raises
    UnicodeEncodeError when the response is written out.
    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _tool_name(call: Any) -> str:
    """The tool's name, whatever shape the SDK object turns out to be."""
    name = getattr(getattr(call, "function", None), "name", None)
    return name if isinstance(name, str) and name else "unknown"


def _flatten(message: Any) -> tuple[str, dict[int, int], int]:
    """Join the content blocks, and say where each one landed.

    v2 returns content as a list of blocks, possibly empty. The map is block
    index to start offset in the joined text, which is what lets citations be
    rebased.
    """
    blocks = getattr(message, "content", None) or []
    parts: list[str] = []
    offsets: dict[int, int] = {}
    cursor = 0

    for index, block in enumerate(blocks):
        if getattr(block, "type", "text") != "text":
            continue
        text = getattr(block, "text", None)
        if not text:
            continue
        offsets[index] = cursor
        parts.append(text)
        cursor += len(text) + 1  # the newline join() will add

    joined = "\n".join(parts)
    # Stripping shifts every offset left by however much came off the front.
    # The shift is returned rather than folded in, because clamping a negative
    # result to zero here silently corrupts a citation that pointed into the
    # whitespace: "  Buzz Aldrin" cited at 2:13 would come back as "zz Aldrin w".
    lead = len(joined) - len(joined.lstrip())
    return joined.strip(), offsets, lead
