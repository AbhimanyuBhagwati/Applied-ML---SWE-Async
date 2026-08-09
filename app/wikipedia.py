"""Wikipedia search, via the MediaWiki `list=search` and `prop=extracts` modules.

Spec: https://www.mediawiki.org/wiki/API:Search
"""

import asyncio
import html
import logging
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

from app.config import Settings

log = logging.getLogger(__name__)

# Snippets come back with the matched terms wrapped in
# <span class="searchmatch">. That is the only markup the module emits, so a
# tag strip is enough here and there is no reason to pull in a parser.
TAGS = re.compile(r"<[^>]+>")

# TextExtracts caps exlimit at 20, so more hits than that need more requests.
EXTRACT_BATCH = 20
# And exchars is capped at 1200, so asking for more is an invalid request.
MAX_EXCHARS = 1200

# What the caller is told. Transport exception text names the host and path,
# which for a misconfigured api_url can be internal. Details go to the log.
UNREACHABLE = "Could not reach Wikipedia."
UNREADABLE = "Wikipedia returned a response that could not be read."
RATE_LIMITED = "Wikipedia is rate limiting us."
# MediaWiki signals throttling in the body, with a 200.
RATE_LIMIT_CODES = frozenset({"ratelimited", "maxlag"})
# Both carry Retry-After per Wikimedia's guidance.
BACKOFF_STATUSES = frozenset({429, 503})
# Used when Retry-After is missing or unreadable.
DEFAULT_BACKOFF = 30.0
# The ceiling on any wait we will hold. Longer than a request budget already,
# so a broken header can't take Wikipedia search out until restart.
MAX_BACKOFF = 300.0

# Wikimedia asks clients to keep to a handful of concurrent requests. The
# cooldown alone doesn't bound this: many /chat requests can pass the check
# together before any of them gets a 429 back.
MAX_CONCURRENCY = 3

# The schema text and the code have to agree on these or they drift apart.
DEFAULT_RESULTS = 3
MAX_RESULTS_HINT = 5

# Log lines get the same treatment as Cohere's: a URL or query echoed back
# over several lines can forge log entries, and logs get shipped elsewhere.
LOGGED_CHARS = 200

# The schema the model sees. It decides whether to call this from the
# description, so the wording is doing real work.
WIKIPEDIA_TOOL = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": (
            "Search English Wikipedia. Returns the matching articles, each "
            "with the text around the match and the article's opening "
            "paragraphs. Use this for questions about people, places, events, "
            "dates or any other real-world fact, rather than answering from "
            "memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for. Keywords work better than a "
                        "whole question: 'Apollo 11 crew' beats 'who was on "
                        "the crew of Apollo 11'."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": f"How many articles to return. 1 to {MAX_RESULTS_HINT}, default {DEFAULT_RESULTS}.",
                },
            },
            "required": ["query"],
        },
    },
}


class WikipediaError(RuntimeError):
    """Wikipedia couldn't be reached, or refused the search.

    The message is shown to the caller, so it must stay free of hostnames,
    paths and other configuration detail.
    """


class WikipediaClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http_client
        # When Wikimedia says back off, we back off.
        self._blocked_until = 0.0
        self._in_flight = asyncio.Semaphore(MAX_CONCURRENCY)

    async def search(
        self, query: Any, limit: Any = DEFAULT_RESULTS
    ) -> list[dict[str, Any]]:
        """Full text search, with each hit's opening paragraphs attached.

        Two steps. `list=search` ranks the articles; `prop=extracts` says what
        they contain. The snippet alone is ~200 characters around the match and
        often isn't the answer.

        Both arguments come from the model, so neither is trusted.
        """
        query = query.strip() if isinstance(query, str) else ""
        if not query:
            # srsearch is required, and an empty one is a nosrsearch error.
            # Cheaper to answer it here than to spend the round trip.
            raise WikipediaError("Search query was empty.")

        payload = await self._request(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": self._clamp(limit),
                # Defaults to size|wordcount|timestamp|snippet. Size is not
                # useful to a model, the rest are.
                "srprop": "snippet|wordcount|timestamp",
            }
        )

        # A hit with no title can't be linked to, and an entry that isn't a
        # dict isn't a hit. Dropping them beats emitting an article called ""
        # pointing at the wiki's front page.
        # Truncated locally as well as upstream: an endpoint ignoring srlimit
        # would otherwise walk past the result and context bounds.
        hits = [
            hit
            for hit in _listing(payload, "search", required=True)
            if isinstance(hit, dict) and _text(hit.get("title"))
        ][: self._clamp(limit)]
        extracts = await self._extracts([hit.get("pageid") for hit in hits])

        return [
            {
                "title": _text(hit.get("title")),
                "url": self._article_url(_text(hit.get("title"))),
                "snippet": _plain_text(_text(hit.get("snippet"))),
                "extract": extracts.get(hit.get("pageid"), ""),
                "wordcount": hit.get("wordcount"),
                "last_edited": hit.get("timestamp"),
            }
            for hit in hits
        ]

    async def _extracts(self, page_ids: list[Any]) -> dict[int, str]:
        """Opening paragraphs for the given articles.

        Batched, because TextExtracts returns at most 20 extracts per request
        and silently drops the rest. `exchars` bounds the response upstream so
        we aren't downloading whole introductions to then throw most away.
        """
        wanted = [str(page_id) for page_id in page_ids if isinstance(page_id, int)]
        if not wanted:
            return {}

        limit = min(self._settings.wikipedia_extract_chars, MAX_EXCHARS)
        found: dict[int, str] = {}

        for start in range(0, len(wanted), EXTRACT_BATCH):
            batch = wanted[start : start + EXTRACT_BATCH]
            try:
                payload = await self._request(
                    {
                        "action": "query",
                        "pageids": "|".join(batch),
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "exlimit": EXTRACT_BATCH,
                        "exchars": limit,
                    }
                )
                for page in _listing(payload, "pages"):
                    if isinstance(page, dict) and isinstance(page.get("pageid"), int):
                        extract = _text(page.get("extract"))
                        if extract:
                            found[page["pageid"]] = _truncate(
                                extract, self._settings.wikipedia_extract_chars
                            )
            except WikipediaError as exc:
                # The search worked, so keep it. Parsing is inside the try for
                # the same reason: a schema change here shouldn't throw away
                # snippets we already have.
                log.warning("Could not fetch extracts: %s", exc)
                break

        return found

    def _back_off(self, seconds: float) -> "WikipediaError":
        """Hold off further requests, and say for how long.

        Longest outstanding wait wins: taking the newest would let a concurrent
        one-second 429 cancel a five-minute one.
        """
        self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)
        return WikipediaError(
            f"{RATE_LIMITED} Try again in about {int(seconds)} seconds."
        )

    def _clamp(self, limit: Any) -> int:
        """The model picks this, so it can be anything at all.

        OverflowError included: int(float("inf")) raises it.
        """
        try:
            wanted = int(limit)
        except (TypeError, ValueError, OverflowError):
            wanted = DEFAULT_RESULTS
        return max(1, min(wanted, self._settings.wikipedia_max_results))

    def _article_url(self, title: str) -> str:
        # Wikipedia titles use underscores for spaces; everything else has to
        # be percent-encoded or titles with & or ? produce broken links.
        return self._settings.wikipedia_article_url + quote(title.replace(" ", "_"))

    async def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._in_flight:
                # Rechecked after taking a permit, not before waiting for one.
                # Callers queued behind the semaphore passed the earlier check
                # while the coast was clear, so without this a 429 arriving
                # mid-queue lets every one of them through anyway.
                waiting = self._blocked_until - time.monotonic()
                if waiting > 0:
                    raise WikipediaError(
                        f"{RATE_LIMITED} Try again in about {int(waiting) + 1} seconds."
                    )

                response = await self._http.get(
                    self._settings.wikipedia_api_url,
                    # formatversion=2 drops the legacy JSON shapes, utf8 stops the
                    # API escaping non-ASCII into \uXXXX.
                    params={"format": "json", "formatversion": 2, "utf8": 1, **params},
                    headers={"User-Agent": self._settings.wikipedia_user_agent},
                    timeout=self._settings.wikipedia_timeout_seconds,
                )
            response.raise_for_status()
            headers = response.headers
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            log.warning(
                "Wikipedia returned %s: %s", exc.response.status_code, _short(exc)
            )
            if exc.response.status_code in BACKOFF_STATUSES:
                # Wikimedia documents Retry-After on both 429 and 503, and asks
                # that clients honour it. Held here and enforced on the next
                # call rather than left to the model to respect.
                raise self._back_off(
                    _retry_seconds(exc.response.headers.get("retry-after"))
                ) from exc
            raise WikipediaError(UNREACHABLE) from exc
        except httpx.HTTPError as exc:
            log.warning("Wikipedia request failed: %s", _short(exc))
            raise WikipediaError(UNREACHABLE) from exc
        except ValueError as exc:
            log.warning("Wikipedia sent something that wasn't JSON: %s", _short(exc))
            raise WikipediaError(UNREADABLE) from exc

        if not isinstance(payload, dict):
            log.warning("Wikipedia sent %s at the top level", type(payload).__name__)
            raise WikipediaError(UNREADABLE)

        # A MediaWiki error arrives as HTTP 200 with an "error" key, so the
        # status check above doesn't catch it. Only the code is passed on:
        # it's a fixed vocabulary (nosrsearch, search-error) while "info" is
        # free text that can quote configuration back.
        error = payload.get("error")
        if isinstance(error, dict):
            log.warning("Wikipedia refused the query: %s", _short(error))
            # MediaWiki reports throttling as a 200 with this code, so the
            # status check above never sees it.
            if _text(error.get("code")) in RATE_LIMIT_CODES:
                # Throttling can arrive as a 200, and it still carries the
                # header saying how long to wait.
                raise self._back_off(_retry_seconds(headers.get("retry-after")))
            raise WikipediaError(
                f"Wikipedia refused the search: {_text(error.get('code')) or 'unknown'}"
            )
        if error is not None:
            raise WikipediaError("Wikipedia refused the search.")

        return payload


def _listing(payload: dict[str, Any], key: str, required: bool = False) -> list[Any]:
    """Dig the list out of payload["query"][key].

    `required` for search, where the API documents a genuine empty result as
    `query.search: []`. A missing structure there means an invalid response or
    a schema change, and reporting it as "nothing matched" would tell the model
    something false about the world. Extracts are looser, because a failure
    there only costs us the extracts.
    """
    query = payload.get("query")
    if query is None:
        if required:
            log.warning("Wikipedia response had no query")
            raise WikipediaError(UNREADABLE)
        return []
    if not isinstance(query, dict):
        log.warning("Wikipedia sent query as %s", type(query).__name__)
        raise WikipediaError(UNREADABLE)

    found = query.get(key)
    if found is None:
        if required:
            log.warning("Wikipedia response had no %s", key)
            raise WikipediaError(UNREADABLE)
        return []
    if not isinstance(found, list):
        log.warning("Wikipedia sent %s as %s", key, type(found).__name__)
        raise WikipediaError(UNREADABLE)
    return found


def _retry_seconds(retry_after: str | None) -> float:
    """How long to wait, from a Retry-After header.

    RFC 9110 allows seconds or an HTTP date, and both turn up. Anything
    unreadable falls back to a fixed pause rather than to no wait at all.
    """
    value = (retry_after or "").strip()
    if not value:
        return DEFAULT_BACKOFF

    if value.isdigit():
        # A floor of a second: "0" would install a block that has already
        # expired and tell the model to wait zero seconds.
        # isdigit() is true for things float() rejects, "superscript two" among
        # them. float rather than int because a header can be arbitrarily long:
        # int() refuses past 4300 digits while float saturates to inf.
        try:
            return min(max(float(value), 1.0), MAX_BACKOFF)
        except ValueError:
            return DEFAULT_BACKOFF

    try:
        moment = parsedate_to_datetime(value)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        seconds = (moment - datetime.now(UTC)).total_seconds()
    except (TypeError, ValueError, OverflowError):
        # A date with an implausible year overflows rather than parsing.
        return DEFAULT_BACKOFF
    return min(max(seconds, 0.0), MAX_BACKOFF)


def _short(value: Any) -> str:
    """One line, bounded, for a log."""
    text = str(value)[:LOGGED_CHARS]
    flattened = " ".join(text.split())
    return flattened + "..." if len(str(value)) > LOGGED_CHARS else flattened


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    # Cut on a word boundary so the model isn't handed half a word.
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _plain_text(snippet: str) -> str:
    """Strip the search-match markup and decode the entities behind it."""
    return html.unescape(TAGS.sub("", snippet)).strip()
