# Cohere Chat

A small FastAPI service with one endpoint: send it a question, it calls the
[Cohere Chat API (v2)](https://docs.cohere.com/v2/reference/chat) and returns
the model's answer.

This is task 1. Wikipedia tool calling and query history come next, on top of
this endpoint.

## Setup

```bash
uv sync
cp .env.example .env
```

Grab a trial key from the [Cohere dashboard](https://dashboard.cohere.com/api-keys)
and put it in `.env` as `COHERE_API_KEY=...`. That file is gitignored.

## Running it

```bash
uv run uvicorn app.main:app --reload
```

Swagger UI is at http://127.0.0.1:8000/docs.

## Endpoints

### POST /chat

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"query": "Who was the second person to walk on the moon?"}'
```

```json
{
  "query": "Who was the second person to walk on the moon?",
  "response": "Buzz Aldrin was the second person to walk on the Moon...",
  "finish_reason": "COMPLETE"
}
```

`query` is required, 1 to 8000 characters after stripping. Both ends of that
range are pinned by tests.

### GET /health

Reports readiness, not just liveness. With no API key it answers 503 and
`"status": "degraded"`, because every `/chat` call would fail and nothing should
be routing traffic here.

## Protecting the endpoint

Both of these are off unless you set them, so local use stays frictionless.

```bash
API_AUTH_TOKEN=some-secret RATE_LIMIT_PER_MINUTE=20 uv run uvicorn app.main:app
```

With a token set, `/chat` wants `Authorization: Bearer some-secret` and answers
401 otherwise. With a limit set, callers over it get a 429 and a `Retry-After`,
and the request never reaches Cohere.

The limit counts every attempt, not only the ones that get as far as Cohere.
A malformed body or a wrong token spends budget too. Counting only successful
calls would leave the service floodable with requests that still cost it work,
and it's checked before the token so a bad-token flood isn't free.

The rate limiter counts in-process, which means it is a speed bump and not a
control: run four workers and you get four times the limit. Real enforcement
belongs in a gateway or a shared store. It is here because what it actually
guards against is one caller stuck in a loop draining a metered key, and that is
usually a mistake rather than an attack.

There is no local quota tracking on purpose. Cohere enforces the quota and is
the authority on it; a counter on this side drifts as soon as anything else uses
the same key, and then you are debugging your own bookkeeping instead.

`MAX_OUTPUT_TOKENS` (4096 by default) caps a single generation. It's a
ceiling, not a reservation: billing follows tokens actually produced, so a
roomy limit costs nothing on an ordinary answer. It needs headroom because
thinking is on by default on the reasoning models and comes out of the same
budget, and Cohere asks for at least 1K left for the response itself.

## Layout

* `app/config.py` reads the key, model name, timeout, retry count, token limit
  and the optional protections from the environment, and bounds all of them.
* `app/security.py` holds the bearer check and the rate limiter.
* `app/chat_service.py` is the only thing that talks to Cohere. It builds the
  request, flattens the reply, and turns SDK exceptions into one `ChatError`
  that carries the upstream status without carrying the upstream body.
* `app/main.py` is the HTTP layer. It owns the Cohere client's lifetime, maps `ChatError`
  onto a status code, and exposes the `ChatService` as a FastAPI dependency,
  which is what lets the tests swap in a fake.
* `app/schemas.py` holds the request and response models.

Two things worth calling out. v2 returns message content as a list of blocks
rather than a string, so `_extract_text` joins the text blocks and skips
anything else. And the SDK refuses to construct a client without a token, so
with no key set we pass a dummy one, the app boots normally, and `/chat`
answers 503 with a message that says what to fix.

The Cohere client takes `timeout` and `max_retries` directly and is an async
context manager, so the lifespan just holds it open with `async with` and its
`__aexit__` closes the httpx client it owns. Two tests pin those assumptions
against the installed SDK rather than trusting them.

## Errors

| Situation | Status |
| --- | --- |
| Bearer token missing or wrong, when one is configured | 401 |
| Query empty, missing, whitespace-only, or over 8000 chars | 422 |
| Unknown field in the request body | 422 |
| Caller went over the local rate limit | 429, with `Retry-After` |
| Cohere rate limited us | 429, with their `Retry-After` passed through |
| Cohere rejected the API key | 503 |
| `finish_reason` of `ERROR` or `TIMEOUT` | 502 |
| Model returned no text at all | 502 |
| Anything else upstream, including our own timeout | 502 |
| No `COHERE_API_KEY` set | 503 |

Queries are stripped before anything else happens. Cohere rejects a
whitespace-only query, so catching it here saves a pointless upstream call, and
that call is metered.

Error responses carry a fixed string per class. Cohere's response bodies and
transport exception text go to the log instead, since they can name
organisations, models or internal hosts.

## Tests

```bash
uv run pytest
```

Nothing touches the network. The Cohere client is replaced with a fake that
either replays content blocks or raises. Warnings are errors, so a deprecation
fails the suite instead of scrolling past.

* `tests/test_chat_service.py` covers the request we build, joining multiple
  content blocks, empty and missing content, skipping non-text blocks, and both
  error paths.
* `tests/test_api.py` covers the two endpoints, the validation cases, stripping,
  the 502 and 503 mappings, and one test that skips the dependency override so
  the real startup wiring is exercised too.
* `tests/test_security.py` covers the bearer check and the rate limiter:
  token comparison, non-ASCII handling, caller identity behind a proxy, the
  sliding window, and both sweep triggers.
* `tests/test_config.py` covers the defaults, env var precedence and the
  value bounds, built with `_env_file=None` so a developer's own .env can't
  change the result.

With coverage:

```bash
uv run pytest --cov=app --cov-branch --cov-report=term-missing
```
