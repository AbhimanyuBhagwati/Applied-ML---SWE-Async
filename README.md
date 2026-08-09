# Cohere Chat

A small FastAPI service with one endpoint: send it a question, it calls the
[Cohere Chat API (v2)](https://docs.cohere.com/v2/reference/chat), lets the
model search Wikipedia if the question needs it, and returns the answer.

Tasks 1 and 2. Query history comes next, on top of this endpoint.

## Setup

```bash
uv sync
cp .env.example .env
```

Grab a trial key from the [Cohere dashboard](https://dashboard.cohere.com/api-keys)
and put it in `.env` as `COHERE_API_KEY=...`. That file is gitignored.

Set `WIKIPEDIA_USER_AGENT` in the same file while you're there. Wikimedia's
[User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)
asks for a contact address and they may throttle or block clients without one.
Something like `yourname/1.0 (you@example.com)` is enough. It runs without it
and logs a warning, but don't leave it that way.

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
  "response": "Buzz Aldrin was the second person to walk on the moon...",
  "finish_reason": "COMPLETE",
  "tool_plan": "I will search Wikipedia for the Apollo 11 crew.",
  "tool_calls": [
    {
      "name": "search_wikipedia",
      "arguments": {"query": "second person to walk on the moon", "limit": 5},
      "result_count": 5,
      "results": [
        {
          "title": "Buzz Aldrin",
          "url": "https://en.wikipedia.org/wiki/Buzz_Aldrin",
          "snippet": "...",
          "extract": "Buzz Aldrin is an American former astronaut...",
          "wordcount": 12000,
          "last_edited": "2026-01-02T03:04:05Z"
        }
      ]
    }
  ],
  "citations": [
    {"start": 0, "end": 11, "text": "Buzz Aldrin",
     "sources": ["https://en.wikipedia.org/wiki/Buzz_Aldrin"]}
  ]
}
```

`tool_calls` and `citations` are there so the grounding can be checked instead
of taken on trust. A question that doesn't need a lookup, like "write a haiku",
comes back with both empty and no search performed.

The model reads up to 1200 characters of each article's opening; the response
carries the first 200 of that. Titles, urls and snippets come back whole, since
those are what say why an article was used. Returning the extracts too made
`tool_calls` 8KB for a 227 byte answer, and the budget allows eight searches.

`query` is required, 1 to 8000 characters after stripping. Both ends of that
range are pinned by tests.

`citations` index into `response`: `response[start:end]` is the cited span.
Citations Cohere raises against its own reasoning, or against text that was
stripped, are dropped rather than returned pointing at the wrong words.

### GET /health

Reports configuration readiness. With no API key it answers 503 and
`"status": "degraded"`, because every `/chat` call would fail and nothing should
be routing traffic here. It does not call Cohere or Wikipedia, so it says the
service is configured to work, not that both upstreams are up.

## The Wikipedia tool

The model is offered one tool, `search_wikipedia`. When it asks for a search we
run it, hand the results back as documents, and call Cohere again; the answer
from that second call is what the caller gets. If the model asks for another
search instead, the loop goes round again, up to `MAX_TOOL_ROUNDS` (4).

The tool makes a search call and then one or more extract calls, and the
second part is the reason it works:

* [`list=search`](https://www.mediawiki.org/wiki/API:Search) ranks the articles
  and returns the snippet of text around each match.
* `prop=extracts` then fetches those articles' opening paragraphs, batched 20
  at a time because that is TextExtracts' own ceiling.

The snippet alone isn't enough. Searching "second person to walk on the moon"
ranks **Buzz Lightyear** first, and the actual Buzz Aldrin snippet reads "is an
American former astronaut, aeronautical engineer, and fighter pilot", which
doesn't answer the question. With snippets only, the model searched four times
and gave up. With the opening paragraphs attached it answers on the first
search. If the extracts call fails the snippets are still returned, so a
degraded answer beats none. Extracts are fetched in batches of 20, because
that's TextExtracts' own limit and it drops the rest silently, and `exchars`
bounds each one at the API rather than downloading a full introduction to
throw most of it away.

### What bounds the work

Per-round settings multiply rather than bound: ten rounds of twenty searches is
800 sequential Wikipedia requests. So there are two layers.

Per round, `MAX_TOOL_ROUNDS` (4) is how many times the model may search before
we stop asking, with one more Cohere call after the last round so the final
results are actually used, and `MAX_TOOL_CALLS_PER_ROUND` (4) is how many
searches one response may ask for.

Across the whole request, whichever of these runs out first ends the searching:

| Setting | Default | Bounds |
| --- | --- | --- |
| `MAX_TOTAL_SEARCHES` | 8 | searches, however they're spread across rounds |
| `TOOL_LOOP_SECONDS` | 45 | wall clock, enforced as a hard timeout returning 504 |

Context size isn't a runtime budget, because a result's size isn't known until
it has been fetched, so a check before fetching can only overshoot. The bound
is static instead: `MAX_TOTAL_SEARCHES` x `WIKIPEDIA_MAX_RESULTS` x
`WIKIPEDIA_EXTRACT_CHARS`, about 48K characters at the defaults. Turned all the
way up those settings allow far more than the model's context window, so raise
them together with an eye on what the model can actually hold.

Once the rounds or the search budget are gone, the final Cohere call withholds
the tool, so the model answers rather than asking for a search we would refuse.
`tool_choice: "NONE"` is the documented alternative but this model returns 400
for it. A call that isn't run still gets a tool message back, because
Cohere rejects the next request if any `tool_call_id` is unanswered.

## A note on scope

There's no authentication or rate limiting here. `/chat` spends a metered
Cohere credential, so anyone who can reach it spends your quota: run it locally
and don't put it on a public address. That belongs in a gateway rather than in
this service, and building it here would be more code than the task itself.

`MAX_OUTPUT_TOKENS` (4096) caps one generation. It's a ceiling, not a
reservation: billing follows tokens actually produced, so a roomy limit costs
nothing on an ordinary answer. It needs headroom because thinking is on by
default on the reasoning models and comes out of the same budget.

## Layout

* `app/config.py` reads the key, model name, timeout, retry count, token limit
  and the loop budgets from the environment, and bounds all of them.
* `app/wikipedia.py` holds the MediaWiki client and the tool schema the model
  sees. The schema's wording is what decides whether the model searches at all.
* `app/chat_service.py` is the only thing that talks to Cohere. It builds the
  request, flattens the reply, and turns SDK exceptions into one `ChatError`
  that carries the upstream status without carrying the upstream body.
* `app/main.py` is the HTTP layer. It owns the Cohere client's lifetime, maps `ChatError`
  onto a status code, and exposes the `ChatService` as a FastAPI dependency,
  which is what lets the tests swap in a fake.
* `app/schemas.py` holds the request and response models.

Two things worth calling out. v2 returns message content as a list of blocks
rather than a string, so `_flatten` joins the text blocks and skips
anything else. And the SDK refuses to construct a client without a token, so
with no key set we pass a dummy one, the app boots normally, and `/chat`
answers 503 with a message that says what to fix.

The Cohere client takes `timeout` and `max_retries` directly and is an async
context manager, so the lifespan just holds it open with `async with` and its
`__aexit__` closes the httpx client it owns.

## Errors

| Situation | Status |
| --- | --- |
| Query empty, missing, whitespace-only, or over 8000 chars | 422 |
| Unknown field in the request body | 422 |
| Cohere rate limited us | 429, with their `Retry-After` passed through |
| Cohere rejected the API key | 503 |
| `finish_reason` of `ERROR` or `TIMEOUT` | 502 |
| Model returned no text at all | 502 |
| Anything else upstream | 502 |
| The request ran past `TOOL_LOOP_SECONDS`, or Cohere timed out | 504 |
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
fails the suite instead of scrolling past. Coverage sits around 98%; the gaps
are branches in defensive paths, and chasing the last few would mean writing
tests to satisfy a number rather than to prove anything.

* `tests/test_chat_service.py` covers the request we build, the tool loop
  (results reaching the second Cohere call, several searches in one turn,
  several rounds, the round cap, Wikipedia being down), citation parsing, and
  both error paths.
* `tests/test_api.py` covers the two endpoints, the validation cases, stripping,
  the 502 and 503 mappings, and one test that skips the dependency override so
  the real startup wiring is exercised too.
* `tests/test_wikipedia.py` covers the MediaWiki client: the parameters the
  spec asks for, snippet markup stripping, URL encoding, the limit clamp, error
  shapes including the ones that arrive as HTTP 200, and extracts degrading to
  snippets when the second call fails.
* `tests/test_config.py` covers the defaults, env var precedence and the
  value bounds, built with `_env_file=None` so a developer's own .env can't
  change the result.

With coverage:

```bash
uv run pytest --cov=app --cov-branch --cov-report=term-missing
```

`tests/test_live_contract.py` is skipped by default and calls the real Cohere
and Wikipedia APIs when you opt in. Everything else mocks both providers, which
proves the code behaves as written but not that its assumptions still match the
APIs. That gap is real: `tool_choice: "NONE"` is documented and looked correct
under mocks, and the live call returned 400.

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_live_contract.py
```

A trial key allows 20 calls a minute, so running these repeatedly will hit a
429 from Cohere. That is the key's limit rather than a failure of the code.

Lint and formatting:

```bash
uv run ruff check app tests && uv run ruff format --check app tests
```
