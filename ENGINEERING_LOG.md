# Engineering log

Log of what I ran and what it turned up. New entry each time I do a coverage
run or a review pass, not otherwise. It ships with the repo, because how this
was built is part of what's being handed in — including the entries where I
was wrong, which stay as written with the correction alongside.

I'm using Claude Code on this, including for the review passes, with OpenAI
Codex as a second reviewer. Where something below was caught by a review
rather than by me, it says so.

## 2026-08-08, before any code

Read all three tasks first, and drew the thing on paper before writing any of
it.

* The tasks compose: task 1 is an HTTP layer over one upstream call, task 2
  turns that call into a loop with a tool in the middle, task 3 records what
  the loop produced. So the seams were drawn for the end state — an HTTP layer
  that owns lifetimes and status codes, a service that owns the conversation
  with Cohere, clients that own their upstreams, and eventually a store that
  owns persistence. Whether task 3 could later land without moving anything
  else was the test of the sketch. (It did.)
* The tool loop was the part worth drawing in detail: tool calls out, every
  call answered or the next request is rejected, results back in as documents,
  then another generation. The budgets went on the sketch before any code
  existed, because the loop is the only part of the system that can run away
  on someone else's metered API.
* Testing shape decided up front too: everything offline behind fakes, plus a
  small opt-in live suite. Mocks prove the code matches my assumptions and say
  nothing about whether my assumptions match the API; both halves are needed.

## 2026-08-08, review pass over task 1

Read back through everything after stripping the project down to task 1.

* `logging.basicConfig` was still sitting in `main.py` after I deleted the code
  that actually logged. Nothing used it, and uvicorn configures its own.
* `httpx` was in the runtime dependencies but nothing in `app/` imports it.
  Only the tests do. Moved to dev.
* The tests were reading my real `.env`. `test_health` asserted the default
  model name, so the moment anyone sets `COHERE_MODEL` in `.env` it fails for
  no good reason. Fixtures pin both env vars now.
* `config.py` had no tests at all. Added `test_config.py`, built with
  `_env_file=None` so a local `.env` can't change the result.
* Smaller stuff: a function-local import, wrong import grouping, and a `for`
  loop doing the job of `parametrize` (first failure hides the second).

## 2026-08-08, first coverage run

```
uv run pytest --cov=app --cov-branch --cov-report=term-missing
```

16 tests, 99%. One line missed: `main.py:29`, `return
request.app.state.chat_service`.

Worth chasing rather than rounding up. Every API test overrode that dependency,
so nothing verified the route reads the service the app builds at startup.
Added a test that skips the override. 17 tests, 100% line and branch.

The percentage on its own doesn't mean much, it only says every line ran. The
edge cases are the part I care about: empty content, missing content, non-text
blocks, both upstream failure modes, and the validation cases.

## 2026-08-08, second review pass

* `" "` and `"\n\t "` passed validation. `min_length=1` doesn't strip first, so
  a whitespace-only query went upstream and came back an error, spending a
  metered call to learn something checkable locally. Strips first now.
* Unknown fields were silently dropped, so `{"query": "hi", "temperture": 0.9}`
  looked like it worked. Now `extra="forbid"`.
* A Starlette deprecation warning had been in the output for a while and I'd
  been ignoring it. Turned warnings into errors in `pyproject.toml` and added
  `httpx2`, which is what it was asking for.
* `.coverage` wasn't gitignored.
* README said 16 tests when there were 17. Moved the count out of the README
  entirely, since it drifted twice. It lives in these entries now.

21 tests, still 100% line and branch.

## 2026-08-08, cleared the open list

Had five things parked as "not blockers". Looked at them again and none of them
were actually hard, so keeping a list of things I knew were wrong made no sense.
All five done.

* Timeout and retries are set in config now, 30s and 1. The SDK's own defaults
  are 300s and 2 retries, which is up to fifteen minutes on one request.
* We pass our own `httpx.AsyncClient` and close it in the lifespan. Needed
  anyway for the timeout, since the SDK stops applying its default the moment
  you inject a client. The SDK's internal client has no close method at all,
  which is why this was never fixable from the outside.

  **Wrong, see the entry below.** The client takes `timeout` directly and
  closes itself. Leaving the mistake here rather than quietly editing it,
  since the point of these notes is what I actually thought at the time.
* Upstream failures map properly: 429 stays 429 and passes `Retry-After`
  through, a rejected key is 503 because that's our problem and not the
  caller's, everything else is 502. Response bodies go to the log, and the
  caller gets a fixed string per class.
* `finish_reason` of `ERROR` or `TIMEOUT` is a 502 rather than a 200 with a
  half-written answer. `MAX_TOKENS` still returns 200, it's a truncation not a
  failure.
* `/health` answers 503 and `"degraded"` with no key set.

41 tests, 100% line and branch. Chasing the last branch found `_retry_after`
was never tested against headers that exist but don't include `Retry-After`.

Checked the real endpoint again afterwards, since injecting our own httpx client
is the kind of change that works in tests and breaks in reality. It answered
correctly and shutdown was clean.

## 2026-08-08, the last two open items

Same question as last time: if they're small, why are they on a list instead of
done. `max_tokens` had no defence at all, it's one config value. The other one
was really three things wearing a trenchcoat, and they don't get the same answer.

* `MAX_OUTPUT_TOKENS`, 1024 by default, passed on every call.
* Bearer token auth in `security.py`, off unless `API_AUTH_TOKEN` is set. Off by
  default so local use and someone else's curl still work without ceremony.
  `secrets.compare_digest` rather than `==`, so a wrong token can't be recovered
  by timing the reply.
* A sliding window rate limiter, off unless `RATE_LIMIT_PER_MINUTE` is set.
  Blocked calls never reach Cohere, which is the point.

Did not add quota tracking, and this isn't a scope excuse. Cohere enforces the
quota and is the authority on it. A counter on our side drifts the moment
anything else uses the same key, and then the bug is in my bookkeeping rather
than anywhere real. Wrong layer.

Being honest about the limiter: it counts per process, so four workers means
four times the limit. It is a speed bump. Real enforcement is a gateway or a
shared store. Kept it because the realistic failure is one caller in a loop, not
an attacker, and it costs 37 lines.

One thing that nearly bit me: the limiter's dict gains an entry per distinct
caller and would never lose one, which is a slow leak dressed as a rate limit.
Idle callers get dropped now, and there's a test that would fail if that
regressed.

61 tests, 100% line and branch. Ran auth and the limiter against the live
endpoint too: 401 without a token, 401 with a wrong one, then 200, 200, 429 with
`Retry-After: 54`.

## 2026-08-08, review pass, and I had a fact wrong

The worst one first. I'd written, in a code comment and up in these notes, that
the SDK client can't be closed and that injecting an httpx client is the only
way to set a timeout. Both false. `AsyncClientV2` takes `timeout` and
`max_retries` directly, and it's an async context manager whose `__aexit__`
closes the httpx client it owns.

The check I'd run was `[m for m in dir(AsyncClientV2) if 'clos' in m.lower()]`,
which came back empty and I took as proof. `__aexit__` doesn't contain "clos".
I'd looked at the constructor signature with `timeout` right there in it and
still wrote the opposite a few minutes later.

So the injected client is gone. `main.py` holds the SDK client in an
`async with` and passes `timeout` and `max_retries`, httpx is back to being a
test-only dependency, and there are now two tests that pin both assumptions
against the installed SDK instead of asserting them in a comment.

The rest of the same pass:

* Tests weren't isolated any more. Setting `API_AUTH_TOKEN` in your own shell
  made the happy-path test return 401. I'd pinned two env vars by hand in the
  fixture and then added four more settings without going back. Fixed
  structurally: an autouse fixture clears every variable named after a
  `Settings` field, so a setting added later is covered without anyone
  remembering, and it chdirs to a tmp dir so the relative `.env` isn't found
  either. Both repros now pass.
* Config took nonsense: negative timeout, negative retries, zero output tokens,
  negative rate limit. That last one silently disabled the limiter, which is
  the kind of thing you notice from the bill. Bounded now, with tests.
* The limiter swept every caller on every request, which is O(callers) per
  request and worst exactly when busy. Sweeps past a threshold now.
* It also keyed on the TCP peer, which behind a proxy is the proxy, so all
  traffic shared one bucket. There's a `CLIENT_IP_HEADER` setting now, off by
  default, because reading a forwarded header nobody is overwriting lets any
  caller forge an identity per request and never hit the limit at all.
* Logged provider bodies are truncated. A log is still somewhere things get
  shipped offsite from.
* `logging.basicConfig()` was running at import again and fighting whatever the
  host had configured. Gone, for the second time.
* `test_timeout_and_retries_are_set` only ever checked the timeout. The name
  claimed more than the body did.

75 tests, 100% line and branch. Checked the live endpoint again after pulling
the injected client out, since that's the change most likely to pass tests and
fail in reality.

### What I thought was left at this point

Nothing I'd call a defect. For production the rate limiter would move to a
gateway, and there'd be structured request logs with a request id rather than
the plain warnings in there now.

**That was too confident, see below.** The next pass found eight more, three of
them substantive, and two of those only showed up by running the thing. Worth
remembering that I wrote "nothing I'd call a defect" about code that had a 500
in it.

## 2026-08-08, third review pass

All confirmed, all mine.

* Test isolation was still leaking. I cleared `NAME.upper()` only, and
  pydantic-settings matches case-insensitively, so `api_auth_token=x` in a
  shell still reached the app and turned the happy path into a 401. Now walks
  the real environment and matches case-insensitively against the field names.
  Verified with the lowercase repro.
* The limiter's threshold only delayed the problem. Once the map crossed it,
  active callers were never removed, so it never dropped back under and swept
  on every request after that. Measured it: 47 sweeps across 50 requests with
  ten active callers. The bar now doubles to whatever survived, so the same run
  sweeps twice.
* `secrets.compare_digest` raises TypeError on non-ASCII `str`, so `sëcret` as
  a token was a 500. Encoded before comparing.
* README still said `main.py` owns the httpx client, left over from the design
  I deleted.
* The SDK-internals test built a real client and never closed it.
* `REQUEST_TIMEOUT_SECONDS=inf` passed `gt=0` and removed the timeout again.
* Whitespace-only key and model were accepted. Trimmed now, so a variable set
  to spaces reads as absent instead of as a key made of spaces.
* Logged provider bodies are flattened as well as truncated. A body with
  newlines can forge extra log entries, and the body is one of the few places a
  caller's own input comes back.

Two things I only found by running it rather than testing it:

* A non-ASCII token can't work at all, even without the crash. Header bytes
  carry no charset: curl sends UTF-8, Starlette decodes latin-1, so it never
  matches and the service answers 401 forever looking broken rather than
  misconfigured. Rejected at startup now. My unit test had missed this because
  Starlette's Headers round-trips a Python str through latin-1 cleanly, which
  the wire does not.

  **Overstated, see the next entry.** Latin-1 bytes do round-trip; what
  breaks is UTF-8 on the wire against a UTF-8 string here. Rejecting is
  still right, on RFC 6750 grounds rather than impossibility.
* One live call came back `{"response": ""}` with `finish_reason: MAX_TOKENS`.
  Most likely the model spent the budget on reasoning and left the text blocks
  empty, though I never saw the token counts, so that's inference not fact.
  Couldn't reproduce it on three retries, but a 200 carrying an empty answer is
  a defect whether or not it's rare, so no text is now an error.

91 tests, 100% line and branch.

## Still open

Known, not fixed, and I'd rather name them than claim there's nothing left.

* The rate limiter is per process. Four workers, four times the limit. Real
  enforcement belongs in a gateway or a shared store.
* No structured request logging: no request id, no latency, no token usage.
  The warnings in there now are enough to debug a failure and not much else.
* `MAX_OUTPUT_TOKENS` is 1024, which this model can spend on reasoning alone.
  That's what produced the empty answer above. Higher would waste less, and
  cost more per call, and I haven't measured where the line is.
* No quota tracking, deliberately, for the reason two entries up.

## 2026-08-08, fourth review pass

Two corrections to me first, both fair.

I'd written that a non-ASCII bearer token is impossible to transmit. It isn't.
Raw latin-1 bytes round-trip through this stack fine; what's broken is that
UTF-8 on the wire comes back decoded as latin-1, so the two ends disagree.
Rejecting non-ASCII is still right, but the reason is RFC 6750 defining an
ASCII grammar for bearer credentials, not impossibility. Fixed the wording.

And attributing that empty response to reasoning tokens was an inference I
stated as fact. I never saw the token counts. It's consistent with what the
docs say about thinking being on by default, and it's still a guess.

The findings:

* The token limit was the real bug and the 502 only handled the symptom.
  Checked the docs: thinking is on by default on the reasoning models, shares
  `max_tokens`, and Cohere asks for at least 1K left for the response. At 1024
  total the model can reason the whole allowance away.

  Underneath that I had a premise wrong. `max_tokens` is a ceiling, not a
  reservation. Billing follows tokens actually generated, so a tight limit
  never saved quota on a normal call, it only truncated it. Default is 4096.
* `CLIENT_IP_HEADER` took any Unicode, and looking up a header name outside the
  RFC 7230 grammar raises rather than missing, so `秘密` was a 500. Constrained
  to the field-name grammar. A whitespace-only forwarded value also became the
  identity `""`, putting every such caller in one bucket. Stripped first now.
* Invalid requests consumed rate-limit budget. Kept that, because counting only
  the calls that reach Cohere leaves the service floodable with malformed
  bodies that still cost it work. But it was inconsistent: auth ran first, so a
  422 cost budget and a 401 didn't. The limiter now runs first, so every
  attempt costs the same, and the policy is written down and tested instead of
  being an accident of dependency order.
* The sweep was amortised on CPU but not on time. After a burst raised the bar,
  a quieter period would never reach it again and the expired entries would sit
  there. Second trigger on age.
* Transport exception strings were still logged raw, while provider bodies were
  sanitised. Same treatment now.
* Docs: README claimed the Cohere client was the dependency when it's
  `ChatService`, the error table missed `TIMEOUT` and the empty-output case, and
  the test list didn't mention `test_security.py`. Query length boundaries at 1
  and 8000 are now pinned rather than assumed.

105 tests, 100% line and branch.

## 2026-08-08, task 2, and what left with it

The Wikipedia tool. The design story — snippets alone ranking Buzz Lightyear
first for "second person to walk on the moon", and `prop=extracts` being the
fix — is in the README, so this entry is what changed around it.

* Removed `security.py` and `test_security.py`. The README's scope note says
  why: auth and rate limiting belong in a gateway, and the per-process limiter
  I'd built is a speed bump that looks like enforcement. Shipping the wrong
  layer dressed as the right one is worse than naming the gap. That retires
  the "Still open" line above about the limiter being per-process.
* A review pass (Claude Code) over the finished task 2 found three real ones,
  each confirmed by reading the code before fixing:
  * The soft time budget could never fire. Its deadline and the hard timeout
    around the loop shared the same duration and the same start, so the
    graceful path — withhold the tool, answer from what was gathered — was
    dead code, and the test that covered it called `_loop` directly and
    bypassed the race. It now reserves `min(request_timeout, tool_loop / 2)`
    for the final answering call.
  * `except cohere.core.ApiError` only worked by accident. The SDK resolves
    submodules lazily, and a module-level annotation happened to import
    `cohere.core` as a side effect; deferred annotations or a Protocol would
    have made the except clause itself raise, turning every API error into an
    uncaught 500. Now an eager `from cohere.core import ApiError`, in its own
    commit, because the bug is in task 1's code.
  * Tool calls in one round ran serially — up to eight sequential round trips
    against the same deadline. Now planned first, then run together, with the
    Wikipedia client's own concurrency cap bounding the fan-out.
* Smaller ones from the same pass, all applied: a floor under Retry-After "0",
  the default result count deduplicated into one constant the schema hint
  derives from, `MAX_BACKOFF` down from an hour to 300s so one broken header
  can't take the tool out until restart, both Wikipedia URLs added to
  `.env.example`, and two comments that had drifted from the code they sat on.

## 2026-08-08, task 3

The endpoint is one sentence, so most of the work was deciding what it doesn't
say. Writing the decisions down here because that's the part worth defending.

* **SQLite, one table.** A list in memory empties on restart and differs per
  worker, which makes "complete history" untrue in two ways. sqlite3 is
  stdlib so it costs nothing. The driver is synchronous, so writes hop to a
  worker thread; aiosqlite would avoid the hop and for millisecond writes it
  isn't worth the dependency. One shared connection behind a lock, because
  check_same_thread=False is permitted rather than safe.
* **Store the whole answer, not just query and response.** The literal reading
  of the task is those two fields, but task 2 exists to produce auditable
  grounding and a history that drops the searches and citations throws that
  away. tool_calls and citations go in as JSON, previewed exactly as the
  response previews them, so the record is what the caller was given.
* **Recording lives in main.py, after the answer exists.** Persistence isn't
  ChatService's concern. And a storage failure is logged and swallowed: the
  answer is already produced and paid for, so losing a row must not turn a
  working request into a 500. Reads still raise, because a history endpoint
  that quietly returned nothing is indistinguishable from an empty history.
* **Successes only.** A rejected query never reached the model and a failed one
  has no model-generated response, so neither has anything to pair the query
  with. That's the task's own wording, applied.
* **Newest first, paged, bounded.** Two tasks spent arguing unbounded anything
  is a bug; an endpoint that returns a history growing forever would be the
  same mistake. The cost of newest-first is that offsets shift as turns
  arrive, so deep paging on a busy history can repeat or skip a row. Worth it
  for the ordering a caller actually wants.

Not built, on purpose: auth, DELETE, search, filtering. With no auth there's no
caller to scope to, so the history is global, which is a privacy surface and
not just a quota one. That's two sentences in the README rather than a feature.

The layering from task 1 is why this was cheap: a store injected the same way
ChatService is, and nothing else needed to move.

Nine store tests and ten through the endpoints. No live test: the history never
leaves the process, so there's no external assumption for one to check.

172 tests, 98% coverage. Ran all three endpoints live afterwards: three
questions, then GET /history showing them newest first with their searches and
citations, then a restart with the history still there.
