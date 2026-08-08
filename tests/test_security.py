import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.datastructures import Headers

from app.config import Settings
from app.security import RateLimiter, client_key, require_token


class FakeRequest:
    def __init__(self, headers=None, host="1.2.3.4"):
        self.headers = Headers(headers or {})
        self.client = type("Client", (), {"host": host})() if host else None


def settings_with(token=""):
    return Settings(cohere_api_key="k", api_auth_token=token, _env_file=None)


def test_no_token_configured_lets_everything_through():
    require_token(FakeRequest(), settings_with(token=""))


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "secret"},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Basic secret"},
        {"Authorization": "Bearer "},
    ],
    ids=["missing", "no-scheme", "wrong-token", "wrong-scheme", "empty-token"],
)
def test_bad_credentials_are_rejected(headers):
    with pytest.raises(HTTPException) as caught:
        require_token(FakeRequest(headers), settings_with(token="secret"))

    assert caught.value.status_code == 401
    assert caught.value.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_the_right_token_is_accepted(scheme):
    require_token(
        FakeRequest({"Authorization": f"{scheme} secret"}), settings_with(token="secret")
    )


def test_client_key_uses_the_peer_by_default():
    plain = settings_with()

    assert client_key(FakeRequest(host="9.9.9.9"), plain) == "9.9.9.9"
    assert client_key(FakeRequest(host=None), plain) == "unknown"


def test_a_forwarded_header_is_ignored_unless_configured():
    """Trusting it unasked would let any caller forge an identity per request."""
    request = FakeRequest({"X-Forwarded-For": "5.5.5.5"}, host="10.0.0.1")

    assert client_key(request, settings_with()) == "10.0.0.1"


@pytest.mark.parametrize(
    "value,expected",
    [("5.5.5.5", "5.5.5.5"), ("5.5.5.5, 10.0.0.1, 10.0.0.2", "5.5.5.5"), ("", "10.0.0.1")],
    ids=["single", "chain-takes-first", "empty-falls-back"],
)
def test_configured_header_is_used(value, expected):
    settings = Settings(
        cohere_api_key="k", client_ip_header="X-Forwarded-For", _env_file=None
    )
    request = FakeRequest({"X-Forwarded-For": value}, host="10.0.0.1")

    assert client_key(request, settings) == expected


def test_disabled_limiter_never_blocks():
    limiter = RateLimiter(per_minute=0)

    assert all(limiter.check("a", now=0.0) is None for _ in range(100))
    assert limiter.enabled is False


def test_limit_applies_per_caller():
    limiter = RateLimiter(per_minute=2)

    assert limiter.check("a", now=0.0) is None
    assert limiter.check("a", now=1.0) is None
    assert limiter.check("a", now=2.0) is not None
    # A different caller has their own budget.
    assert limiter.check("b", now=2.0) is None


def test_the_window_slides():
    limiter = RateLimiter(per_minute=1, window_seconds=60.0)
    limiter.check("a", now=0.0)

    wait = limiter.check("a", now=30.0)
    assert wait == pytest.approx(30.0)

    assert limiter.check("a", now=60.0) is None


def test_old_hits_expire_while_recent_ones_still_count():
    """A caller who keeps trickling in stays tracked, but their stale hits drop."""
    limiter = RateLimiter(per_minute=2, window_seconds=60.0)
    limiter.check("a", now=0.0)
    limiter.check("a", now=50.0)

    assert limiter.check("a", now=55.0) is not None

    # At t=70 the first hit has aged out but the second hasn't, so the caller
    # isn't pruned outright, one slot just frees up.
    assert limiter.check("a", now=70.0) is None
    assert limiter.check("a", now=71.0) is not None


def test_idle_callers_are_forgotten(monkeypatch):
    """The map would otherwise grow one entry per caller and never shrink."""
    monkeypatch.setattr(RateLimiter, "SWEEP_THRESHOLD", 10)
    limiter = RateLimiter(per_minute=5)
    for i in range(10):
        limiter.check(f"caller-{i}", now=0.0)
    assert len(limiter._hits) == 10

    limiter.check("someone-new", now=120.0)

    assert len(limiter._hits) == 1


def test_a_small_map_inside_the_window_is_left_alone(monkeypatch):
    """Sweeping every request is O(callers) per request, worst exactly when busy."""
    monkeypatch.setattr(RateLimiter, "SWEEP_THRESHOLD", 100)
    limiter = RateLimiter(per_minute=5)
    for i in range(10):
        limiter.check(f"caller-{i}", now=0.0)

    limiter.check("someone-new", now=30.0)

    # Too small to bother with, and not old enough to have gone stale.
    assert len(limiter._hits) == 11


def test_a_quiet_period_still_gets_swept(monkeypatch):
    """Size alone isn't enough.

    A burst raises the bar to twice what survived. If traffic then drops, the
    map never reaches the new bar again and the expired entries sit there for
    good. Age is the second trigger.
    """
    monkeypatch.setattr(RateLimiter, "SWEEP_THRESHOLD", 4)
    limiter = RateLimiter(per_minute=100)
    for i in range(20):
        limiter.check(f"burst-{i}", now=0.0)
    assert limiter._sweep_at > len(limiter._hits)

    # One caller, much later. Nothing new arrives to push the map over the bar.
    limiter.check("straggler", now=300.0)

    assert len(limiter._hits) == 1


def test_a_non_ascii_token_presented_against_an_ascii_one():
    """compare_digest demands ASCII when handed str, and raises TypeError.

    Reachable from outside, so it has to be a 401 and not a 500.
    """
    with pytest.raises(HTTPException) as caught:
        require_token(
            FakeRequest({"Authorization": "Bearer sëcret"}), settings_with(token="secret")
        )

    assert caught.value.status_code == 401


@pytest.mark.parametrize("token", ["sëcret", "秘密", "tøken-123"])
def test_a_non_ascii_token_cannot_be_configured(token):
    """Rejected at startup, because it could never match anything.

    Header bytes carry no charset. A UTF-8 token on the wire comes back
    decoded as latin-1, so the service would answer 401 forever and look
    broken rather than misconfigured.
    """
    with pytest.raises(ValidationError):
        settings_with(token=token)


@pytest.mark.parametrize("token", ["sëcret", "秘密"])
def test_a_non_ascii_token_still_cannot_crash_the_handler(token):
    """Belt and braces: if validation were bypassed, still a 401."""
    settings = Settings.model_construct(cohere_api_key="k", api_auth_token=token)

    with pytest.raises(HTTPException) as caught:
        require_token(FakeRequest({"Authorization": "Bearer wrong"}), settings)

    assert caught.value.status_code == 401


def test_sweeping_amortises_when_callers_stay_active(monkeypatch):
    """A fixed threshold swept on every request once the map crossed it.

    Active callers are never removed, so the map never drops back under the
    line and the O(callers) walk runs again and again.
    """
    monkeypatch.setattr(RateLimiter, "SWEEP_THRESHOLD", 3)
    limiter = RateLimiter(per_minute=100)
    sweeps = []
    original = limiter._forget_idle_callers
    monkeypatch.setattr(
        limiter, "_forget_idle_callers", lambda now: (sweeps.append(now), original(now))[1]
    )

    # Ten callers, all active, all hitting repeatedly inside the window.
    for round_number in range(5):
        for caller in range(10):
            limiter.check(f"caller-{caller}", now=float(round_number))

    # Doubling the bar each time means a handful of sweeps, not one per request.
    assert len(sweeps) <= 3, f"swept {len(sweeps)} times across 50 requests"


@pytest.mark.parametrize("value", ["秘密", "X Forwarded", "hdr:name", "a\nb"])
def test_an_impossible_header_name_is_refused_at_startup(value):
    """Looking up a name outside the RFC 7230 grammar raises, so it's a 500."""
    with pytest.raises(ValidationError):
        Settings(cohere_api_key="k", client_ip_header=value, _env_file=None)


@pytest.mark.parametrize("value", ["   ", "", " , 10.0.0.9"])
def test_a_blank_forwarded_value_falls_back_to_the_peer(value):
    """Otherwise every such caller shares one bucket keyed on the empty string."""
    settings = Settings(
        cohere_api_key="k", client_ip_header="X-Forwarded-For", _env_file=None
    )
    request = FakeRequest({"X-Forwarded-For": value}, host="10.0.0.1")

    assert client_key(request, settings) == "10.0.0.1"
