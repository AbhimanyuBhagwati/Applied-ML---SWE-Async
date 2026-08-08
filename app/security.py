"""Optional gate in front of /chat. Both parts are off unless configured."""

import secrets
import time
from collections import deque

from fastapi import HTTPException, Request

from app.config import Settings


def require_token(request: Request, settings: Settings) -> None:
    """Check the bearer token, if one is configured.

    With `API_AUTH_TOKEN` unset this does nothing, so local use and the
    reviewer's own curl still work without ceremony.
    """
    expected = settings.api_auth_token
    if not expected:
        return

    scheme, _, presented = request.headers.get("Authorization", "").partition(" ")
    # compare_digest so a wrong token can't be recovered by timing the reply.
    # Encoded first: given str it demands ASCII, and raises TypeError otherwise,
    # which would turn a non-ASCII token into a 500 instead of a 401.
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def client_key(request: Request, settings: Settings) -> str:
    """Who to count this request against.

    The TCP peer is the proxy, not the caller, whenever there's a proxy in
    front. `CLIENT_IP_HEADER` names the header to read instead, and is only
    safe to set when something trustworthy is actually setting that header:
    read it unconditionally and any caller can forge an identity per request
    and never hit the limit.
    """
    if settings.client_ip_header:
        # X-Forwarded-For is a chain; the original client is first. Stripped
        # before the truth test, or a header of spaces becomes the identity ""
        # and every such caller shares one bucket.
        forwarded = request.headers.get(settings.client_ip_header, "").split(",")[0].strip()
        if forwarded:
            return forwarded

    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding window per caller, counted in this process only.

    Being per process makes this a speed bump rather than a real control: run
    four workers and you get four times the limit. Anything serious belongs in
    a gateway or a shared store. It is still worth having, because what it
    guards against is one caller in a loop draining a metered credential, and
    that is usually a mistake rather than an attack.
    """

    # Smallest map worth sweeping at all.
    SWEEP_THRESHOLD = 1024

    def __init__(self, per_minute: int, window_seconds: float = 60.0) -> None:
        self._per_minute = per_minute
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._sweep_at = self.SWEEP_THRESHOLD
        self._last_sweep: float | None = None

    @property
    def enabled(self) -> bool:
        return self._per_minute > 0

    def check(self, key: str, now: float | None = None) -> float | None:
        """None if the call is allowed, else seconds until it would be."""
        if not self.enabled:
            return None

        now = time.monotonic() if now is None else now

        # Two triggers. Size, so a growing map gets collected, with the bar
        # doubling afterwards: a fixed one means a burst of genuinely active
        # callers sweeps on every request, since nothing is removed and the map
        # never drops back under the line. And age, because after a big burst
        # raises the bar, a quieter period would otherwise never reach it again
        # and the expired entries would sit there for good.
        stale = self._last_sweep is not None and now - self._last_sweep >= self._window
        if len(self._hits) >= self._sweep_at or (stale and self._hits):
            self._forget_idle_callers(now)
            self._sweep_at = max(self.SWEEP_THRESHOLD, len(self._hits) * 2)
            self._last_sweep = now
        elif self._last_sweep is None:
            self._last_sweep = now

        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self._window:
            hits.popleft()

        if len(hits) >= self._per_minute:
            return self._window - (now - hits[0])

        hits.append(now)
        return None

    def _forget_idle_callers(self, now: float) -> None:
        # Without this the map gains an entry per distinct caller and never
        # loses one, which is its own slow denial of service.
        for key in [
            key
            for key, hits in self._hits.items()
            if not hits or now - hits[-1] >= self._window
        ]:
            del self._hits[key]
