from functools import lru_cache
from typing import Annotated

from pydantic import Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

# Strings that are meant to hold a value. Stripped, so a variable set to spaces
# reads as absent rather than as a key made of whitespace.
Trimmed = Annotated[str, StringConstraints(strip_whitespace=True)]
TrimmedRequired = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class Settings(BaseSettings):
    """Config from env vars, or a .env file if there is one.

    The numeric fields are bounded. A negative rate limit would silently switch
    the limiter off, which is the sort of thing you find out afterwards.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    cohere_api_key: Trimmed = ""
    cohere_model: TrimmedRequired = "command-a-plus-05-2026"

    # The SDK defaults to 300 seconds and 2 retries, so a slow upstream call can
    # tie up a connection for a quarter of an hour. Both are set explicitly.
    # allow_inf_nan is off or `inf` passes `gt=0` and removes the timeout again.
    request_timeout_seconds: float = Field(30.0, gt=0, allow_inf_nan=False)
    max_retries: int = Field(1, ge=0)

    # Ceiling on one generation, to bound a runaway call. Not a reservation:
    # billing follows tokens actually generated, so a roomy ceiling costs
    # nothing on an ordinary answer and a tight one just truncates it.
    #
    # 4096 rather than 1024 because thinking is on by default on the reasoning
    # models and shares this budget, and Cohere asks for at least 1K left for
    # the response itself. At 1024 total the model could reason away the whole
    # allowance and return nothing, which is exactly what was seen once.
    max_output_tokens: int = Field(4096, gt=0)

    # Both off by default: unset token means no auth, 0 means no rate limit.
    # That keeps local use and `curl` frictionless while making it one env var
    # to turn either on.
    # Printable ASCII only. Header bytes have no agreed charset, so a token
    # with, say, an "e-umlaut" goes out as UTF-8 and comes back decoded as
    # latin-1, never matching. Rejecting it at startup beats a service that
    # answers 401 forever for no visible reason.
    api_auth_token: Annotated[
        str, StringConstraints(strip_whitespace=True, pattern=r"^[\x21-\x7e]*$")
    ] = ""
    rate_limit_per_minute: int = Field(0, ge=0)

    # Name of a header holding the real client IP, for when this sits behind a
    # proxy and the TCP peer is the proxy rather than the caller. Off by
    # default: trusting a client-supplied header without a proxy in front lets
    # anyone forge their identity and walk straight past the rate limit.
    #
    # Constrained to the RFC 7230 field-name grammar. Anything outside it can't
    # be a header name, and looking one up raises rather than simply missing.
    client_ip_header: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, pattern=r"^[A-Za-z0-9!#$%&'*+.^_`|~-]*$"
        ),
    ] = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
