from functools import lru_cache
from typing import Annotated

from pydantic import Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

# Strings that are meant to hold a value. Stripped, so a variable set to spaces
# reads as absent rather than as a key made of whitespace.
Trimmed = Annotated[str, StringConstraints(strip_whitespace=True)]
TrimmedRequired = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Not a real address. Wikimedia asks for one that reaches a human, so this is
# written to be obviously unset rather than to look plausible.
DEFAULT_USER_AGENT = "cohere-chat/0.1 (WIKIPEDIA_USER_AGENT is unset)"


class Settings(BaseSettings):
    """Config from env vars, or a .env file if there is one.

    The numeric fields are bounded, so a nonsense value fails at startup
    rather than quietly disabling whatever it controls.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    cohere_api_key: Trimmed = ""
    cohere_model: TrimmedRequired = "command-a-plus-05-2026"

    # The SDK would otherwise allow 300s and 2 retries. allow_inf_nan is off
    # because `inf` passes gt=0 and removes the timeout again.
    request_timeout_seconds: float = Field(30.0, gt=0, allow_inf_nan=False)
    max_retries: int = Field(1, ge=0)

    # A ceiling, not a reservation: billing follows tokens generated. Roomy
    # because thinking is on by default and shares this budget, and Cohere asks
    # for at least 1K left for the answer itself.
    max_output_tokens: int = Field(4096, gt=0)

    # Separate settings rather than derived: index.php isn't guaranteed to sit
    # next to api.php on every wiki.
    wikipedia_api_url: TrimmedRequired = "https://en.wikipedia.org/w/api.php"
    wikipedia_article_url: TrimmedRequired = "https://en.wikipedia.org/wiki/"
    # Wikimedia's policy asks for a contact address and they may block clients
    # without one. Startup warns while this is still the placeholder.
    wikipedia_user_agent: TrimmedRequired = DEFAULT_USER_AGENT
    wikipedia_timeout_seconds: float = Field(10.0, gt=0, allow_inf_nan=False)
    # srlimit allows 500, which is far more context than the model needs.
    wikipedia_max_results: int = Field(5, gt=0, le=50)
    # Capped at the API's own exchars ceiling, so the local limit can't claim
    # to be something exchars will never honour.
    wikipedia_extract_chars: int = Field(1200, gt=0, le=1200)

    # Where the query history lives. Relative, so it sits beside wherever the
    # app was started from.
    history_database_path: TrimmedRequired = "history.db"

    # Rounds of searching, not Cohere calls: there is always one more call
    # after the last round to turn the results into an answer.
    max_tool_rounds: int = Field(4, gt=0, le=10)
    # Without this the round cap bounds nothing: one response can ask for 25.
    max_tool_calls_per_round: int = Field(4, gt=0, le=20)

    # Per-round caps multiply rather than bound: ten rounds of twenty searches
    # is 800 requests. These bound the request as a whole. Context size isn't
    # one of them, because it can't be checked before fetching; the bound there
    # is static, at max_total_searches x wikipedia_max_results x
    # wikipedia_extract_chars, which at the defaults is about 48K characters.
    max_total_searches: int = Field(8, gt=0, le=100)
    tool_loop_seconds: float = Field(45.0, gt=0, allow_inf_nan=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
