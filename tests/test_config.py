import pytest
from pydantic import ValidationError

from app.config import Settings


def clean_settings(monkeypatch, **overrides):
    """Settings built from nothing: no .env file, no inherited env vars."""
    for name in ("COHERE_API_KEY", "COHERE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, **overrides)


def test_defaults(monkeypatch):
    settings = clean_settings(monkeypatch)

    assert settings.cohere_model == "command-a-plus-05-2026"
    # Empty rather than None, so /chat can 503 instead of the app failing to boot.
    assert settings.cohere_api_key == ""


def test_env_vars_win(monkeypatch):
    monkeypatch.setenv("COHERE_MODEL", "command-r-08-2024")
    monkeypatch.setenv("COHERE_API_KEY", "from-the-environment")

    settings = Settings(_env_file=None)

    assert settings.cohere_model == "command-r-08-2024"
    assert settings.cohere_api_key == "from-the-environment"


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_timeout_seconds", -1),
        ("request_timeout_seconds", 0),
        ("max_retries", -1),
        ("max_output_tokens", 0),
        ("max_output_tokens", -100),
        ("rate_limit_per_minute", -1),
        ("cohere_model", ""),
    ],
    ids=[
        "negative-timeout",
        "zero-timeout",
        "negative-retries",
        "zero-tokens",
        "negative-tokens",
        "negative-rate-limit",
        "blank-model",
    ],
)
def test_nonsense_values_are_refused(field, value):
    """A negative rate limit used to just switch the limiter off, quietly."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_zero_is_allowed_where_it_means_off():
    settings = Settings(_env_file=None, rate_limit_per_minute=0, max_retries=0)

    assert settings.rate_limit_per_minute == 0
    assert settings.max_retries == 0


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_infinite_timeout_is_refused(value):
    """`inf` sails past gt=0 and quietly removes the timeout again."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_timeout_seconds=value)


@pytest.mark.parametrize("blank", ["   ", "\t", "\n "])
def test_whitespace_only_strings_are_not_values(blank):
    """A key of spaces would read as configured and then fail upstream."""
    settings = Settings(_env_file=None, cohere_api_key=blank, api_auth_token=blank)

    assert settings.cohere_api_key == ""
    assert settings.api_auth_token == ""

    with pytest.raises(ValidationError):
        Settings(_env_file=None, cohere_model=blank)


def test_values_are_trimmed():
    settings = Settings(_env_file=None, cohere_api_key="  key  ", cohere_model=" m ")

    assert settings.cohere_api_key == "key"
    assert settings.cohere_model == "m"
