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
        ("request_timeout_seconds", float("inf")),
        ("max_output_tokens", 0),
        ("wikipedia_extract_chars", 9000),
        ("cohere_model", ""),
    ],
    ids=["negative", "infinite", "zero", "over-the-api-ceiling", "blank"],
)
def test_nonsense_values_are_refused(field, value):
    """A nonsense value should fail at startup, not be quietly accepted."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_the_shipped_env_example_actually_starts(tmp_path, monkeypatch):
    """A fresh clone follows the README, so this file has to work as-is.

    A blank WIKIPEDIA_USER_AGENT= line overrode the default and the app
    refused to boot on the documented setup path.
    """
    from pathlib import Path

    example = Path(__file__).parent.parent / ".env.example"
    env = tmp_path / ".env"
    env.write_text(example.read_text() + "\nCOHERE_API_KEY=some-key\n")
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.cohere_api_key == "some-key"
    assert settings.wikipedia_user_agent
