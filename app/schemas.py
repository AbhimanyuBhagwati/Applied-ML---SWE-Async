from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Strips first, then checks the length, so " " fails the same as "". Cohere
# rejects whitespace-only input anyway, and it's better to catch that here as a
# 422 than to spend an upstream call discovering it.
Query = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
]


class ChatRequest(BaseModel):
    # Reject unknown fields rather than ignoring them, so a typo'd key is a 422
    # instead of a silently different request.
    model_config = ConfigDict(extra="forbid")

    query: Query = Field(..., examples=["Who was the second person to walk on the moon?"])


class ToolCall(BaseModel):
    """A search the model asked for, and what came back."""

    name: str
    arguments: dict[str, Any]
    result_count: int
    results: list[dict[str, Any]]


class Citation(BaseModel):
    start: int | None = None
    end: int | None = None
    text: str | None = None
    sources: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    query: str
    response: str
    finish_reason: str | None = None
    tool_plan: str | None = None
    # The searches behind the answer, so the grounding can be checked rather
    # than taken on trust.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    """One recorded turn. Rhymes with ChatResponse, plus id and created_at.

    It keeps the grounding as well as the answer: a history that dropped the
    searches and citations would throw away the part task 2 exists to produce.
    """

    id: int
    created_at: str
    query: str
    response: str
    finish_reason: str | None = None
    tool_plan: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class HistoryPage(BaseModel):
    # `total` is what says the page isn't the whole history: without it, a
    # short page and the end of the history look the same.
    total: int
    limit: int
    offset: int
    entries: list[HistoryEntry]


class HealthResponse(BaseModel):
    # "degraded" is served with a 503, so a load balancer stops routing here.
    status: Literal["ok", "degraded"]
    model: str
    cohere_api_key_configured: bool
