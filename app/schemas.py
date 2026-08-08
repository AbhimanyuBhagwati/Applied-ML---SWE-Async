from typing import Annotated, Literal

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

    query: Query = Field(
        ..., examples=["Who was the second person to walk on the moon?"]
    )


class ChatResponse(BaseModel):
    query: str
    response: str
    finish_reason: str | None = None


class HealthResponse(BaseModel):
    # "degraded" is served with a 503, so a load balancer stops routing here.
    status: Literal["ok", "degraded"]
    model: str
    cohere_api_key_configured: bool
