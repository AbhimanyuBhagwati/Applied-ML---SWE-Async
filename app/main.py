import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict
from typing import Annotated

import cohere
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from app.chat_service import ChatError, ChatService, preview
from app.config import DEFAULT_USER_AGENT, Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ToolCall
from app.wikipedia import WikipediaClient

# What the caller is told for each class of upstream failure. Fixed strings, so
# no provider diagnostic or transport detail can escape through the response.
UPSTREAM_ERRORS = {
    429: (429, "Upstream rate limit reached. Try again shortly."),
    401: (503, "Server misconfigured: Cohere rejected the API key."),
    403: (503, "Server misconfigured: Cohere rejected the API key."),
    # Cohere's own code for an invalid token.
    498: (503, "Server misconfigured: Cohere rejected the API key."),
}
DEFAULT_UPSTREAM_ERROR = (502, "The model provider could not be reached.")
# Upstream saying it timed out means the same to a caller as us timing out.
TIMEOUT_STATUSES = frozenset({408, 504})

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    if settings.wikipedia_user_agent == DEFAULT_USER_AGENT:
        # Wikimedia's policy is that clients identify themselves with a way
        # to reach someone, and they block ones that don't.
        log.warning(
            "WIKIPEDIA_USER_AGENT is unset. Wikimedia's User-Agent policy "
            "asks for a contact address and they may block requests without one."
        )

    # Ours, for Wikipedia. Cohere brings its own. Registered on the stack
    # straight away so it still gets closed if the Cohere client fails to open.
    stack = AsyncExitStack()
    http_client = await stack.enter_async_context(httpx.AsyncClient())
    app.state.http_client = http_client
    app.state.wikipedia = WikipediaClient(settings, http_client)

    # The Cohere client is an async context manager whose __aexit__ closes the
    # httpx client it owns. Left alone the SDK allows 300s and 2 retries per
    # call, so both are set here.
    async with (
        stack,
        cohere.AsyncClientV2(
            # It won't build without a token, so pass a dummy one when there's no
            # key. The app still starts and /chat returns a 503 saying what to fix.
            api_key=settings.cohere_api_key or "unset",
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        ) as client,
    ):
        app.state.cohere_client = client
        app.state.chat_service = ChatService(
            client=client,
            wikipedia=app.state.wikipedia,
            settings=settings,
        )
        yield


app = FastAPI(title="Cohere Chat", version="0.1.0", lifespan=lifespan)


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(settings: SettingsDep, response: Response) -> HealthResponse:
    configured = bool(settings.cohere_api_key)
    if not configured:
        # Readiness, not just liveness. Every /chat call will fail without a
        # key, so this instance should not be receiving traffic.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if configured else "degraded",
        model=settings.cohere_model,
        cohere_api_key_configured=configured,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    payload: ChatRequest, service: ChatServiceDep, settings: SettingsDep
) -> ChatResponse:
    if not settings.cohere_api_key:
        raise HTTPException(
            status_code=503,
            detail="COHERE_API_KEY is not configured. Set it in .env and restart.",
        )

    try:
        result = await service.chat(payload.query)
    except ChatError as exc:
        if exc.timed_out or exc.upstream_status in TIMEOUT_STATUSES:
            raise HTTPException(
                status_code=504, detail="The request took too long."
            ) from exc
        code, detail = UPSTREAM_ERRORS.get(exc.upstream_status, DEFAULT_UPSTREAM_ERROR)
        headers = (
            {"Retry-After": exc.retry_after} if code == 429 and exc.retry_after else None
        )
        raise HTTPException(status_code=code, detail=detail, headers=headers) from exc

    return ChatResponse(
        query=payload.query,
        response=result.text,
        finish_reason=result.finish_reason,
        tool_plan=result.tool_plan,
        # Dataclasses in, pydantic out; pydantic won't coerce between them.
        # Results are trimmed here: the model needed the full articles, the
        # caller is only checking what the answer was built on.
        tool_calls=[
            ToolCall(
                **{
                    **asdict(call),
                    "results": [preview(result) for result in call.results],
                }
            )
            for call in result.tool_calls
        ],
        citations=result.citations,
    )
