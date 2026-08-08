import math
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import cohere
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from app.chat_service import ChatError, ChatService
from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.security import RateLimiter, client_key, require_token

# What the caller is told for each class of upstream failure. Fixed strings, so
# no provider diagnostic or transport detail can escape through the response.
UPSTREAM_ERRORS = {
    429: (429, "Upstream rate limit reached. Try again shortly."),
    401: (503, "Server misconfigured: Cohere rejected the API key."),
    403: (503, "Server misconfigured: Cohere rejected the API key."),
}
DEFAULT_UPSTREAM_ERROR = (502, "The model provider could not be reached.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    # The client is an async context manager, and its __aexit__ closes the
    # httpx client it owns. Left to itself the SDK would allow 300s and 2
    # retries per call, so both are set here.
    async with cohere.AsyncClientV2(
        # It won't build without a token, so pass a dummy one when there's no
        # key. The app still starts and /chat returns a 503 saying what to fix.
        api_key=settings.cohere_api_key or "unset",
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    ) as client:
        app.state.cohere_client = client
        app.state.chat_service = ChatService(client=client, settings=settings)
        yield


app = FastAPI(title="Cohere Chat", version="0.1.0", lifespan=lifespan)


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


async def guard(request: Request, settings: SettingsDep) -> None:
    """Rate limit then auth, both no-ops unless they've been configured.

    The limit counts every attempt, including ones that go on to fail
    validation or authentication. That is deliberate. Counting only the calls
    that reach Cohere would leave a caller free to flood the service with
    malformed bodies or bad tokens, and the limiter is cheaper than the work it
    stands in front of. It does mean a client hammering an invalid request
    spends its own budget, which is the intended answer.

    Counting before the auth check keeps it consistent: a wrong token costs the
    same as a wrong body. The other order made a bad-token flood free.
    """
    wait = request.app.state.rate_limiter.check(client_key(request, settings))
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many requests.",
            headers={"Retry-After": str(math.ceil(wait))},
        )

    require_token(request, settings)


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


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(guard)],
)
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
        code, detail = UPSTREAM_ERRORS.get(exc.upstream_status, DEFAULT_UPSTREAM_ERROR)
        headers = (
            {"Retry-After": exc.retry_after} if code == 429 and exc.retry_after else None
        )
        raise HTTPException(status_code=code, detail=detail, headers=headers) from exc

    return ChatResponse(
        query=payload.query,
        response=result.text,
        finish_reason=result.finish_reason,
    )
