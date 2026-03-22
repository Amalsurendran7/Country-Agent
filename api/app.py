"""
FastAPI 0.135.1 application.

Pydantic v2 patterns used:
  - model_config = ConfigDict(...)  instead of inner class Config
  - ConfigDict(str_strip_whitespace=True)  handles .strip() declaratively
  - @field_validator with mode="after" where input is already coerced

FastAPI patterns used:
  - asynccontextmanager lifespan  (on_event is deprecated)
  - Annotated + Depends  for dependency injection (request-ID)
  - Exception handler returns JSONResponse with consistent shape

LangGraph 1.1.3 integration:
  - MemorySaver checkpointer requires thread_id in every ainvoke() config
  - We use the per-request UUID as the thread_id for traceability
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.graph import country_agent
from config.settings import settings

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "Country Agent starting | intent=%s | synthesis=%s",
        settings.intent_model,
        settings.synthesis_model,
    )
    yield
    logger.info("Country Agent stopped.")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Country Information Agent",
    description=(
        "LangGraph agent answering natural-language questions about countries "
        "via Together.ai inference and the REST Countries API."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,   # strips whitespace on all str fields
        frozen=True,                 # immutable after creation
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What currency does Japan use?"],
    )

    @field_validator("question", mode="after")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("question must not be blank after stripping whitespace")
        return v


class QueryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id:      str
    question:        str
    answer:          str
    country:         Optional[str]
    fields_resolved: list[str]
    sources:         list[str]
    latency_ms:      int


class HealthResponse(BaseModel):
    status:          str
    version:         str
    intent_model:    str
    synthesis_model: str


# ── Request-ID middleware ──────────────────────────────────────────────────

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


# ── Dependency: inject request_id into endpoints ───────────────────────────

def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


RequestID = Annotated[str, Depends(_get_request_id)]


# ── Global error handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error | path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "service":         "Country Information Agent",
        "version":         "3.0.0",
        "llm_provider":    "Together.ai",
        "intent_model":    settings.intent_model,
        "synthesis_model": settings.synthesis_model,
        "docs":            "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    return HealthResponse(
        status="ok",
        version="3.0.0",
        intent_model=settings.intent_model,
        synthesis_model=settings.synthesis_model,
    )


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query(body: QueryRequest, request_id: RequestID):
    logger.info("[%s] question=%r", request_id, body.question)

    start = time.monotonic()

    # MemorySaver requires thread_id — use request_id for full traceability
    config = {"configurable": {"thread_id": request_id}}
    final_state = await country_agent.ainvoke(
        {"user_query": body.question},
        config=config,
    )

    latency_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "[%s] done | %dms | country=%r | fields=%s",
        request_id,
        latency_ms,
        final_state.get("country_name"),
        final_state.get("requested_fields", []),
    )

    return QueryResponse(
        request_id=request_id,
        question=body.question,
        answer=final_state.get("answer", "No answer produced."),
        country=final_state.get("country_name"),
        fields_resolved=final_state.get("requested_fields") or [],
        sources=final_state.get("sources") or [],
        latency_ms=latency_ms,
    )
