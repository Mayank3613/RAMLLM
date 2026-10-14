"""FastAPI server providing OpenAI-compatible endpoints and RAMLLM extensions."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ramllm.backends import LlamaCppAdapter
from ramllm.backends.base import BackendProcess
from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.governor.monitor import MemoryGovernor, TierMetrics
from ramllm.core.planner import create_execution_plans
from ramllm.core.profiler import probe_hardware
from ramllm.core.schemas import ExecutionPlan, HardwareProfile, ModelInfo, UserRequest


class ReplanRequest(BaseModel):
    """Payload for dynamic runtime replanning."""

    vram_budget_bytes: dict[int, int] | None = None
    ram_budget_bytes: int | None = None
    context: int | None = None


class ServerState:
    """Shared state for the RAMLLM API server."""

    def __init__(self) -> None:
        self.model_path: str = ""
        self.model_info: ModelInfo | None = None
        self.hw_profile: HardwareProfile | None = None
        self.user_request: UserRequest | None = None
        self.active_plan: ExecutionPlan | None = None
        self.backend_process: BackendProcess | None = None
        self.governor: MemoryGovernor | None = None
        self.internal_port: int = 18080
        self.adapter: LlamaCppAdapter = LlamaCppAdapter()
        self.lock: asyncio.Lock = asyncio.Lock()


state = ServerState()


def launch_active_session() -> None:
    """Plan and launch the backend session with memory governor."""
    if not state.model_info or not state.hw_profile or not state.user_request:
        raise RuntimeError("Server state not initialized.")

    plans = create_execution_plans(state.model_info, state.hw_profile, state.user_request)
    chosen = plans[0]
    if chosen.verdict == "INFEASIBLE":
        raise RuntimeError(f"Plan is INFEASIBLE: {chosen.warnings}")

    state.active_plan = chosen
    state.backend_process = state.adapter.launch(chosen, port=state.internal_port, timeout_s=40.0)
    state.governor = MemoryGovernor(state.backend_process, state.hw_profile)
    state.governor.start()


def shutdown_active_session() -> None:
    """Stop memory governor and terminate backend engine."""
    if state.governor:
        state.governor.stop()
        state.governor = None
    if state.backend_process:
        state.adapter.stop(state.backend_process)
        state.backend_process = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifespan managing backend lifecycle."""
    if state.model_path:
        launch_active_session()
    yield
    shutdown_active_session()


api_app = FastAPI(
    title="RAMLLM Server",
    description="OpenAI-compatible LLM inference server with strict RAM/VRAM budget governor.",
    version="0.1.0",
    lifespan=lifespan,
)


@api_app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}


@api_app.get("/ramllm/plan")
async def get_plan() -> ExecutionPlan:
    """Return the currently executing ExecutionPlan."""
    if not state.active_plan:
        raise HTTPException(status_code=503, detail="No active plan scheduled.")
    return state.active_plan


@api_app.get("/ramllm/metrics")
async def get_metrics() -> TierMetrics:
    """Return real-time per-tier memory metrics and governor warnings."""
    if not state.governor:
        raise HTTPException(status_code=503, detail="Governor not running.")
    return state.governor.get_latest_metrics()


@api_app.post("/ramllm/replan")
async def replan(req: ReplanRequest) -> ExecutionPlan:
    """Change memory budgets or context at runtime and replan execution."""
    async with state.lock:
        if not state.user_request or not state.model_info or not state.hw_profile:
            raise HTTPException(status_code=503, detail="Server not initialized with model.")

        shutdown_active_session()

        # Update request
        ctx = req.context or state.user_request.min_context
        vram_dict = (
            req.vram_budget_bytes
            if req.vram_budget_bytes is not None
            else state.user_request.vram_budget_bytes
        )
        ram_bytes = (
            req.ram_budget_bytes
            if req.ram_budget_bytes is not None
            else state.user_request.ram_budget_bytes
        )

        state.user_request = UserRequest(
            model=state.model_path,
            vram_budget_bytes=vram_dict,
            ram_budget_bytes=ram_bytes,
            min_context=ctx,
            target_context=ctx,
        )

        launch_active_session()
        if not state.active_plan:
            raise HTTPException(status_code=500, detail="Failed to establish new plan.")
        return state.active_plan


@api_app.get("/v1/models")
async def list_models_api() -> dict[str, Any]:
    """OpenAI-compatible models list endpoint."""
    model_name = state.model_info.arch if state.model_info else "ramllm-model"
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ramllm",
            }
        ],
    }


@api_app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """OpenAI-compatible chat completions proxy with streaming SSE support."""
    body = await request.json()
    is_stream = body.get("stream", False)
    target_url = f"http://127.0.0.1:{state.internal_port}/v1/chat/completions"

    client = httpx.AsyncClient(timeout=120.0)

    if is_stream:

        async def event_generator() -> AsyncGenerator[bytes, None]:
            try:
                async with client.stream("POST", target_url, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            resp = await client.post(target_url, json=body)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        finally:
            await client.aclose()


@api_app.post("/v1/completions")
async def completions(request: Request) -> Response:
    """OpenAI-compatible text completions proxy."""
    body = await request.json()
    is_stream = body.get("stream", False)
    target_url = f"http://127.0.0.1:{state.internal_port}/v1/completions"

    client = httpx.AsyncClient(timeout=120.0)

    if is_stream:

        async def event_generator() -> AsyncGenerator[bytes, None]:
            try:
                async with client.stream("POST", target_url, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            resp = await client.post(target_url, json=body)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        finally:
            await client.aclose()


def create_app(
    model_path: str,
    vram_budget_bytes: dict[int, int] | None = None,
    ram_budget_bytes: int | None = None,
    context: int = 4096,
    internal_port: int = 18080,
) -> FastAPI:
    """Configure server state and return FastAPI application instance."""
    state.model_path = model_path
    state.internal_port = internal_port
    state.model_info = analyze_gguf(model_path)
    state.hw_profile = probe_hardware(quick=True)
    state.user_request = UserRequest(
        model=model_path,
        vram_budget_bytes=vram_budget_bytes,
        ram_budget_bytes=ram_budget_bytes,
        min_context=context,
        target_context=context,
    )
    return api_app
