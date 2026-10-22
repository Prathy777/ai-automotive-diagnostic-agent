"""HTTP API for the simulator, diagnostic harness, and vehicle assistant."""
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import investigate
from .knowledge import KNOWLEDGE, search_knowledge
from .providers.base import ProviderConfigurationError, ProviderRequestError
from .providers.factory import create_provider, provider_health, provider_settings
from .simulator import Vehicle

SCENARIOS = ["normal", "misfire", "overheating", "alternator_failure", "wheel_imbalance", "fuel_air_mixture"]
SESSIONS: dict[str, dict[str, Any]] = {}
app = FastAPI(title="AI Automotive Diagnostic Agent", version="1.0.0",
              description="Simulation-first, explainable vehicle diagnostics and assistant")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])


class InvestigationRequest(BaseModel):
    scenario: str = Field(pattern="^(normal|misfire|overheating|alternator_failure|wheel_imbalance|fuel_air_mixture)$")
    max_steps: int = Field(default=8, ge=1, le=8)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


@app.get("/api/health")
async def health():
    settings = await provider_health()
    return {"status": "ok", **settings, "simulator_status": "ready"}


def demo_enabled() -> bool:
    return bool(provider_settings()["demo_mode"])


@app.get("/api/config")
async def config():
    return {**(await provider_health()), "model_timeout": int(os.getenv("MODEL_TIMEOUT", "60")),
            "scenarios": SCENARIOS}


@app.get("/api/vehicle")
async def vehicle():
    return {"vehicle_id": "SIM-001", "name": "Digital Twin Vehicle", "type": "simulated passenger car",
            "environment": "simulation"}


@app.get("/api/telemetry")
async def telemetry():
    return {"vehicle_id": "SIM-001", "telemetry": Vehicle().telemetry(), "environment": "simulation"}


@app.get("/api/knowledge")
async def knowledge(q: str = "", code: str | None = None):
    query = code or q
    return {"items": search_knowledge(query) if query else KNOWLEDGE}


@app.post("/api/investigations")
async def start(req: InvestigationRequest):
    try:
        result = await investigate(req.scenario, req.max_steps)
    except Exception as exc:
        raise HTTPException(500, "Investigation could not be completed") from exc
    SESSIONS[result["session_id"]] = result
    return result


@app.get("/api/investigations/{session_id}")
async def get_investigation(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Investigation not found")
    return SESSIONS[session_id]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    query = req.message.strip()
    knowledge = search_knowledge(query)
    session = SESSIONS.get(req.session_id or "")
    # Do not include the simulator's scenario label in assistant context.
    context = ({key: session.get(key) for key in ("observations", "hypotheses", "diagnosis", "confidence", "next_action")}
               if session else {"vehicle": (await telemetry())["telemetry"]})
    should_investigate = any(word in query.lower() for word in ("investigate", "diagnose", "check my", "find the cause"))
    try:
        model = create_provider("demo" if demo_enabled() else None)
        try:
            answer = await model.answer_question(query, context, knowledge)
        finally:
            await model.aclose()
        provider = model.name
    except (ProviderConfigurationError, ProviderRequestError) as exc:
        raise HTTPException(503, {"detail": str(exc), "code": getattr(exc, "code", "provider_configuration_error")}) from exc
    return {"answer": answer, "provider": provider, "session_id": req.session_id,
            "knowledge_references": [item["id"] for item in knowledge], "suggest_investigation": should_investigate}


# In the production image, serve the built static dashboard from this same origin.
static_root = Path(__file__).resolve().parents[2] / "frontend"
if not (static_root / "index.html").is_file():
    static_root = Path(__file__).resolve().parents[1] / "frontend"
if (static_root / "index.html").is_file():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    @app.get("/")
    async def dashboard():
        return FileResponse(static_root / "index.html")

    app.mount("/src", StaticFiles(directory=static_root / "src"), name="frontend-src")
