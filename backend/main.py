"""
backend/main.py
---------------
FastAPI application entry point for the NeuroAgent backend.

Server startup
--------------
    # Development (auto-reload on file changes)
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2

Environment variables (set in .env or system environment)
---------------------------------------------------------
    MOCK_MODE                 true | false        (default: false → real pipeline)
    MAX_UPLOAD_MB             float               (default: 500)
    LOW_CONFIDENCE_THRESHOLD  float               (default: 0.60)
    HIGH_CONFIDENCE_THRESHOLD float               (default: 0.80)
    ALLOWED_ORIGINS           comma-separated URLs (default: *)
    LOG_LEVEL                 DEBUG|INFO|WARNING   (default: INFO)

Registered routes
-----------------
    GET  /health       — liveness probe (no auth required)
    GET  /             — redirect to /docs
    POST /api/analyze  — main analysis endpoint (see routes/analyze.py)
    GET  /docs         — Swagger UI (auto-generated)
    GET  /redoc        — ReDoc UI (auto-generated)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.pipeline import OrchestratorPipeline
from backend.routes.analyze import router as analyze_router

# ── Environment ────────────────────────────────────────────────────────────────

load_dotenv()  # Load from c:\Neuro Agent\.env if present

# ── Logging ────────────────────────────────────────────────────────────────────

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuroagent.backend")


# ── CORS ───────────────────────────────────────────────────────────────────────

# In development: allow all origins (Streamlit runs on a different port).
# In production: set ALLOWED_ORIGINS=http://your-frontend-domain.com in .env.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_ALLOWED_ORIGINS: list[str] = (
    ["*"] if _raw_origins.strip() == "*" else [o.strip() for o in _raw_origins.split(",")]
)


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    ON STARTUP:
      - Instantiate OrchestratorPipeline (loads/compiles LangGraph graph in real mode).
      - Attach pipeline to app.state so endpoints can access it via request.app.state.pipeline.

    ON SHUTDOWN:
      - Log shutdown (no explicit teardown needed; Python GC handles the rest).
    """
    logger.info("=" * 60)
    logger.info("NeuroAgent Backend starting up")
    logger.info("  Mock mode     : %s", os.getenv("MOCK_MODE", "false"))
    logger.info("  Log level     : %s", _LOG_LEVEL)
    logger.info("  CORS origins  : %s", _ALLOWED_ORIGINS)
    logger.info("=" * 60)

    # Load pipeline once — expensive in real mode (LangGraph compile + model weights)
    app.state.pipeline = OrchestratorPipeline()
    logger.info("Pipeline ready: %r", app.state.pipeline)

    yield  # Application runs here

    logger.info("NeuroAgent Backend shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="NeuroAgent API",
    description=(
        "REST API for the NeuroAgent multi-agent brain MRI diagnosis system.\n\n"
        "Upload an MRI scan and patient metadata to receive a structured radiology "
        "report with Grad-CAM heatmap, confidence score, and cited literature.\n\n"
        "**Pipeline agents**: Vision · Clinical History · RAG Literature · "
        "Report Generation · Verification · Explainability\n\n"
        "**Orchestration**: LangGraph\n\n"
        "**Mode**: Set `MOCK_MODE=false` (default) in `.env` for the live pipeline. "
        "Set `MOCK_MODE=true` for UI testing without GPU."
    ),
    version="0.1.0",
    contact={
        "name": "Nandan Bomale",
        "url": "https://github.com/Nandan-Bomale/neuro-agent",
    },
    license_info={
        "name": "Academic — not for clinical use",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "analysis",
            "description": "Brain MRI analysis endpoint — the core of the system.",
        },
        {
            "name": "health",
            "description": "Liveness and readiness probes for deployment monitoring.",
        },
    ],
)


# ── Middleware ─────────────────────────────────────────────────────────────────


app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────


app.include_router(analyze_router)


# ── Built-in endpoints ─────────────────────────────────────────────────────────


@app.get(
    "/health",
    tags=["health"],
    summary="Liveness probe",
    response_description="Returns 200 OK with pipeline mode when the server is running.",
)
async def health_check():
    """
    Simple liveness probe.

    Returns basic server status and pipeline operating mode.
    Does **not** check whether the vision model weights are loaded — it only
    confirms the server process is alive and the pipeline object was created.

    Use this endpoint in Docker health checks and monitoring dashboards.
    """
    return {
        "status": "ok",
        "service": "neuroagent-backend",
        "version": app.version,
        "mock_mode": os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes"),
        "pipeline_mode": "mock" if os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes") else "real",
    }


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")
