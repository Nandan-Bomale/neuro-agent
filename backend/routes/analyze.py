"""
backend/routes/analyze.py
--------------------------
POST /analyze  — the single endpoint the Streamlit frontend calls.

Request format (multipart/form-data)
-------------------------------------
  scan_file    : UploadFile  — MRI scan (.nii, .nii.gz, .png, .jpg, .jpeg)
  patient_json : str (Form) — JSON-encoded PatientData

The endpoint:
  1. Validates the uploaded file type and size.
  2. Parses and validates the patient JSON into a PatientData model.
  3. Saves the upload to a secure temp file (cleaned up after the call).
  4. Calls pipeline.run() — which either hits the real LangGraph orchestrator
     or the mock, depending on PIPELINE_MODE env var.
  5. Returns a fully populated AnalysisResponse as JSON.

Error handling
--------------
  400  Bad request   — unsupported file type, missing patient_json
  413  Too large     — file exceeds MAX_UPLOAD_MB
  422  Validation    — patient_json fails Pydantic validation
  500  Server error  — pipeline raised an unexpected exception

Dependency injection
--------------------
  The OrchestratorPipeline instance is injected via FastAPI's app.state so the
  model is loaded once at startup, not per-request.

  Access pattern in the endpoint:
      pipeline: OrchestratorPipeline = request.app.state.pipeline
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.schemas import AnalysisResponse, ErrorResponse, PatientData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

# ── Configuration ──────────────────────────────────────────────────────────────

#: Maximum allowed upload size in megabytes.
MAX_UPLOAD_MB: float = float(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES: int = int(MAX_UPLOAD_MB * 1024 * 1024)

#: Allowed MRI file extensions.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".nii", ".gz", ".png", ".jpg", ".jpeg", ".dcm"}
)

#: Temporary directory for storing uploaded scans during processing.
_TEMP_DIR = Path(tempfile.gettempdir()) / "neuroagent_uploads"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _validate_file_extension(filename: str) -> str:
    """
    Validate that the uploaded file has an allowed extension.

    Handles double extensions like .nii.gz correctly.

    Args:
        filename: Original filename from the upload.

    Returns:
        The validated suffix (e.g. '.nii', '.gz').

    Raises:
        HTTPException 400 if the extension is not in ALLOWED_EXTENSIONS.
    """
    p = Path(filename)
    # Handle .nii.gz
    suffix = "".join(p.suffixes).lower()
    if not suffix:
        suffix = p.suffix.lower()
    # Check each suffix component
    for ext in p.suffixes:
        if ext.lower() in ALLOWED_EXTENSIONS:
            return suffix
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        ),
    )


async def _save_upload_to_temp(upload: UploadFile, suffix: str) -> Path:
    """
    Stream the uploaded file to a named temp file and return its path.

    Args:
        upload: FastAPI UploadFile object.
        suffix: File extension (for correct downstream handling).

    Returns:
        Path to the saved temp file.

    Raises:
        HTTPException 413 if the file exceeds MAX_UPLOAD_BYTES.
        HTTPException 500 if writing fails.
    """
    unique_name = f"scan_{uuid.uuid4().hex}{suffix}"
    temp_path = _TEMP_DIR / unique_name

    try:
        total_bytes = 0
        with temp_path.open("wb") as f:
            while chunk := await upload.read(1024 * 256):  # 256 KB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File too large ({total_bytes / 1e6:.1f} MB). "
                            f"Maximum allowed: {MAX_UPLOAD_MB} MB."
                        ),
                    )
                f.write(chunk)

        logger.info(
            "Saved upload '%s' → %s (%.2f MB)",
            upload.filename,
            temp_path,
            total_bytes / 1e6,
        )
        return temp_path

    except HTTPException:
        raise
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file. Please try again.",
        ) from exc


# ── Endpoint ───────────────────────────────────────────────────────────────────


@router.post(
    "/analyze",
    summary="Analyse a brain MRI scan (Streaming SSE)",
    description="Streams execution events (AgentExecutionEvent) in real-time as Server-Sent Events.",
    operation_id="analyze_mri",
)
async def analyze(
    request: Request,
    scan_file: Annotated[
        UploadFile,
        File(description="MRI scan file (.nii, .nii.gz, .png, .jpg, .jpeg)"),
    ],
    patient_json: Annotated[
        str,
        Form(description="JSON string encoding the PatientData schema.")
    ],
):
    run_id = str(uuid.uuid4())
    logger.info(
        "POST /analyze (stream) | run_id=%s | file='%s' | size=%s",
        run_id,
        scan_file.filename,
        scan_file.size,
    )

    if not scan_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload.")
    suffix = _validate_file_extension(scan_file.filename)

    try:
        patient_dict = json.loads(patient_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"patient_json is not valid JSON: {exc}")

    try:
        patient_data = PatientData.model_validate(patient_dict)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Validation failed: {exc.errors()}")

    temp_path = await _save_upload_to_temp(scan_file, suffix)
    pipeline = request.app.state.pipeline

    # Create an async generator to feed the StreamingResponse
    async def event_generator():
        try:
            async for event in pipeline.stream_run(
                scan_path=str(temp_path),
                patient_data=patient_data.to_pipeline_dict(),
                run_id=run_id
            ):
                yield event
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
                logger.debug("Cleaned up temp file: %s", temp_path)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

from fastapi.responses import StreamingResponse

@router.post("/analyze/stream")
async def analyze_stream(
    request: Request,
    scan_file: UploadFile = File(...),
    patient_json: str = Form(...)
):
    suffix = _validate_file_extension(scan_file.filename)
    patient_dict = json.loads(patient_json)
    patient_data = PatientData.model_validate(patient_dict)
    temp_path = await _save_upload_to_temp(scan_file, suffix)
    run_id = str(uuid.uuid4())
    
    async def event_generator():
        try:
            pipeline = request.app.state.pipeline
            async for event in pipeline.stream_run(str(temp_path), patient_data.to_pipeline_dict(), run_id):
                yield event
        finally:
            if temp_path and temp_path.exists():
                try: temp_path.unlink()
                except: pass
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")
