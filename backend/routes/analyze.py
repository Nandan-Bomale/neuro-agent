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
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or missing fields"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        422: {"model": ErrorResponse, "description": "Patient data validation error"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
    summary="Analyse a brain MRI scan",
    description=(
        "Upload an MRI scan file and patient metadata (as JSON string). "
        "The pipeline runs all seven agents — Vision, Clinical History, RAG Literature, "
        "Report Generation, Verification, and Explainability — and returns a structured "
        "radiology report with a base64 Grad-CAM heatmap image."
    ),
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
        Form(
            description=(
                "JSON string encoding the PatientData schema. "
                'Example: {"age": 45, "sex": "M", "symptoms": ["headache"], '
                '"medical_history": [], "medications": [], "referring_notes": "", '
                '"scan_modality": "FLAIR"}'
            )
        ),
    ],
) -> AnalysisResponse:
    """
    POST /api/analyze

    Accepts multipart/form-data with:
      - scan_file     : UploadFile
      - patient_json  : str (JSON-encoded PatientData)

    Returns AnalysisResponse on success.
    """
    run_id = str(uuid.uuid4())
    logger.info(
        "POST /analyze | run_id=%s | file='%s' | size=%s",
        run_id,
        scan_file.filename,
        scan_file.size,
    )

    # ── 1. Validate file extension ────────────────────────────────────────────
    if not scan_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload.",
        )
    suffix = _validate_file_extension(scan_file.filename)

    # ── 2. Parse + validate patient JSON ─────────────────────────────────────
    try:
        patient_dict = json.loads(patient_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"patient_json is not valid JSON: {exc}",
        ) from exc

    try:
        patient_data = PatientData.model_validate(patient_dict)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Patient data validation failed: {exc.errors()}",
        ) from exc

    # ── 3. Save upload to temp file ───────────────────────────────────────────
    temp_path: Path | None = None
    try:
        temp_path = await _save_upload_to_temp(scan_file, suffix)

        # ── 4. Run the pipeline ───────────────────────────────────────────────
        pipeline = request.app.state.pipeline
        result: AnalysisResponse = await pipeline.run(
            scan_path=str(temp_path),
            patient_data=patient_data.to_pipeline_dict(),
            run_id=run_id,
        )

        logger.info(
            "POST /analyze complete | run_id=%s | confidence=%.3f | status=%s | time=%.2fs",
            run_id,
            result.confidence,
            result.pipeline_status,
            result.processing_time_s,
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline error | run_id=%s | %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {exc}",
        ) from exc

    finally:
        # ── 5. Always clean up the temp file ─────────────────────────────────
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
                logger.debug("Cleaned up temp file: %s", temp_path)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean up temp file %s: %s", temp_path, cleanup_exc)
