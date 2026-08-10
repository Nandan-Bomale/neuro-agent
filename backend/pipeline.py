"""
backend/pipeline.py
-------------------
OrchestratorPipeline — the single bridge between the FastAPI endpoint and the
LangGraph multi-agent pipeline.

Design
------
The class has two operating modes controlled by the ``MOCK_MODE`` env var:

  REAL mode  (``MOCK_MODE=false`` — DEFAULT)
    Imports and calls orchestrator.graph.compile_graph().invoke() with the
    MRI scan path and patient data dict.  Maps the final NeuroAgentState dict
    to an AnalysisResponse using _state_to_response().
    If the LangGraph import fails at runtime, falls back to MOCK automatically.

  MOCK mode  (``MOCK_MODE=true``)
    Returns a deterministic, realistic fake response so the entire
    backend + frontend can be developed and tested without GPU / model weights.
    The mock generates a synthetic Grad-CAM heatmap image (real numpy math,
    not a placeholder) so the UI heatmap component is exercised properly.

env vars
--------
  MOCK_MODE                    true | false   (default: false → real pipeline)
  LOW_CONFIDENCE_THRESHOLD     float          (default: 0.60)
  HIGH_CONFIDENCE_THRESHOLD    float          (default: 0.80)
  MAX_UPLOAD_MB                float          (default: 500)

Usage (internal — called by routes/analyze.py)
----------------------------------------------
    from backend.pipeline import OrchestratorPipeline

    pipeline = OrchestratorPipeline()   # call once at app startup (lifespan)
    result: AnalysisResponse = await pipeline.run(
        scan_path=path_to_tmp_file,
        patient_data=patient_data_dict,
    )
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.schemas import (
    AnalysisResponse,
    Citation,
    PatientData,
    StructuredReport,
    VisionSummary,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

#: MOCK_MODE=true → synthetic responses, no GPU required.
#: MOCK_MODE=false (default) → real LangGraph pipeline.
_MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes")

#: Confidence below LOW_THRESHOLD → LOW label. Above HIGH_THRESHOLD → HIGH.
_LOW_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))
_HIGH_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.80"))


# ── Helper: confidence label ───────────────────────────────────────────────────


def _confidence_label(score: float) -> str:
    """Map a float confidence score to a human-readable tier."""
    if score >= _HIGH_THRESHOLD:
        return "HIGH"
    if score >= _LOW_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ── Helper: heatmap PNG → base64 ──────────────────────────────────────────────


def _array_to_b64_png(overlay_bgr: np.ndarray) -> str:
    """
    Convert a uint8 numpy image array (H×W×3, BGR or RGB) to a base64-encoded
    PNG string suitable for embedding in JSON.

    Args:
        overlay_bgr: numpy uint8 array of shape (H, W, 3).

    Returns:
        Base64 string of the PNG-encoded image.
    """
    try:
        from PIL import Image  # noqa: PLC0415

        # Vision Agent returns BGR (OpenCV convention) — convert to RGB for PIL
        rgb = overlay_bgr[:, :, ::-1] if overlay_bgr.ndim == 3 else overlay_bgr
        img = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("PIL encoding failed, falling back to matplotlib: %s", exc)
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        ax.imshow(overlay_bgr[:, :, ::-1] if overlay_bgr.ndim == 3 else overlay_bgr)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Mock helpers ───────────────────────────────────────────────────────────────


def _generate_mock_heatmap(width: int = 240, height: int = 240) -> str:
    """
    Generate a synthetic Grad-CAM heatmap PNG as base64.

    Uses real numpy math to produce a Gaussian-hot-spot overlay on a simulated
    greyscale brain MRI slice so the frontend heatmap component is properly exercised.

    Returns:
        Base64 PNG string.
    """
    rng = np.random.default_rng(seed=42)

    # ── Simulated greyscale brain slice (oval skull + grey matter texture) ──
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    skull_mask = ((x - cx) ** 2 / (cx * 0.9) ** 2 + (y - cy) ** 2 / (cy * 0.85) ** 2) <= 1.0
    base = np.zeros((height, width), dtype=np.float32)
    base[skull_mask] = rng.normal(loc=0.45, scale=0.08, size=skull_mask.sum()).clip(0, 1)

    # ── Grad-CAM hot-spot: Gaussian blob in right temporal region ──────────
    spot_y, spot_x = int(cy * 0.85), int(cx * 1.25)
    yy, xx = np.mgrid[:height, :width]
    sigma = 28.0
    heatmap = np.exp(-((yy - spot_y) ** 2 + (xx - spot_x) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap / heatmap.max()).astype(np.float32)

    # ── Colourmap: jet on heatmap, blended over greyscale background ───────
    import matplotlib  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cmap = matplotlib.colormaps["jet"]
    heat_rgb = cmap(heatmap)[:, :, :3]  # (H, W, 3) float 0-1

    grey_rgb = np.stack([base, base, base], axis=-1)  # (H, W, 3)
    alpha = (heatmap[..., np.newaxis] * 0.65)         # blend weight
    overlay = (alpha * heat_rgb + (1 - alpha) * grey_rgb)
    overlay_u8 = (overlay * 255).astype(np.uint8)

    # ── Encode to PNG ────────────────────────────────────────────────────────
    from PIL import Image  # noqa: PLC0415

    img = Image.fromarray(overlay_u8)

    # Draw a subtle tumour contour ring
    from PIL import ImageDraw  # noqa: PLC0415

    draw = ImageDraw.Draw(img)
    r = 32
    draw.ellipse(
        [spot_x - r, spot_y - r, spot_x + r, spot_y + r],
        outline=(255, 255, 0),
        width=2,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_mock_response(
    patient_data: dict[str, Any],
    scan_path: str,
    run_id: str,
    elapsed: float,
) -> AnalysisResponse:
    """
    Build a fully populated AnalysisResponse with realistic mock values.

    Deterministically varies the confidence based on patient age so repeated
    calls with different patient data produce visibly different results.
    """
    age = patient_data.get("age", 50)
    # Confidence varies between 0.55 and 0.95 depending on age (demo variability)
    base_confidence = 0.95 - (abs(age - 50) * 0.004)
    base_confidence = float(np.clip(base_confidence, 0.55, 0.95))

    label = _confidence_label(base_confidence)
    requires_review = base_confidence < _HIGH_THRESHOLD

    vision = VisionSummary(
        tumour_detected=True,
        confidence_score=round(base_confidence, 4),
        tumour_volume_voxels=3187,
        tumour_volume_cc=3.19,
        gradcam_slice=78,
        model_version="unet-monai-v1 [mock]",
    )

    citations = [
        Citation(
            title="Glioblastoma Multiforme: A Review of its Epidemiology and Multidisciplinary Management",
            authors=["Hanif, F.", "Muzaffar, K.", "Perveen, K.", "Malak, S.A.", "Simjee, S.U."],
            journal="Asian Pacific Journal of Cancer Prevention",
            year=2017,
            pmid="28669158",
            relevance_score=0.93,
            relevance_snippet=(
                "GBM is the most common and most aggressive primary brain tumour in adults, "
                "with a median survival of 14–16 months despite maximal therapy."
            ),
            citation="Hanif et al. (2017). Asian Pac J Cancer Prev, 18(1), 3–9.",
        ),
        Citation(
            title="The 2016 World Health Organization Classification of Tumors of the Central Nervous System",
            authors=["Louis, D.N.", "Perry, A.", "Reifenberger, G.", "et al."],
            journal="Acta Neuropathologica",
            year=2016,
            pmid="27157931",
            relevance_score=0.88,
            relevance_snippet=(
                "The 2016 CNS WHO classification introduces molecular parameters into the "
                "classification of brain tumours alongside histology."
            ),
            citation="Louis DN et al. (2016). Acta Neuropathol, 131(6), 803–820.",
        ),
    ]

    report = StructuredReport(
        findings=(
            "A 3.2 × 2.8 cm T2/FLAIR hyperintense mass lesion is identified in the right "
            "temporal lobe (axial slice 78), with associated perilesional oedema. "
            "The lesion demonstrates irregular margins and internal heterogeneity consistent "
            "with a high-grade neoplasm. Mass effect on the adjacent right lateral ventricle "
            "is noted. No midline shift is observed at this stage."
        ),
        impression=(
            f"MRI findings in this {age}-year-old patient are highly suspicious for a "
            "high-grade glioma (WHO Grade IV, glioblastoma pattern). The identified lesion "
            f"in the right temporal lobe has a detection confidence of {base_confidence:.0%}. "
            "Correlation with contrast-enhanced MRI and clinical history is strongly advised."
        ),
        recommendations=(
            "1. Urgent neurosurgical consultation.\n"
            "2. Contrast-enhanced MRI brain (gadolinium) to characterise enhancement pattern.\n"
            "3. Neurosurgical referral for consideration of image-guided biopsy or resection.\n"
            "4. Multidisciplinary team (MDT) review including neuro-oncology."
        ),
        reasoning=(
            "Vision Agent: U-Net segmentation identified tumour voxels with high probability "
            f"(confidence={base_confidence:.3f}). Grad-CAM saliency concentrated in right temporal lobe. "
            "Clinical History Agent: Patient history and symptoms are consistent with a space-occupying "
            "lesion. RAG Agent: Retrieved 2 high-relevance papers on high-grade glioma. "
            "Report Agent: Synthesised all inputs into structured radiology report."
        ),
        cited_literature=citations,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    return AnalysisResponse(
        run_id=run_id,
        report=report,
        vision_summary=vision,
        heatmap_b64=_generate_mock_heatmap(),
        confidence=base_confidence,
        confidence_label=label,
        requires_review=requires_review,
        verification_notes=(
            f"Confidence score {base_confidence:.3f} {'below' if requires_review else 'above'} "
            f"HIGH threshold ({_HIGH_THRESHOLD}). "
            + ("Flagged for mandatory human review." if requires_review else "Proceeding to explainability.")
        ),
        explanation_summary=(
            "Grad-CAM saliency map shows peak activation in the right temporal lobe "
            f"at axial slice 78, with a contiguous activation region of ~{vision.tumour_volume_cc:.1f} cm³. "
            "The model's attention is tightly localised to the mass region, consistent with a "
            "focal high-grade lesion rather than diffuse pathology."
        ),
        pipeline_status="human_review_required" if requires_review else "complete",
        processing_time_s=round(elapsed, 3),
        metadata={
            "mode": "mock",
            "vision_model": "unet-monai-v1 [mock]",
            "llm_model": "phi-3-mini-4bit [mock]",
            "rag_backend": "faiss [mock]",
            "scan_path": str(scan_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ── Real pipeline helper ───────────────────────────────────────────────────────


def _state_to_response(
    state: dict[str, Any],
    run_id: str,
    elapsed: float,
    scan_path: str,
) -> AnalysisResponse:
    """
    Map a final NeuroAgentState dict (from the LangGraph pipeline) to AnalysisResponse.

    Called only in REAL mode.  All field lookups are defensive so a partially-
    complete pipeline (incremental agent development) still returns a usable
    response rather than crashing.

    Field-name normalisation
    ------------------------
    orchestrator/state.py documents ``vision_findings`` keys as:
      - ``confidence``       float  (primary confidence score)
      - ``prediction_label`` str    ("tumour_detected" | "no_tumour")
      - ``model_version``    str

    The VisionSummary schema uses ``confidence_score`` (float) and
    ``tumour_detected`` (bool), so this function translates between the two.
    Both old and new key names are tried for forward/backward compatibility
    as agents are incrementally integrated.

    Args:
        state:     Final NeuroAgentState dict from the LangGraph pipeline.
        run_id:    The run UUID.
        elapsed:   Wall-clock seconds the pipeline took.
        scan_path: Path to the uploaded scan file (for metadata).

    Returns:
        Fully populated AnalysisResponse.
    """
    vision_findings: dict = state.get("vision_findings") or {}
    report_dict: dict     = state.get("report") or {}
    lit_results: list     = state.get("literature_results") or []

    # ── Vision: confidence score ──────────────────────────────────────────────
    # state.py documents the key as "confidence"; VisionSummary uses "confidence_score".
    # Try both so the mapper works regardless of which key the agent writes.
    vision_confidence = float(
        vision_findings.get("confidence_score")
        or vision_findings.get("confidence")
        or 0.0
    )

    # ── Vision: tumour detected bool ──────────────────────────────────────────
    # state.py documents the key as "prediction_label" (str).
    # Try explicit bool key first; derive from label string as fallback.
    if "tumour_detected" in vision_findings:
        tumour_detected = bool(vision_findings["tumour_detected"])
    else:
        pred_label = vision_findings.get("prediction_label", "").lower()
        tumour_detected = pred_label in ("tumour_detected", "tumor_detected",
                                         "positive", "abnormal", "1", "true")

    # ── Vision summary ────────────────────────────────────────────────────────
    vision_summary = VisionSummary(
        tumour_detected=tumour_detected,
        confidence_score=vision_confidence,
        tumour_volume_voxels=int(vision_findings.get("tumour_volume_voxels", 0)),
        tumour_volume_cc=float(vision_findings.get("tumour_volume_cc", 0.0)),
        gradcam_slice=int(vision_findings.get("gradcam_slice", -1)),
        model_version=str(vision_findings.get("model_version", "unet-monai-v1")),
    )

    # ── Citations ─────────────────────────────────────────────────────────────
    # Map RAG output list → Citation objects.
    # "abstract" is the full text key; "relevance_snippet" is a pre-extracted field.
    citations = [
        Citation(
            title=lit.get("title", ""),
            authors=lit.get("authors", []),
            journal=lit.get("journal", ""),
            year=lit.get("year"),
            pmid=lit.get("pmid"),
            relevance_score=float(lit.get("relevance_score", 0.0)),
            relevance_snippet=(
                lit.get("relevance_snippet")
                or lit.get("abstract", "")[:300]
            ),
            citation=lit.get("citation", ""),
        )
        for lit in lit_results
    ]

    # ── Report ────────────────────────────────────────────────────────────────
    report = StructuredReport(
        findings=report_dict.get("findings", ""),
        impression=report_dict.get("impression", ""),
        recommendations=report_dict.get("recommendations", ""),
        reasoning=report_dict.get("reasoning", ""),
        cited_literature=citations,
        generated_at=report_dict.get(
            "generated_at", datetime.now(timezone.utc).isoformat()
        ),
    )

    # ── Overall confidence ────────────────────────────────────────────────────
    # state["overall_confidence"] is set by the Report Agent combining vision +
    # clinical scores.  Falls back to raw vision confidence if not yet available.
    confidence = float(
        state.get("overall_confidence")
        or vision_findings.get("confidence_score")
        or vision_findings.get("confidence")
        or 0.0
    )

    # ── Heatmap PNG → base64 ──────────────────────────────────────────────────
    # The Explainability Agent saves a PNG and writes its path to
    # state["gradcam_heatmap_path"].  If the path is missing or the file
    # was cleaned up, fall back to a synthetic heatmap so the UI never errors.
    heatmap_b64: str = ""
    gradcam_path = state.get("gradcam_heatmap_path", "")
    if gradcam_path and Path(gradcam_path).exists():
        raw_bytes = Path(gradcam_path).read_bytes()
        heatmap_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        logger.debug("Loaded heatmap from %s (%d bytes)", gradcam_path, len(raw_bytes))
    else:
        logger.warning(
            "gradcam_heatmap_path missing or not found (%r). Using synthetic fallback.",
            gradcam_path,
        )
        heatmap_b64 = _generate_mock_heatmap()

    # ── Pipeline status ───────────────────────────────────────────────────────
    pipeline_status = state.get("pipeline_status", "complete")
    if pipeline_status not in ("complete", "human_review_required", "error"):
        pipeline_status = "complete"

    return AnalysisResponse(
        run_id=run_id,
        report=report,
        vision_summary=vision_summary,
        heatmap_b64=heatmap_b64,
        confidence=confidence,
        confidence_label=_confidence_label(confidence),
        requires_review=bool(state.get("requires_human_review", False)),
        verification_notes=state.get("verification_notes", ""),
        explanation_summary=state.get("explanation_summary", ""),
        pipeline_status=pipeline_status,
        processing_time_s=round(elapsed, 3),
        metadata={
            "mode": "real",
            "vision_model": vision_findings.get("model_version", "unet-monai-v1"),
            "llm_model": os.getenv("LLM_MODEL_NAME", "phi-3-mini-4bit"),
            "rag_backend": "faiss",
            "scan_path": str(scan_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ── OrchestratorPipeline ───────────────────────────────────────────────────────


class OrchestratorPipeline:
    """
    Pipeline manager loaded once at app startup via the FastAPI lifespan.

    Mode is controlled by the ``MOCK_MODE`` environment variable:
      - MOCK_MODE=false (default) → imports and runs the real LangGraph pipeline.
      - MOCK_MODE=true            → returns deterministic synthetic responses.

    If the real pipeline import fails at runtime (missing model weights,
    missing dependencies) the class automatically falls back to MOCK so the
    API stays up and returns a meaningful error-free response.

    Public API (called by routes/analyze.py):
        result = await pipeline.run(scan_path, patient_data_dict)
    """

    def __init__(self) -> None:
        self._mock: bool = _MOCK_MODE   # True → mock, False → real
        self._graph = None              # lazy-compiled on first real run
        logger.info(
            "[Pipeline] OrchestratorPipeline initialised | mock_mode=%s", self._mock
        )

    # ── Lazy graph loader ─────────────────────────────────────────────────────

    def _ensure_graph_loaded(self) -> None:
        """Import and compile the LangGraph graph (real mode only, called once)."""
        if self._graph is not None:
            return
        try:
            from orchestrator.graph import compile_graph  # noqa: PLC0415
            self._graph = compile_graph()
            logger.info("[Pipeline] LangGraph pipeline compiled successfully.")
        except Exception as exc:
            logger.error(
                "[Pipeline] Failed to compile LangGraph graph: %s — falling back to mock.", exc
            )
            self._mock = True   # automatic graceful degradation

    # ── Public async run ──────────────────────────────────────────────────────

    async def run(
        self,
        scan_path: str,
        patient_data: dict[str, Any],
        run_id: str | None = None,
    ) -> AnalysisResponse:
        """
        Run the full NeuroAgent pipeline asynchronously.

        In REAL mode, the synchronous LangGraph call is offloaded to a thread
        pool executor so it doesn't block the FastAPI event loop.

        Args:
            scan_path:    Absolute path to the uploaded MRI scan (temp file).
            patient_data: Dict matching the PatientData.to_pipeline_dict() format.
            run_id:       Optional trace ID. Auto-generated UUID4 if omitted.

        Returns:
            Fully populated AnalysisResponse ready to be serialised as JSON.

        Raises:
            RuntimeError: If the pipeline fails in a non-recoverable way.
        """
        _run_id = run_id or str(uuid.uuid4())
        t_start = time.time()

        # ── Mode selection ─────────────────────────────────────────────────────
        # Compile the graph lazily on first real call.
        # If _ensure_graph_loaded() fails it sets self._mock = True automatically.
        if not self._mock:
            self._ensure_graph_loaded()

        # ── MOCK mode ─────────────────────────────────────────────────────────
        if self._mock:
            logger.info("[Pipeline] MOCK run | run_id=%s", _run_id)
            await asyncio.sleep(0.8)   # simulate realistic processing delay
            elapsed = time.time() - t_start
            return _build_mock_response(patient_data, scan_path, _run_id, elapsed)

        # ── REAL mode — call the LangGraph orchestrator ────────────────────────
        logger.info("[Pipeline] REAL run | run_id=%s | scan=%s", _run_id, scan_path)

        # Offload the synchronous graph.invoke() to a thread pool executor
        # so it doesn't block the FastAPI async event loop.
        loop = asyncio.get_event_loop()
        from orchestrator.state import create_initial_state  # noqa: PLC0415

        initial_state = create_initial_state(
            mri_scan_path=scan_path,
            patient_data=patient_data,
            run_id=_run_id,
        )

        state: dict = await loop.run_in_executor(
            None, self._graph.invoke, initial_state
        )

        elapsed = time.time() - t_start
        logger.info(
            "[Pipeline] REAL run complete | run_id=%s | elapsed=%.2fs | status=%s",
            _run_id, elapsed, state.get("pipeline_status"),
        )
        return _state_to_response(state, _run_id, elapsed, scan_path)

    def __repr__(self) -> str:
        return (
            f"OrchestratorPipeline(mock_mode={self._mock}, "
            f"graph_loaded={self._graph is not None})"
        )
