"""
frontend/app.py
----------------
NeuroAgent Streamlit dashboard — main entry point.

Run with:
    streamlit run frontend/app.py

Environment variables (read from .env or system):
    BACKEND_URL   URL of the FastAPI backend (default: http://localhost:8000)
    LOG_LEVEL     DEBUG|INFO|WARNING  (default: INFO)

Page layout:
    - Sidebar:   Logo + MRI upload + patient form + Run button
    - Main area: Landing page (before run) OR full results dashboard (after run)

State management (st.session_state):
    result       dict     Full API response
    scan_bytes   bytes    Raw uploaded file bytes (for original-scan display)
    scan_name    str      Filename of the uploaded scan
    last_file_id str      Fingerprint of the last uploaded file (reset on new upload)
    error_msg    str|None Error message from the last failed API call
"""

from __future__ import annotations

# ── sys.path fix (MUST be before all project imports) ─────────────────────────
# Streamlit adds the *script's directory* (frontend/) to sys.path, which means
# `from frontend.X import Y` fails because Python looks for frontend/frontend/X.
# We insert the project root so all `from frontend.X` and `from backend.X`
# imports resolve correctly regardless of launch directory.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

# ── Load env before anything else ─────────────────────────────────────────────
load_dotenv()

# ── Page config (MUST be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="NeuroAgent — Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Nandan-Bomale/neuro-agent",
        "Report a Bug": "https://github.com/Nandan-Bomale/neuro-agent/issues",
        "About": "NeuroAgent · Multi-Agent AI System for Brain MRI Diagnosis Support · Academic Project",
    },
)

# ── Import frontend modules (after set_page_config) ───────────────────────────
# sys.path now includes project root, so `from frontend.X` resolves correctly.
from frontend.styles import inject_css  # noqa: E402
from frontend.components.upload_panel import render_upload_panel  # noqa: E402
from frontend.components.report_viewer import render_results  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("neuroagent.frontend")

# ── Config ─────────────────────────────────────────────────────────────────────
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
ANALYZE_URL: str = f"{BACKEND_URL}/api/analyze"
HEALTH_URL:  str = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT: int = 180  # seconds


# ── API helpers ────────────────────────────────────────────────────────────────


def _check_backend_health() -> bool:
    """Return True if the backend health endpoint responds successfully."""
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _call_analyze_api(scan_file: Any, patient_data: dict) -> dict:
    """
    POST the scan + patient JSON to the /api/analyze endpoint.

    Args:
        scan_file:    Streamlit UploadedFile object.
        patient_data: Dict matching PatientData schema.

    Returns:
        Full API response dict on success.

    Raises:
        requests.HTTPError: On 4xx/5xx responses.
        requests.ConnectionError: If backend is not reachable.
        requests.Timeout: If the pipeline takes too long.
        ValueError: If the response is not valid JSON.
    """
    patient_json = json.dumps(patient_data)
    scan_bytes   = scan_file.getvalue()
    mime_type    = scan_file.type or "application/octet-stream"

    response = requests.post(
        ANALYZE_URL,
        files={"scan_file": (scan_file.name, scan_bytes, mime_type)},
        data={"patient_json": patient_json},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _file_fingerprint(f: Any) -> str:
    """Create a stable fingerprint for an uploaded file to detect changes."""
    return f"{f.name}_{f.size}"


# ── Landing page ───────────────────────────────────────────────────────────────


def _render_landing() -> None:
    """Render the hero + features landing page shown before the first analysis."""
    st.markdown(
        """
        <div class="landing-hero">
            <div class="landing-hero-title">Upload a scan to get started</div>
            <div class="landing-hero-sub">
                NeuroAgent analyses brain MRI scans using a seven-agent AI pipeline —
                detecting abnormalities, reasoning over clinical context, retrieving
                evidence from medical literature, and generating a structured
                radiology report in seconds.
            </div>
        </div>

        <div class="feature-grid">
            <div class="feature-tile">
                <span class="feature-icon">👁️</span>
                <div class="feature-title">Vision Agent</div>
                <div class="feature-desc">U-Net segmentation detects and outlines tumour regions with pixel-level precision.</div>
            </div>
            <div class="feature-tile">
                <span class="feature-icon">🧬</span>
                <div class="feature-title">Clinical Reasoning</div>
                <div class="feature-desc">LLM-powered agent cross-references patient history and symptoms with the scan findings.</div>
            </div>
            <div class="feature-tile">
                <span class="feature-icon">📚</span>
                <div class="feature-title">RAG Literature</div>
                <div class="feature-desc">Retrieves and cites relevant PubMed papers using semantic search over medical literature.</div>
            </div>
            <div class="feature-tile">
                <span class="feature-icon">📝</span>
                <div class="feature-title">Report Generation</div>
                <div class="feature-desc">Fine-tuned LLM writes a structured, evidence-cited radiology report in clinical language.</div>
            </div>
            <div class="feature-tile">
                <span class="feature-icon">🛡️</span>
                <div class="feature-title">Verification Agent</div>
                <div class="feature-desc">Checks pipeline confidence and flags low-certainty cases for mandatory human review.</div>
            </div>
            <div class="feature-tile">
                <span class="feature-icon">🎯</span>
                <div class="feature-title">Grad-CAM Heatmap</div>
                <div class="feature-desc">Shows exactly which region of the scan drove the model's conclusion — full explainability.</div>
            </div>
        </div>

        <div class="pipeline-flow">
            <span style="font-size:0.65rem;color:#475569;font-weight:700;
                         text-transform:uppercase;letter-spacing:0.12em;margin-right:0.4rem;">Pipeline</span>
            <span class="pipeline-node">👁️ Vision</span>
            <span class="pipeline-arrow">+</span>
            <span class="pipeline-node">🧬 Clinical</span>
            <span class="pipeline-arrow">+</span>
            <span class="pipeline-node">📚 RAG</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">📝 Report</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">🛡️ Verify</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">🎯 Explain</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Error display ──────────────────────────────────────────────────────────────


def _render_error(msg: str) -> None:
    """Render a styled error card in the main area."""
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,rgba(127,29,29,0.5),rgba(153,27,27,0.3));
                    border:1px solid rgba(220,38,38,0.4);border-left:4px solid #dc2626;
                    border-radius:14px;padding:1.5rem 1.8rem;margin:1.5rem 0;">
            <div style="font-weight:700;color:#f87171;font-size:0.95rem;margin-bottom:0.5rem;">
                ❌ Analysis Failed
            </div>
            <div style="color:#fca5a5;font-size:0.85rem;line-height:1.6;font-family:monospace;">
                {msg}
            </div>
            <div style="color:#7f1d1d;font-size:0.75rem;margin-top:0.8rem;">
                Make sure the backend is running: <code style="color:#991b1b;">uvicorn backend.main:app --reload</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main header ────────────────────────────────────────────────────────────────


def _render_header() -> None:
    """Render the NeuroAgent main area header."""
    # Backend status indicator (non-blocking)
    backend_ok = _check_backend_health()
    status_dot   = "🟢" if backend_ok else "🔴"
    status_label = "Backend online" if backend_ok else "Backend offline"
    status_mode  = " · mock mode" if backend_ok else ""

    try:
        if backend_ok:
            r = requests.get(HEALTH_URL, timeout=3)
            mode = r.json().get("pipeline_mode", "mock")
            status_mode = f" · {mode} mode"
    except Exception:
        pass

    st.markdown(
        f"""
        <div class="main-header">
            <span class="main-header-icon">🧠</span>
            <h1 class="main-header-title">NeuroAgent</h1>
            <div class="main-header-sub">Multi-Agent AI System for Brain MRI Diagnosis Support</div>
            <div style="margin-top:0.75rem;display:inline-flex;align-items:center;gap:0.4rem;
                        background:rgba(12,20,40,0.6);border:1px solid rgba(99,102,241,0.15);
                        border-radius:20px;padding:0.25rem 0.9rem;font-size:0.68rem;color:#64748b;">
                {status_dot} {status_label}{status_mode}
                &nbsp;·&nbsp; {BACKEND_URL}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main app ───────────────────────────────────────────────────────────────────


def main() -> None:
    """NeuroAgent Streamlit application entry point."""

    # ── Inject design system ──────────────────────────────────────────────────
    inject_css()

    # ── Full-width header ─────────────────────────────────────────────────────
    _render_header()

    # ── Two-column layout: controls left | results right ─────────────────────
    col_controls, col_results = st.columns([1, 2.2], gap="large")

    with col_controls:
        scan_file, patient_data, run_clicked = render_upload_panel()

    # ── Track file changes — reset results on new upload ──────────────────────
    if scan_file is not None:
        fp = _file_fingerprint(scan_file)
        if st.session_state.get("last_file_id") != fp:
            for key in ("result", "scan_bytes", "scan_name", "error_msg"):
                st.session_state.pop(key, None)
            st.session_state["last_file_id"] = fp

    # ── Handle Run button click ───────────────────────────────────────────────
    if run_clicked and scan_file is not None:
        st.session_state.pop("error_msg", None)
        with col_results:
            with st.spinner("🧠  Running NeuroAgent pipeline…"):
                t0 = time.time()
                try:
                    result = _call_analyze_api(scan_file, patient_data)
                    st.session_state["result"]     = result
                    st.session_state["scan_bytes"] = scan_file.getvalue()
                    st.session_state["scan_name"]  = scan_file.name
                    elapsed = time.time() - t0
                    logger.info(
                        "Analysis complete | conf=%.2f | label=%s | time=%.1fs",
                        result.get("confidence", 0),
                        result.get("confidence_label"),
                        elapsed,
                    )
                    st.rerun()

                except requests.ConnectionError:
                    st.session_state["error_msg"] = (
                        f"Cannot connect to the backend at {BACKEND_URL}.\n"
                        "Please start: uvicorn backend.main:app --reload"
                    )
                except requests.Timeout:
                    st.session_state["error_msg"] = (
                        f"Request timed out after {REQUEST_TIMEOUT}s. Try again."
                    )
                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.session_state["error_msg"] = (
                        f"Backend error ({exc.response.status_code}): {detail}"
                    )
                except Exception as exc:
                    st.session_state["error_msg"] = f"Unexpected error: {exc}"

    # ── Right column: results or landing page ─────────────────────────────────
    with col_results:
        if "error_msg" in st.session_state and st.session_state["error_msg"]:
            _render_error(st.session_state["error_msg"])

        if "result" in st.session_state:
            render_results(
                api_response=st.session_state["result"],
                scan_bytes=st.session_state["scan_bytes"],
                scan_filename=st.session_state["scan_name"],
            )
        else:
            _render_landing()



if __name__ == "__main__":
    main()
