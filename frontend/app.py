"""
frontend/app.py
----------------
NeuroAgent Streamlit dashboard — main entry point.
Redesigned for a Futuristic Medical Dashboard aesthetic.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

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

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroAgent — Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.styles import inject_css  # noqa: E402
from frontend.components.upload_panel import render_upload_panel  # noqa: E402
from frontend.components.report_viewer import render_results  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("neuroagent.frontend")

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
ANALYZE_URL: str = f"{BACKEND_URL}/api/analyze"
HEALTH_URL:  str = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT: int = 180


def _check_backend_health() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _call_analyze_api(scan_file: Any, patient_data: dict) -> dict:
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
    return f"{f.name}_{f.size}"


def _render_header() -> None:
    """Render the sleek top navbar."""
    backend_ok = _check_backend_health()
    status_color = "#00ff88" if backend_ok else "#ff4444"
    status_text = "Backend Online" if backend_ok else "Backend Offline"
    
    html = f"""
    <div class="top-header">
        <div class="header-logo">
            <span>🧠</span> NEUROAGENT
        </div>
        <div class="header-search">
            <span>🔍</span> Search Patient ID, Diagnosis...
        </div>
        <div class="header-profile">
            <span style="color:#8892b0;font-size:0.75rem;">{status_text}</span>
            <span class="online-dot" style="background-color:{status_color};box-shadow:0 0 8px {status_color};"></span>
            <img src="https://ui-avatars.com/api/?name=Evelyn+Reed&background=7b2ff7&color=fff&rounded=true&size=36" style="margin-left:10px;"/>
            <span>Dr. Evelyn Reed | <span style="color:#00ff88;">Online</span> 🔔</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_sidebar() -> None:
    """Render the sleek vertical navigation in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;gap:1.5rem;padding-top:2rem;">
                <div style="background:rgba(0,200,255,0.15); border:1px solid rgba(0,200,255,0.4); border-radius:12px; padding:0.8rem 1rem; color:#00c8ff; font-weight:600; display:flex; align-items:center; gap:0.8rem; cursor:pointer; box-shadow:0 0 15px rgba(0,200,255,0.1);">
                    <span style="font-size:1.2rem;">📊</span> Dashboard
                </div>
                <div style="padding:0.8rem 1rem; color:#8892b0; font-weight:500; display:flex; align-items:center; gap:0.8rem; cursor:pointer;">
                    <span style="font-size:1.2rem;">👥</span> Patients
                </div>
                <div style="padding:0.8rem 1rem; color:#8892b0; font-weight:500; display:flex; align-items:center; gap:0.8rem; cursor:pointer;">
                    <span style="font-size:1.2rem;">📄</span> Reports
                </div>
                <div style="padding:0.8rem 1rem; color:#8892b0; font-weight:500; display:flex; align-items:center; gap:0.8rem; cursor:pointer;">
                    <span style="font-size:1.2rem;">⚙️</span> Settings
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )


def _render_error(msg: str) -> None:
    st.error(f"Analysis Failed: {msg}")


def main() -> None:
    inject_css()
    _render_header()
    _render_sidebar()

    # Determine if we have a result
    has_result = "result" in st.session_state

    if not has_result:
        # Show upload panel in a nice centered card container
        st.markdown('<div style="max-width:800px;margin:0 auto;">', unsafe_allow_html=True)
        scan_file, patient_data, run_clicked = render_upload_panel()
        st.markdown('</div>', unsafe_allow_html=True)
        
        if run_clicked and scan_file is not None:
            st.session_state.pop("error_msg", None)
            with st.spinner("🧠 Initializing Agents & Running Analysis..."):
                t0 = time.time()
                try:
                    result = _call_analyze_api(scan_file, patient_data)
                    st.session_state["result"] = result
                    st.session_state["scan_bytes"] = scan_file.getvalue()
                    st.session_state["scan_name"] = scan_file.name
                    st.session_state["patient_data"] = patient_data
                    st.rerun()
                except Exception as exc:
                    _render_error(str(exc))
    else:
        # Show the futuristic results dashboard
        render_results(
            api_response=st.session_state["result"],
            scan_bytes=st.session_state["scan_bytes"],
            scan_filename=st.session_state["scan_name"],
        )

if __name__ == "__main__":
    main()
