"""
frontend/app.py
----------------
NeuroAgent Streamlit dashboard — main entry point.
Refactored for a premium top-navigation layout.
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

st.set_page_config(
    page_title="NeuroAgent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed", # Hide standard sidebar completely
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

def _render_top_nav() -> None:
    """Render the sleek horizontal top navigation bar with Lucide SVGs."""
    backend_ok = _check_backend_health()
    status_color = "#00ff88" if backend_ok else "#ff4444"
    status_text = "Online" if backend_ok else "Offline"
    
    html = f"""
    <div class="top-nav-bar">
        <div class="nav-left">
            <div class="brand-logo">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00c8ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/></svg>
                NEUROAGENT
            </div>
            <div class="main-menu">
                <div class="menu-item active">Dashboard</div>
                <div class="menu-item">Patients</div>
                <div class="menu-item">Reports</div>
                <div class="menu-item">Settings</div>
            </div>
        </div>
        
        <div class="nav-right">
            <div class="search-box">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                Search Patient ID, Diagnosis...
            </div>
            <div class="profile-section">
                <div class="status-indicator">
                    <span class="dot" style="background-color:{status_color};box-shadow:0 0 8px {status_color};"></span>
                    API {status_text}
                </div>
                <div style="width:1px;height:20px;background:rgba(255,255,255,0.1);margin:0 0.5rem;"></div>
                <img src="https://ui-avatars.com/api/?name=Dr.+Evelyn&background=7b2ff7&color=fff&rounded=true&size=32" />
                Dr. Evelyn Reed
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-left:8px;color:#00c8ff;cursor:pointer;"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_error(msg: str) -> None:
    st.error(f"Analysis Failed: {msg}")


def main() -> None:
    inject_css()
    _render_top_nav()

    has_result = "result" in st.session_state

    if not has_result:
        st.markdown('<div style="max-width:800px;margin:0 auto;">', unsafe_allow_html=True)
        scan_file, patient_data, run_clicked = render_upload_panel()
        st.markdown('</div>', unsafe_allow_html=True)
        
        if run_clicked and scan_file is not None:
            st.session_state.pop("error_msg", None)
            with st.spinner("Initializing Agents & Running Analysis..."):
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
        render_results(
            api_response=st.session_state["result"],
            scan_bytes=st.session_state["scan_bytes"],
            scan_filename=st.session_state["scan_name"],
        )

if __name__ == "__main__":
    main()
