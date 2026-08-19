"""
frontend/components/live_dashboard.py
-------------------------------------
Handles the Live Step-by-Step Execution Dashboard using Server-Sent Events.
"""

from __future__ import annotations
import json
import requests
import streamlit as st
from typing import Any

def render_live_dashboard(scan_file: Any, patient_data: dict, analyze_url: str) -> None:
    """Streams events from the backend and dynamically updates the UI."""
    
    col_timeline, col_proof = st.columns([1, 1.5], gap="large")
    
    with col_timeline:
        st.markdown("<h3 style='color:#00c8ff;'>Execution Timeline</h3>", unsafe_allow_html=True)
        timeline_container = st.container()
        
    with col_proof:
        st.markdown("<h3 style='color:#7b2ff7;'>Visual Evidence</h3>", unsafe_allow_html=True)
        proof_container = st.container()
        
    patient_json = json.dumps(patient_data)
    scan_bytes = scan_file.getvalue()
    mime_type = scan_file.type or "application/octet-stream"

    try:
        response = requests.post(
            analyze_url,
            files={"scan_file": (scan_file.name, scan_bytes, mime_type)},
            data={"patient_json": patient_json},
            stream=True,
            timeout=180,
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    payload_str = decoded_line[6:]
                    event = json.loads(payload_str)
                    
                    _render_event(event, timeline_container, proof_container)
                    
    except Exception as exc:
        with timeline_container:
            st.error(f"Streaming failed: {exc}")


def _render_event(event: dict, timeline_container, proof_container) -> None:
    agent_name = event.get("agent_name", "Unknown")
    input_sum = event.get("input_summary", "")
    output_data = event.get("output_data", {})
    img_b64 = event.get("image_b64")
    
    icon_map = {
        "vision": "👁️",
        "tumor_classification": "🧫",
        "radiogenomics": "🧬",
        "surgical": "🔪",
        "prognostic": "⏳",
        "clinical_trials": "📋",
        "neuro_oncologist": "👨‍⚕️",
        "explainability": "🎯",
        "report": "📝",
        "error": "❌"
    }
    
    icon = icon_map.get(agent_name, "⚙️")
    name_display = agent_name.replace("_", " ").title()
    
    # ── 1. Timeline Update ──────────────────────────────────────────────
    with timeline_container:
        with st.expander(f"{icon} {name_display} Agent Finished", expanded=True):
            st.markdown(f"**Input:** {input_sum}")
            st.json(output_data)
            
    # ── 2. Proof Panel Update ───────────────────────────────────────────
    if img_b64:
        with proof_container:
            st.markdown(f"#### {icon} {name_display} Visual Output")
            st.image(f"data:image/png;base64,{img_b64}", width=None)
            
    if event.get("is_final") or agent_name == "neuro_oncologist":
        with proof_container:
            st.success("Pipeline Execution Complete!")
