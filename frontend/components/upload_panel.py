"""
frontend/components/upload_panel.py
-------------------------------------
Upload panel presented as a modern glassmorphism card with SVGs.
"""

from __future__ import annotations
from typing import Any
import streamlit as st

def render_upload_panel() -> tuple[Any, dict, bool]:
    st.markdown('<div class="agent-card" style="padding: 3rem; max-width: 700px; margin: 4rem auto; text-align:center;">', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00c8ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="filter:drop-shadow(0 0 15px rgba(0,200,255,0.6)); margin-bottom:1rem;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/></svg>
            <h2 class="outfit-font" style="margin:0; font-weight:700; font-size:2rem; color:#fff;">Start New Analysis</h2>
            <div style="color:#94a3b8; font-size:0.95rem; margin-top:0.5rem; max-width:450px; margin-left:auto; margin-right:auto;">Upload a brain MRI scan and patient details for multi-agent evaluation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="color:#00c8ff; font-weight:600; text-transform:uppercase; font-size:0.75rem; letter-spacing:0.1em; margin-bottom:0.5rem; text-align:left;">1. MRI Scan</div>', unsafe_allow_html=True)
    scan_file = st.file_uploader("Upload Scan", type=["nii", "gz", "png", "jpg", "jpeg"], label_visibility="collapsed")

    st.markdown('<div style="color:#00c8ff; font-weight:600; text-transform:uppercase; font-size:0.75rem; letter-spacing:0.1em; margin-top:1.5rem; margin-bottom:0.5rem; text-align:left;">2. Patient Details</div>', unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a: age = st.number_input("Age", min_value=0, value=45)
    with col_b: sex = st.selectbox("Sex", ["M", "F", "Other"])
    with col_c: modality = st.selectbox("Modality", ["FLAIR", "T1", "T1ce", "T2", "DWI"])

    symptoms_raw = st.text_input("Symptoms (comma-separated)", placeholder="headache, blurred vision...")
    history_raw = st.text_input("Medical History", placeholder="hypertension...")
    
    st.markdown('<br>', unsafe_allow_html=True)
    run_clicked = st.button("Execute Pipeline", disabled=scan_file is None, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    def _split(raw: str) -> list[str]:
        return [s.strip() for s in raw.split(",") if s.strip()]

    patient_data: dict = {
        "age": int(age),
        "sex": sex,
        "symptoms": _split(symptoms_raw),
        "medical_history": _split(history_raw),
        "medications": [],
        "referring_notes": "",
        "scan_modality": modality,
    }

    return scan_file, patient_data, run_clicked
