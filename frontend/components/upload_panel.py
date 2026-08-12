"""
frontend/components/upload_panel.py
-------------------------------------
Sidebar upload panel: MRI file uploader + patient details form.

Renders entirely inside `with st.sidebar:` context (called from app.py).

Returns
-------
    scan_file   : UploadFile | None   — the uploaded file object
    patient_data: dict                — validated patient metadata dict
    run_clicked : bool                — True only on the frame the button is clicked
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_upload_panel() -> tuple[Any, dict, bool]:
    """
    Render the sidebar logo, file uploader, patient form, and run button.

    Returns (scan_file, patient_data_dict, run_clicked).
    """
    # ── Sidebar logo ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sb-logo">
            <span class="sb-logo-icon">🧠</span>
            <div class="sb-logo-title">NeuroAgent</div>
            <div class="sb-logo-sub">Brain MRI Diagnosis Support</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Scan upload ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-section">📁 Scan Upload</div>',
        unsafe_allow_html=True,
    )

    scan_file = st.file_uploader(
        "MRI Scan",
        type=["nii", "gz", "png", "jpg", "jpeg"],
        help=(
            "Supported formats: NIfTI (.nii, .nii.gz) for multi-modal scans, "
            "PNG / JPG for single-slice images."
        ),
        label_visibility="collapsed",
        key="scan_uploader",
    )

    if scan_file is not None:
        file_kb = scan_file.size / 1024
        size_str = f"{file_kb:.0f} KB" if file_kb < 1024 else f"{file_kb/1024:.1f} MB"
        st.markdown(
            f'<div style="color:#34d399;font-size:0.72rem;margin-top:0.3rem;'
            f'padding:0.3rem 0.6rem;background:rgba(4,120,87,0.12);'
            f'border-radius:6px;border:1px solid rgba(5,150,105,0.2);">'
            f'✓ &nbsp;<strong>{scan_file.name}</strong> &nbsp;({size_str})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#475569;font-size:0.72rem;margin-top:0.3rem;'
            'text-align:center;">Drag & drop or click to browse</div>',
            unsafe_allow_html=True,
        )

    # ── Patient details ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-section">👤 Patient Details</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input(
            "Age (yrs)",
            min_value=0,
            max_value=130,
            value=45,
            step=1,
            key="patient_age",
        )
    with col_b:
        sex = st.selectbox(
            "Sex",
            options=["M", "F", "Other"],
            key="patient_sex",
        )

    modality = st.selectbox(
        "Scan Modality",
        options=["FLAIR", "T1", "T1ce", "T2", "DWI", "Other"],
        help="Primary MRI modality of the uploaded scan",
        key="patient_modality",
    )

    symptoms_raw = st.text_input(
        "Symptoms",
        placeholder="headache, blurred vision, nausea",
        help="Comma-separated list of reported symptoms",
        key="patient_symptoms",
    )

    history_raw = st.text_area(
        "Medical History",
        placeholder="hypertension, no prior malignancy",
        height=62,
        help="Relevant past conditions, comma-separated",
        key="patient_history",
    )

    medications_raw = st.text_input(
        "Medications",
        placeholder="amlodipine 5mg, aspirin",
        help="Current medications, comma-separated",
        key="patient_meds",
    )

    referring_notes = st.text_area(
        "Referring Notes",
        placeholder="Free-text clinical notes from the referring clinician…",
        height=72,
        key="patient_notes",
    )

    # ── Build patient dict ────────────────────────────────────────────────────
    def _split(raw: str) -> list[str]:
        return [s.strip() for s in raw.split(",") if s.strip()]

    patient_data: dict = {
        "age": int(age),
        "sex": sex,
        "symptoms": _split(symptoms_raw),
        "medical_history": _split(history_raw),
        "medications": _split(medications_raw),
        "referring_notes": referring_notes.strip(),
        "scan_modality": modality,
    }

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button(
        "🔬  Run NeuroAgent Analysis",
        disabled=scan_file is None,
        use_container_width=True,
        key="run_btn",
    )

    if scan_file is None:
        st.markdown(
            '<div style="color:#334155;font-size:0.7rem;text-align:center;'
            'margin-top:0.4rem;">Upload a scan to enable analysis</div>',
            unsafe_allow_html=True,
        )

    # ── Sidebar footer ────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<hr style="border-color:rgba(99,102,241,0.1);">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#334155;font-size:0.65rem;text-align:center;line-height:1.6;">'
        '🔬 <strong style="color:#475569;">NeuroAgent</strong> v0.1.0<br>'
        'Academic project · Not for clinical use<br>'
        '<span style="color:#1e3060;">Nandan Bomale · 2026</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    return scan_file, patient_data, run_clicked
