"""
frontend/components/upload_panel.py
-------------------------------------
Left-column upload panel: MRI file uploader + patient details form.

Renders inside a `st.columns()` context in app.py (NOT in a sidebar).
Returns (scan_file, patient_data_dict, run_clicked).
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_upload_panel() -> tuple[Any, dict, bool]:
    """
    Render the MRI upload widget, patient form, and run button.
    Returns (scan_file, patient_data_dict, run_clicked).
    """

    # ── Panel header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.2rem;">
            <span style="font-size:2.2rem;filter:drop-shadow(0 0 12px rgba(99,102,241,0.9));">🧠</span>
            <div style="font-size:1.2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#34d399);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;letter-spacing:-0.03em;margin-top:0.2rem;">NeuroAgent</div>
            <div style="font-size:0.6rem;color:#475569;letter-spacing:0.1em;
                        text-transform:uppercase;margin-top:0.1rem;">Brain MRI Analysis</div>
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
        help="Supported: NIfTI (.nii, .nii.gz), PNG, JPG/JPEG",
        label_visibility="collapsed",
        key="scan_uploader",
    )

    if scan_file is not None:
        file_kb = scan_file.size / 1024
        size_str = f"{file_kb:.0f} KB" if file_kb < 1024 else f"{file_kb/1024:.1f} MB"
        st.markdown(
            f'<div style="color:#34d399;font-size:0.72rem;margin-top:0.3rem;'
            f'padding:0.3rem 0.7rem;background:rgba(4,120,87,0.12);'
            f'border-radius:7px;border:1px solid rgba(5,150,105,0.25);">'
            f'✓ &nbsp;<strong>{scan_file.name}</strong>&nbsp; ({size_str})</div>',
            unsafe_allow_html=True,
        )

    # ── Patient details ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-section">👤 Patient Details</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age (yrs)", min_value=0, max_value=130,
                              value=45, step=1, key="patient_age")
    with col_b:
        sex = st.selectbox("Sex", ["M", "F", "Other"], key="patient_sex")

    modality = st.selectbox(
        "Scan Modality",
        ["FLAIR", "T1", "T1ce", "T2", "DWI", "Other"],
        help="Primary MRI modality of the uploaded scan",
        key="patient_modality",
    )

    symptoms_raw = st.text_input(
        "Symptoms",
        placeholder="headache, blurred vision, nausea",
        help="Comma-separated",
        key="patient_symptoms",
    )

    history_raw = st.text_area(
        "Medical History",
        placeholder="hypertension, no prior malignancy",
        height=58,
        help="Comma-separated past conditions",
        key="patient_history",
    )

    medications_raw = st.text_input(
        "Medications",
        placeholder="amlodipine 5mg, aspirin",
        help="Comma-separated",
        key="patient_meds",
    )

    referring_notes = st.text_area(
        "Referring Notes",
        placeholder="Free-text clinical notes from the referring clinician…",
        height=68,
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
            'margin-top:0.35rem;">Upload a scan to enable analysis</div>',
            unsafe_allow_html=True,
        )

    # ── Small footer ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="color:#1e3060;font-size:0.62rem;text-align:center;'
        'margin-top:2rem;padding-top:0.8rem;border-top:1px solid rgba(99,102,241,0.1);">'
        'NeuroAgent v0.1.0 · Academic project · Not for clinical use'
        '</div>',
        unsafe_allow_html=True,
    )

    return scan_file, patient_data, run_clicked
