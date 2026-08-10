"""
frontend/components/report_viewer.py
--------------------------------------
Full analysis results renderer using a responsive 3-column grid.
"""

from __future__ import annotations
import json
from typing import Any
import streamlit as st
from frontend.components.heatmap_display import render_heatmap_centerpiece

def _render_vision_card(vision_summary: dict, confidence: float, label: str) -> None:
    detected = vision_summary.get("tumour_detected", False)
    status = "Confirmed" if detected else "Clear"
    color = "#ff4444" if detected else "#00ff88"
    
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">👁️</div>
                <div>
                    <div class="ac-title">Vision Agent</div>
                    <div class="ac-subtitle">Tumor Segmentation</div>
                </div>
            </div>
            <div class="ac-body" style="text-align:center; margin-top:1rem;">
                <div style="font-family:'Outfit'; font-size:1.5rem; font-weight:700; color:{color}; margin-bottom:0.5rem;">{status}</div>
                <div style="font-size:0.8rem; color:#8892b0;">Confidence: {int(confidence*100)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def _render_report_card(report: dict) -> None:
    summary = report.get("impression", "Generating report...")
    if len(summary) > 120:
        summary = summary[:117] + "..."
        
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">📝</div>
                <div>
                    <div class="ac-title">Report Gen Agent</div>
                    <div class="ac-subtitle">Report Preview</div>
                </div>
            </div>
            <div class="ac-body">
                <strong>Summary:</strong> {summary}<br><br>
                <strong>Status:</strong> <span style="color:#00c8ff;">Draft Generated</span>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def _render_clinical_card(patient: dict) -> None:
    age = patient.get("age", "?")
    sex = patient.get("sex", "?")
    symptoms = ", ".join(patient.get("symptoms", [])) or "None reported"
    meds = ", ".join(patient.get("medications", [])) or "None reported"
    
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">📋</div>
                <div>
                    <div class="ac-title">Clinical History Agent</div>
                    <div class="ac-subtitle">Patient Details</div>
                </div>
            </div>
            <div class="ac-body">
                <div style="font-family:'Outfit'; font-size:1.1rem; font-weight:600; color:#fff; margin-bottom:0.5rem;">Patient ID: {id(patient) % 10000:04d} | {sex} {age}</div>
                <strong>Symptoms:</strong> {symptoms}<br>
                <strong>Meds:</strong> {meds}
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def _render_verification_card(confidence: float) -> None:
    pct = int(confidence * 100)
    rot = (pct / 100) * 180 - 135
    
    if pct >= 80:
        msg = '<div class="success-text">Ready for Review</div>'
    else:
        msg = '<div class="warning-text">Warning: Manual Review Recommended</div>'
        
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">🛡️</div>
                <div>
                    <div class="ac-title">Verification Agent</div>
                    <div class="ac-subtitle">AI Confidence Score</div>
                </div>
            </div>
            <div class="confidence-gauge-container">
                <div class="gauge-semi">
                    <div class="gauge-fill" style="transform: rotate({rot}deg);"></div>
                    <div class="gauge-value">{pct}%</div>
                </div>
                {msg}
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def _render_literature_card(cited_literature: list) -> None:
    lis = ""
    for i, cit in enumerate(cited_literature[:2]):
        title = cit.get("title", "Unknown")
        year = cit.get("year", "")
        lis += f'<li><span class="lit-num">{i+1}.</span> <span>{title} ({year})</span></li>'
        
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">📚</div>
                <div>
                    <div class="ac-title">Literature Agent</div>
                    <div class="ac-subtitle">Research Articles</div>
                </div>
            </div>
            <ul class="lit-list">
                {lis if lis else "<li style='color:#8892b0;'>No relevant literature found.</li>"}
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

def render_results(api_response: dict[str, Any], scan_bytes: bytes, scan_filename: str) -> None:
    st.markdown('<div style="padding-top: 1rem;"></div>', unsafe_allow_html=True)
    
    report          = api_response.get("report", {})
    vision_summary  = api_response.get("vision_summary", {})
    confidence      = float(api_response.get("confidence", 0.0))
    label           = api_response.get("confidence_label", "LOW")
    heatmap_b64     = api_response.get("heatmap_b64", "")
    cited_lit       = report.get("cited_literature", [])
    
    # Extract patient mock data from frontend state since API response doesn't mirror it back entirely
    patient_data = st.session_state.get("patient_data", {"age": 42, "sex": "F", "symptoms": ["Migraines"], "medications": ["Temozolomide"]})

    # The 3-Column Layout
    col_left, col_center, col_right = st.columns([1, 1.4, 1], gap="large")
    
    with col_left:
        _render_vision_card(vision_summary, confidence, label)
        _render_report_card(report)
        
    with col_center:
        if heatmap_b64:
            render_heatmap_centerpiece(heatmap_b64, scan_filename)
        else:
            st.warning("No heatmap data available.")
            
    with col_right:
        _render_clinical_card(patient_data)
        _render_verification_card(confidence)
        _render_literature_card(cited_lit)
