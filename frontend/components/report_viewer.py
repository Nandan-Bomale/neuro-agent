"""
frontend/components/report_viewer.py
--------------------------------------
Full analysis results renderer using a responsive 3-column grid and SVGs.
"""

from __future__ import annotations
import json
from typing import Any
import streamlit as st
from frontend.components.heatmap_display import render_heatmap_centerpiece

def _render_vision_card(vision_summary: dict, confidence: float, label: str) -> None:
    detected = vision_summary.get("tumour_detected", False)
    status = "Confirmed" if detected else "Clear"
    status_class = "status-confirmed" if detected else "status-clear"
    
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                </div>
                <div>
                    <div class="ac-title">Vision Agent</div>
                    <div class="ac-subtitle">Tumor Segmentation</div>
                </div>
            </div>
            <div class="ac-body" style="text-align:center;">
                <div class="{status_class}">{status}</div>
                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">Confidence: <span style="color:#e2e8f0;font-weight:600;">{int(confidence*100)}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def _render_report_card(report: dict) -> None:
    summary = report.get("impression", "Generating report...")
    if len(summary) > 130:
        summary = summary[:127] + "..."
        
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>
                </div>
                <div>
                    <div class="ac-title">Report Gen Agent</div>
                    <div class="ac-subtitle">Report Preview</div>
                </div>
            </div>
            <div class="ac-body">
                <span class="text-highlight">Summary:</span> {summary}<br><br>
                <div class="status-msg" style="color:#00c8ff;">Draft Generated</div>
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
                <div class="ac-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/><path d="M8 14h.01"/><path d="M12 14h.01"/><path d="M16 14h.01"/><path d="M8 18h.01"/><path d="M12 18h.01"/><path d="M16 18h.01"/></svg>
                </div>
                <div>
                    <div class="ac-title">Clinical History Agent</div>
                    <div class="ac-subtitle">Patient Details</div>
                </div>
            </div>
            <div class="ac-body">
                <div style="font-family:'Outfit'; font-size:1.15rem; font-weight:600; color:#fff; margin-bottom:0.8rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.5rem;">Patient ID: {id(patient) % 10000:04d} | {sex} {age}</div>
                <span class="text-highlight">Symptoms:</span> {symptoms}<br><br>
                <span class="text-highlight">Meds:</span> {meds}
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def _render_verification_card(confidence: float) -> None:
    pct = int(confidence * 100)
    rot = (pct / 100) * 180 - 135
    
    if pct >= 80:
        msg = '<div class="status-msg" style="color:#00ff88;">Ready for Review</div>'
    else:
        msg = '<div class="status-msg" style="color:#ff9500;">Warning: Manual Review Recommended</div>'
        
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                </div>
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
        lis += f'<li><span class="lit-num">{i+1}.</span> <div><span class="lit-title">{title}</span> ({year})</div></li>'
        
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="ac-header">
                <div class="ac-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>
                </div>
                <div>
                    <div class="ac-title">Literature Agent</div>
                    <div class="ac-subtitle">Research Articles</div>
                </div>
            </div>
            <ul class="lit-list">
                {lis if lis else "<li style='color:#64748b; font-style:italic;'>No relevant literature found.</li>"}
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

def render_results(api_response: dict[str, Any], scan_bytes: bytes, scan_filename: str) -> None:
    report          = api_response.get("report", {})
    vision_summary  = api_response.get("vision_summary", {})
    confidence      = float(api_response.get("confidence", 0.0))
    label           = api_response.get("confidence_label", "LOW")
    heatmap_b64     = api_response.get("heatmap_b64", "")
    cited_lit       = report.get("cited_literature", [])
    
    patient_data = st.session_state.get("patient_data", {"age": 45, "sex": "M", "symptoms": ["headache"], "medications": []})

    col_left, col_center, col_right = st.columns([1, 1.4, 1], gap="medium")
    
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
