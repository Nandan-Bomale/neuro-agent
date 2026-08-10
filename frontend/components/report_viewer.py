"""
frontend/components/report_viewer.py
--------------------------------------
Full analysis results renderer — everything shown in the main area after the
pipeline completes.

Sections rendered (in order)
-----------------------------
  1. Status banner      — "Requires Human Review" (red) or "Analysis Complete" (green)
  2. Confidence badge + key metric tiles
  3. MRI + heatmap visualisation (delegates to heatmap_display.py)
  4. Structured report  — tabbed: Findings | Impression | Recommendations | Reasoning
  5. Literature citations — expandable citation cards
  6. Technical details  — collapsible: run_id, models, processing time
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from frontend.components.heatmap_display import render_heatmap_display

logger = logging.getLogger(__name__)


# ── Internal renderers ─────────────────────────────────────────────────────────


def _render_status_banner(requires_review: bool, verification_notes: str) -> None:
    """Render the red (review required) or green (pass) status banner."""
    if requires_review:
        st.markdown(
            f"""
            <div class="banner-review">
                <span class="banner-icon">⚠️</span>
                <div>
                    <div class="banner-title-review">Human Review Required</div>
                    <div class="banner-text">{verification_notes or
                    'Confidence below threshold. This case must be reviewed by a radiologist before clinical action.'}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="banner-pass">
                <span class="banner-icon">✅</span>
                <div>
                    <div class="banner-title-pass">Analysis Complete — Within Confidence Threshold</div>
                    <div class="banner-text">{verification_notes or
                    'All agents completed successfully. Report is ready for radiologist review.'}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_confidence_panel(
    confidence: float,
    label: str,
    vision_summary: dict[str, Any],
    requires_review: bool,
    processing_time: float,
) -> None:
    """
    Render the circular confidence badge alongside three key metric tiles.
    """
    badge_class = f"cb-{label.lower()}"
    label_class = f"cl-{label.lower()}"
    pct = int(round(confidence * 100))

    badge_html = f"""
    <div class="confidence-wrap">
        <div class="confidence-badge {badge_class}">
            <span class="cb-score">{pct}%</span>
            <span class="cb-label {label_class}">{label}</span>
            <span class="cb-sub">Overall Confidence</span>
        </div>
    </div>
    """

    detected = vision_summary.get("tumour_detected", False)
    vol_cc   = vision_summary.get("tumour_volume_cc", 0.0)

    detected_val   = "Detected" if detected else "Not Found"
    detected_color = "#f87171" if detected else "#34d399"
    review_val     = "Required" if requires_review else "Not Required"
    review_color   = "#fbbf24" if requires_review else "#34d399"

    metrics_html = f"""
    <div class="metric-grid">
        <div class="metric-tile">
            <div class="metric-val" style="color:{detected_color};">{detected_val}</div>
            <div class="metric-lbl">Tumour Status</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{vol_cc:.1f} <span style="font-size:1rem;font-weight:500;color:#64748b;">cm³</span></div>
            <div class="metric-lbl">Lesion Volume</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val" style="color:{review_color};">{review_val}</div>
            <div class="metric-lbl">Human Review</div>
        </div>
    </div>
    <div style="text-align:right;color:#334155;font-size:0.65rem;margin-top:-0.3rem;">
        ⏱ Pipeline completed in <strong style="color:#475569;">{processing_time:.2f}s</strong>
    </div>
    """

    col_badge, col_metrics = st.columns([1, 2.2], gap="large")
    with col_badge:
        st.markdown(badge_html, unsafe_allow_html=True)
    with col_metrics:
        st.markdown(metrics_html, unsafe_allow_html=True)


def _render_report_tabs(report: dict[str, Any]) -> None:
    """Render the structured report in a tabbed interface."""
    st.markdown(
        '<div class="section-header">📋 Clinical Report</div>',
        unsafe_allow_html=True,
    )

    tab_findings, tab_impression, tab_recs, tab_reasoning = st.tabs(
        ["🔍 Findings", "💡 Impression", "📌 Recommendations", "🔗 Reasoning"]
    )

    with tab_findings:
        findings = report.get("findings", "No findings recorded.")
        st.markdown(
            f'<div class="report-text">{findings}</div>',
            unsafe_allow_html=True,
        )

    with tab_impression:
        impression = report.get("impression", "No impression recorded.")
        st.markdown(
            f'<div class="report-text">{impression}</div>',
            unsafe_allow_html=True,
        )

    with tab_recs:
        recs = report.get("recommendations", "No recommendations recorded.")
        # Render numbered list nicely if newline-separated
        lines = [ln.strip() for ln in recs.split("\n") if ln.strip()]
        if lines:
            items_html = "".join(
                f'<div style="display:flex;gap:0.75rem;margin-bottom:0.6rem;">'
                f'<span style="color:#6366f1;font-weight:700;flex-shrink:0;">{i+1}.</span>'
                f'<span class="report-text" style="margin:0;">{line.lstrip("0123456789. ")}</span>'
                f"</div>"
                for i, line in enumerate(lines)
            )
            st.markdown(items_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="report-text">{recs}</div>',
                unsafe_allow_html=True,
            )

    with tab_reasoning:
        reasoning = report.get("reasoning", "Reasoning chain not available.")
        st.markdown(
            f'<div class="report-reasoning">{reasoning}</div>',
            unsafe_allow_html=True,
        )


def _render_citations(cited_literature: list[dict]) -> None:
    """Render literature citations as expandable styled cards."""
    n = len(cited_literature)
    if n == 0:
        return

    st.markdown(
        f'<div class="section-header">📚 Literature Citations '
        f'<span style="color:#334155;font-weight:400;font-size:0.8rem;'
        f'text-transform:none;letter-spacing:0;">({n} paper{"s" if n != 1 else ""} retrieved)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for i, cit in enumerate(cited_literature):
        title   = cit.get("title", "Unknown Title")
        authors = cit.get("authors", [])
        journal = cit.get("journal", "")
        year    = cit.get("year")
        pmid    = cit.get("pubmed_id") or cit.get("pmid")
        snippet = cit.get("relevance_snippet", "")
        rel     = float(cit.get("relevance_score", 0.0))
        cit_str = cit.get("citation_string") or cit.get("citation", "")

        # Author string
        if authors:
            if len(authors) > 3:
                author_str = ", ".join(authors[:3]) + " et al."
            else:
                author_str = ", ".join(authors)
        else:
            author_str = "Authors not listed"

        # Meta line
        meta_parts = [author_str]
        if journal:
            meta_parts.append(journal)
        if year:
            meta_parts.append(str(year))
        meta_str = " · ".join(meta_parts)

        pmid_html = (
            f'<a class="cit-pmid" href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
            f'target="_blank">PubMed ↗</a>'
            if pmid
            else ""
        )

        rel_pct = int(rel * 100)
        rel_bar_width = f"{rel_pct}%"

        with st.expander(f"📄  {title[:80]}{'…' if len(title) > 80 else ''}", expanded=False):
            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="cit-title">{title}</div>
                    <div class="cit-meta">{meta_str}{pmid_html}</div>
                    {f'<div class="cit-snippet">"{snippet}"</div>' if snippet else ''}
                    <div class="cit-rel-bar-wrap">
                        <span class="cit-rel-label">Relevance</span>
                        <div class="cit-rel-bar">
                            <div class="cit-rel-fill" style="width:{rel_bar_width};"></div>
                        </div>
                        <span class="cit-rel-label">{rel_pct}%</span>
                    </div>
                    {f'<div style="font-size:0.72rem;color:#334155;margin-top:0.5rem;font-style:italic;">{cit_str}</div>' if cit_str else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_explanation(explanation_summary: str) -> None:
    """Render the Explainability Agent's plain-language summary."""
    if not explanation_summary:
        return
    st.markdown(
        f"""
        <div class="neuro-card" style="border-color:rgba(20,184,166,0.2);">
            <div class="neuro-card-title">🎯 Explainability — Why This Region?</div>
            <div class="neuro-card-body">{explanation_summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_technical_details(
    api_response: dict[str, Any],
    run_id: str,
    processing_time: float,
) -> None:
    """Render technical pipeline metadata in a collapsible expander."""
    metadata = api_response.get("metadata", {})
    vision  = api_response.get("vision_summary", {})

    rows = [
        ("Run ID",         run_id[:36]),
        ("Pipeline Mode",  metadata.get("mode", "—")),
        ("Vision Model",   metadata.get("vision_model") or vision.get("model_version", "—")),
        ("LLM Model",      metadata.get("llm_model", "—")),
        ("RAG Backend",    metadata.get("rag_backend", "—")),
        ("Pipeline Status",api_response.get("pipeline_status", "—")),
        ("Processing Time",f"{processing_time:.3f} s"),
        ("Confidence Threshold", "0.80 (HIGH) / 0.60 (MEDIUM)"),
        ("GradCAM Slice",  str(vision.get("gradcam_slice", "—"))),
        ("Tumour Voxels",  str(vision.get("tumour_volume_voxels", "—"))),
    ]

    tiles_html = "".join(
        f'<div class="tech-detail-row">'
        f'<span class="tech-key">{k}</span>'
        f'<span class="tech-val">{v}</span>'
        f"</div>"
        for k, v in rows
    )

    with st.expander("⚙️  Technical Details", expanded=False):
        st.markdown(
            f'<div class="tech-detail-grid">{tiles_html}</div>',
            unsafe_allow_html=True,
        )
        if run_id:
            st.markdown(
                f'<div style="font-size:0.65rem;color:#334155;margin-top:0.8rem;">'
                f'Full run_id: <code style="color:#475569;">{run_id}</code></div>',
                unsafe_allow_html=True,
            )


# ── Download helpers ───────────────────────────────────────────────────────────


def _build_download_json(api_response: dict[str, Any]) -> str:
    """Serialise the API response to a JSON string for download (heatmap excluded)."""
    import json  # noqa: PLC0415

    download_copy = {k: v for k, v in api_response.items() if k != "heatmap_b64"}
    return json.dumps(download_copy, indent=2, ensure_ascii=False, default=str)


# ── Public entry point ─────────────────────────────────────────────────────────


def render_results(
    api_response: dict[str, Any],
    scan_bytes: bytes,
    scan_filename: str,
) -> None:
    """
    Render the complete analysis results dashboard.

    Called from app.py once the API response is available in session_state.

    Args:
        api_response:  Full JSON dict from POST /api/analyze.
        scan_bytes:    Raw bytes of the uploaded MRI scan (for original view).
        scan_filename: Original filename of the uploaded scan.
    """
    report          = api_response.get("report", {})
    vision_summary  = api_response.get("vision_summary", {})
    confidence      = float(api_response.get("confidence", 0.0))
    label           = api_response.get("confidence_label", "LOW")
    requires_review = bool(api_response.get("requires_review", False))
    heatmap_b64     = api_response.get("heatmap_b64", "")
    explanation     = api_response.get("explanation_summary", "")
    verification    = api_response.get("verification_notes", "")
    pipeline_status = api_response.get("pipeline_status", "complete")
    processing_time = float(api_response.get("processing_time_s", 0.0))
    run_id          = api_response.get("run_id", "")

    # ── 1. Status banner ──────────────────────────────────────────────────────
    _render_status_banner(requires_review, verification)

    # ── 2. Confidence badge + metrics ─────────────────────────────────────────
    _render_confidence_panel(
        confidence, label, vision_summary, requires_review, processing_time
    )

    # ── 3. Scan visualisation ─────────────────────────────────────────────────
    if heatmap_b64:
        render_heatmap_display(
            scan_bytes=scan_bytes,
            scan_filename=scan_filename,
            heatmap_b64=heatmap_b64,
            vision_summary=vision_summary,
        )

    # ── 4. Structured report tabs ─────────────────────────────────────────────
    cited_literature = report.get("cited_literature", [])
    _render_report_tabs(report)

    # ── 5. Explainability summary ─────────────────────────────────────────────
    _render_explanation(explanation)

    # ── 6. Literature citations ───────────────────────────────────────────────
    _render_citations(cited_literature)

    # ── 7. Download + technical details ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_dl, col_tech = st.columns([1, 2])
    with col_dl:
        json_str = _build_download_json(api_response)
        st.download_button(
            label="⬇️  Download Report (JSON)",
            data=json_str,
            file_name=f"neuroagent_report_{run_id[:8]}.json",
            mime="application/json",
            use_container_width=True,
            key="download_report",
        )
    with col_tech:
        _render_technical_details(api_response, run_id, processing_time)
