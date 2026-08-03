"""
frontend/styles.py
------------------
Complete CSS design system for the NeuroAgent Streamlit dashboard.

Design language:
  - Deep navy dark-mode background
  - Glassmorphism cards with subtle indigo borders
  - Inter typeface (Google Fonts)
  - Electric indigo (#6366f1) primary
  - Medical teal (#14b8a6) secondary / accents
  - Animated confidence badges (pulsing ring glow)
  - Smooth hover transitions throughout

Usage:
    from frontend.styles import inject_css
    inject_css()   # call once, right after st.set_page_config()
"""

from __future__ import annotations

import streamlit as st

# ── Google Fonts ───────────────────────────────────────────────────────────────
_FONT_LINK = (
    '<link href="https://fonts.googleapis.com/css2?family=Inter:'
    'wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">'
)

# ── Full CSS ───────────────────────────────────────────────────────────────────
CSS = """
/* ═══════════════════════════════════════════════════════════════
   NeuroAgent Design System  ·  Dark Medical Dashboard
   ═══════════════════════════════════════════════════════════════ */

/* === Base & Font === */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* === App Background === */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #0f1e3d 0%, #070c18 55%, #080e1f 100%) !important;
    background-attachment: fixed !important;
}

/* === Hide Streamlit Chrome === */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* === Custom Scrollbar === */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #070c18; }
::-webkit-scrollbar-thumb { background: #1e3060; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e22 0%, #0a1228 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebarContent"] {
    background: transparent !important;
}

/* ── Sidebar section labels ─────────────────────────────────────── */
.sb-section {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6366f1;
    margin: 1.4rem 0 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99,102,241,0.18);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Sidebar logo block ──────────────────────────────────────────── */
.sb-logo {
    text-align: center;
    padding: 0.5rem 0 1.2rem;
    border-bottom: 1px solid rgba(99,102,241,0.12);
    margin-bottom: 0.2rem;
}
.sb-logo-icon {
    font-size: 2.2rem;
    display: block;
    filter: drop-shadow(0 0 12px rgba(99,102,241,0.8));
    margin-bottom: 0.3rem;
}
.sb-logo-title {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
}
.sb-logo-sub {
    font-size: 0.65rem;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.1rem;
}

/* ── Main area header ────────────────────────────────────────────── */
.main-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.1);
    margin-bottom: 2rem;
}
.main-header-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 24px rgba(99,102,241,0.9));
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}
.main-header-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 40%, #14b8a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.05em;
    line-height: 1.1;
    margin: 0;
}
.main-header-sub {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Glassmorphism card ──────────────────────────────────────────── */
.neuro-card {
    background: rgba(12, 20, 40, 0.75);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    backdrop-filter: blur(24px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.neuro-card:hover {
    border-color: rgba(99,102,241,0.28);
    box-shadow: 0 4px 30px rgba(99,102,241,0.08);
}
.neuro-card-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.neuro-card-body {
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.75;
}

/* ── Section divider ─────────────────────────────────────────────── */
.section-header {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6366f1;
    margin: 2.2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Confidence badge ────────────────────────────────────────────── */
.confidence-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem 0 1rem;
}
.confidence-badge {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 168px;
    height: 168px;
    border-radius: 50%;
    position: relative;
    cursor: default;
    user-select: none;
}
.cb-high {
    background: radial-gradient(circle, rgba(4,120,87,0.25), rgba(5,150,105,0.08));
    border: 3px solid #059669;
    box-shadow: 0 0 0 0 rgba(5,150,105,0.5);
    animation: glow-high 2.8s ease-in-out infinite;
}
.cb-medium {
    background: radial-gradient(circle, rgba(180,83,9,0.25), rgba(217,119,6,0.08));
    border: 3px solid #d97706;
    box-shadow: 0 0 24px rgba(217,119,6,0.25);
}
.cb-low {
    background: radial-gradient(circle, rgba(185,28,28,0.3), rgba(220,38,38,0.08));
    border: 3px solid #dc2626;
    box-shadow: 0 0 0 0 rgba(220,38,38,0.6);
    animation: glow-low 1.6s ease-in-out infinite;
}
@keyframes glow-high {
    0%,100% { box-shadow: 0 0 15px rgba(5,150,105,0.2), 0 0 0 0 rgba(5,150,105,0.3); }
    50%      { box-shadow: 0 0 35px rgba(5,150,105,0.5), 0 0 0 10px rgba(5,150,105,0); }
}
@keyframes glow-low {
    0%,100% { box-shadow: 0 0 15px rgba(220,38,38,0.3), 0 0 0 0 rgba(220,38,38,0.5); }
    50%      { box-shadow: 0 0 40px rgba(220,38,38,0.7), 0 0 0 12px rgba(220,38,38,0); }
}
.cb-score {
    font-size: 2.7rem;
    font-weight: 900;
    color: #f1f5f9;
    line-height: 1;
    letter-spacing: -0.04em;
}
.cb-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.cb-sub {
    font-size: 0.62rem;
    color: #64748b;
    margin-top: 0.15rem;
    letter-spacing: 0.05em;
}
.cl-high   { color: #34d399; }
.cl-medium { color: #fbbf24; }
.cl-low    { color: #f87171; }

/* ── Metric cards ────────────────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 0.5rem 0 1.2rem;
}
.metric-tile {
    background: rgba(12,20,40,0.7);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 14px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: border-color 0.25s ease;
}
.metric-tile:hover { border-color: rgba(99,102,241,0.3); }
.metric-val {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.metric-lbl {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 0.3rem;
}

/* ── Status banners ──────────────────────────────────────────────── */
.banner-review {
    background: linear-gradient(135deg, rgba(127,29,29,0.55), rgba(153,27,27,0.35));
    border: 1px solid rgba(220,38,38,0.45);
    border-left: 4px solid #dc2626;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
}
.banner-pass {
    background: linear-gradient(135deg, rgba(4,120,87,0.35), rgba(6,95,70,0.2));
    border: 1px solid rgba(5,150,105,0.35);
    border-left: 4px solid #059669;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.banner-icon { font-size: 1.5rem; flex-shrink: 0; }
.banner-title-review { font-weight: 700; color: #f87171; font-size: 0.92rem; }
.banner-title-pass   { font-weight: 700; color: #34d399; font-size: 0.92rem; }
.banner-text { color: #94a3b8; font-size: 0.82rem; margin-top: 0.2rem; line-height: 1.5; }

/* ── Report tab content ──────────────────────────────────────────── */
.report-text {
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.8;
    padding: 0.2rem 0;
}
.report-reasoning {
    color: #94a3b8;
    font-size: 0.86rem;
    line-height: 1.75;
    font-family: 'Inter', monospace;
    background: rgba(7,12,24,0.6);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}

/* ── Citation cards ──────────────────────────────────────────────── */
.citation-card {
    background: rgba(8,14,30,0.6);
    border: 1px solid rgba(20,184,166,0.14);
    border-radius: 11px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-left: 3px solid #14b8a6;
    transition: border-left-color 0.2s ease, box-shadow 0.2s ease;
}
.citation-card:hover {
    border-left-color: #6366f1;
    box-shadow: 0 2px 16px rgba(99,102,241,0.08);
}
.cit-title {
    font-weight: 600;
    color: #e2e8f0;
    font-size: 0.88rem;
    line-height: 1.45;
    margin-bottom: 0.3rem;
}
.cit-meta {
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 0.45rem;
}
.cit-pmid {
    display: inline-block;
    font-size: 0.67rem;
    color: #14b8a6;
    background: rgba(20,184,166,0.1);
    border: 1px solid rgba(20,184,166,0.2);
    border-radius: 4px;
    padding: 1px 6px;
    margin-left: 0.4rem;
    text-decoration: none;
    vertical-align: middle;
}
.cit-snippet {
    font-size: 0.8rem;
    color: #94a3b8;
    font-style: italic;
    line-height: 1.55;
    border-top: 1px solid rgba(99,102,241,0.1);
    padding-top: 0.45rem;
    margin-top: 0.45rem;
}
.cit-rel-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.5rem;
}
.cit-rel-label { font-size: 0.67rem; color: #475569; }
.cit-rel-bar {
    flex: 1;
    height: 3px;
    background: rgba(99,102,241,0.15);
    border-radius: 2px;
    overflow: hidden;
}
.cit-rel-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #14b8a6, #6366f1);
}

/* ── Image containers ────────────────────────────────────────────── */
.img-panel {
    background: #050a14;
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px;
    overflow: hidden;
    position: relative;
}
.img-panel-label {
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(0,0,0,0.82);
    color: #94a3b8;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    backdrop-filter: blur(8px);
    z-index: 10;
    pointer-events: none;
}

/* ── Technical details ───────────────────────────────────────────── */
.tech-detail-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
}
.tech-detail-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.6rem;
    background: rgba(12,20,40,0.5);
    border-radius: 7px;
    border: 1px solid rgba(99,102,241,0.08);
}
.tech-key {
    font-size: 0.65rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    min-width: 80px;
}
.tech-val {
    font-size: 0.75rem;
    color: #94a3b8;
    font-family: 'Inter', monospace;
    word-break: break-all;
}

/* ── Landing page ────────────────────────────────────────────────── */
.landing-hero {
    text-align: center;
    padding: 4rem 2rem 2.5rem;
}
.landing-hero-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.landing-hero-sub {
    color: #475569;
    font-size: 0.9rem;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    max-width: 780px;
    margin: 0 auto 2.5rem;
}
.feature-tile {
    background: rgba(12,20,40,0.6);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px;
    padding: 1.5rem 1rem;
    transition: border-color 0.3s ease, transform 0.3s ease;
    cursor: default;
}
.feature-tile:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.12);
}
.feature-icon { font-size: 1.8rem; display: block; margin-bottom: 0.6rem; }
.feature-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.35rem;
}
.feature-desc { font-size: 0.75rem; color: #64748b; line-height: 1.55; }
.pipeline-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
    flex-wrap: wrap;
    padding: 1.2rem;
    background: rgba(12,20,40,0.5);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px;
    max-width: 780px;
    margin: 0 auto;
}
.pipeline-node {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #a5b4fc;
}
.pipeline-arrow { color: #334155; font-size: 0.9rem; font-weight: 700; }

/* ── Streamlit widget overrides ──────────────────────────────────── */
/* Run button */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.4rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.38) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover:not(:disabled) {
    background: linear-gradient(135deg, #3730a3 0%, #4f46e5 100%) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: rgba(51,65,85,0.5) !important;
    color: #475569 !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(12,20,40,0.5) !important;
    border: 1.5px dashed rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
}
[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* Number input, text input, textarea */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(12,20,40,0.55) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.1) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

/* Select box */
[data-testid="stSelectbox"] > div > div {
    background: rgba(12,20,40,0.55) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(12,20,40,0.65) !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid rgba(99,102,241,0.14) !important;
    border-bottom: none !important;
    padding: 4px 4px 0 !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    padding: 0.5rem 1.1rem !important;
    border: none !important;
    transition: color 0.2s ease, background 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.18) !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(12,20,40,0.55) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 0 12px 12px 12px !important;
    padding: 1.5rem 1.4rem !important;
    margin-top: 0 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(12,20,40,0.6) !important;
    border: 1px solid rgba(20,184,166,0.14) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
}
.streamlit-expanderContent {
    background: rgba(8,14,30,0.4) !important;
    border: 1px solid rgba(99,102,241,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #14b8a6) !important;
    border-radius: 4px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* Divider */
hr {
    border-color: rgba(99,102,241,0.12) !important;
    margin: 1.5rem 0 !important;
}
"""


def inject_css() -> None:
    """Inject the full design system CSS and Google Fonts into the Streamlit page."""
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
