"""
frontend/styles.py
------------------
Award-winning UI CSS for the NeuroAgent Streamlit dashboard.
Focuses on true glassmorphism, precise typography, and sleek modern layout.
"""

from __future__ import annotations
import streamlit as st

_FONT_LINK = (
    '<link href="https://fonts.googleapis.com/css2?family=Inter:'
    'wght@300;400;500;600&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">'
)

CSS = """
/* ═══════════════════════════════════════════════════════════════
   NeuroAgent Premium UI System
   ═══════════════════════════════════════════════════════════════ */

/* === Base & Typography === */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #e2e8f0;
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3, h4, h5, h6, .outfit-font {
    font-family: 'Outfit', sans-serif !important;
}

/* === App Background === */
.stApp {
    background-color: #060913 !important;
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(123, 47, 247, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(0, 200, 255, 0.08) 0%, transparent 40%);
    background-attachment: fixed !important;
}

/* === Streamlit Chrome Hiding & Layout === */
#MainMenu, footer, header[data-testid="stHeader"] { 
    display: none !important; 
}

/* Remove default padding from the main block to let our header touch the top */
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* === Custom Scrollbar === */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 200, 255, 0.5); }


/* ── Top Header & Navigation ────────────────────────────────────── */
.top-nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.nav-left {
    display: flex;
    align-items: center;
    gap: 3rem;
}

.brand-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #ffffff;
}

.brand-logo svg {
    filter: drop-shadow(0 0 8px rgba(0,200,255,0.8));
}

.main-menu {
    display: flex;
    gap: 2rem;
}
.menu-item {
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748b;
    cursor: pointer;
    transition: color 0.2s;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.menu-item:hover { color: #ffffff; }
.menu-item.active { 
    color: #00c8ff; 
    font-weight: 600;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.search-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 30px;
    padding: 0.5rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #64748b;
    font-size: 0.85rem;
    width: 300px;
}

.profile-section {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: #94a3b8;
}
.profile-section img {
    border-radius: 50%;
    border: 2px solid rgba(0,200,255,0.3);
}
.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
}
.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}


/* ── True Glassmorphism Cards ───────────────────────────────────── */
.agent-card {
    /* The trick for glowing gradient borders with a solid dark background */
    background: 
        linear-gradient(#090d1a, #090d1a) padding-box,
        linear-gradient(135deg, rgba(123, 47, 247, 0.6), rgba(0, 200, 255, 0.6)) border-box;
    border: 1px solid transparent;
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
}

.agent-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 200, 255, 0.15);
}

.ac-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding-bottom: 0.8rem;
}
.ac-icon {
    width: 36px;
    height: 36px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #00c8ff;
}
.ac-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    color: #ffffff;
    line-height: 1.1;
}
.ac-subtitle {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 0.2rem;
}
.ac-body {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Typography Utilities inside cards ─────────────────────────── */
.text-highlight {
    color: #00c8ff;
    font-weight: 500;
}
.status-confirmed {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ff4444; /* Medical red for tumors */
    text-shadow: 0 0 15px rgba(255,68,68,0.4);
    text-align: center;
    margin: 1rem 0;
}
.status-clear {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00ff88;
    text-shadow: 0 0 15px rgba(0,255,136,0.4);
    text-align: center;
    margin: 1rem 0;
}


/* ── Central MRI Display ────────────────────────────────────────── */
.mri-center-panel {
    background: radial-gradient(circle at center, rgba(13, 18, 37, 0.8), rgba(6, 9, 19, 0.9));
    border: 1px solid rgba(0, 200, 255, 0.15);
    border-radius: 24px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: inset 0 0 40px rgba(0,0,0,0.8), 0 0 40px rgba(0, 200, 255, 0.05);
}
.mri-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: #e2e8f0;
    margin-bottom: 1.5rem;
    letter-spacing: 0.05em;
}
.mri-image-wrapper {
    margin: 0 auto;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    /* Limit the height dramatically to fit on one screen */
    max-height: 420px; 
    display: flex;
    align-items: center;
    justify-content: center;
}
/* Ensure the inner image tags obey the height */
.mri-image-wrapper img {
    max-height: 420px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 12px;
}

.mri-progress-bar {
    width: 200px;
    height: 3px;
    background: rgba(255,255,255,0.05);
    margin: 1.5rem auto 0;
    border-radius: 2px;
    overflow: hidden;
}
.mri-progress-fill {
    height: 100%;
    width: 100%;
    background: linear-gradient(90deg, transparent, #00c8ff, #7b2ff7, transparent);
    animation: scan 2.5s ease-in-out infinite;
}
@keyframes scan {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}


/* ── Confidence Gauge (CSS Semi-circle) ────────────────────────── */
.confidence-gauge-container {
    text-align: center;
    padding: 1rem 0;
}
.gauge-semi {
    position: relative;
    width: 140px;
    height: 70px;
    margin: 0 auto;
    overflow: hidden;
}
.gauge-semi::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 140px; height: 140px;
    border-radius: 50%;
    border: 8px solid rgba(255, 255, 255, 0.05);
    border-bottom-color: transparent;
    border-right-color: transparent;
    transform: rotate(45deg);
    box-sizing: border-box;
}
.gauge-fill {
    position: absolute;
    top: 0; left: 0;
    width: 140px; height: 140px;
    border-radius: 50%;
    border: 8px solid #00c8ff;
    border-bottom-color: transparent;
    border-right-color: transparent;
    box-sizing: border-box;
    transform: rotate(-135deg); /* Min */
    transition: transform 1.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    filter: drop-shadow(0 0 10px rgba(0,200,255,0.5));
}
.gauge-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    position: absolute;
    bottom: -5px;
    left: 0; right: 0;
}
.status-msg {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}


/* ── Literature List ───────────────────────────────────────────── */
.lit-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.lit-list li {
    font-size: 0.75rem;
    color: #94a3b8;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    gap: 0.6rem;
    line-height: 1.5;
}
.lit-list li:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.lit-num {
    color: #00c8ff;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
}
.lit-title {
    color: #e2e8f0;
    font-weight: 500;
}


/* ── Custom Streamlit Input Styling ────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7 0%, #00c8ff 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 4px 15px rgba(0, 200, 255, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(0, 200, 255, 0.5) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(0, 200, 255, 0.3) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00c8ff !important;
    background: rgba(255,255,255,0.04) !important;
}
[data-testid="stFileUploader"] label {
    color: #8892b0 !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
"""

def inject_css() -> None:
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
