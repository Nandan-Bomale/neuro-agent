"""
frontend/styles.py
------------------
Complete CSS design system for the NeuroAgent Streamlit dashboard.
Refactored for a Futuristic Medical Dashboard aesthetic.

Design language:
  - Deep navy/black background (#0a0e1a)
  - Glassmorphism cards with cyan (#00c8ff) and purple (#7b2ff7) accents
  - Font: Outfit (Headings), Inter (Body)
  - Dynamic micro-animations on hover
"""

from __future__ import annotations

import streamlit as st

_FONT_LINK = (
    '<link href="https://fonts.googleapis.com/css2?family=Inter:'
    'wght@300;400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
)

CSS = """
/* ═══════════════════════════════════════════════════════════════
   NeuroAgent Futuristic Medical Dashboard
   ═══════════════════════════════════════════════════════════════ */

/* === Base & Font === */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #ffffff;
}

h1, h2, h3, h4, h5, h6, .outfit-font {
    font-family: 'Outfit', sans-serif !important;
}

/* === App Background === */
.stApp {
    background-color: #0a0e1a !important;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(123, 47, 247, 0.05), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(0, 200, 255, 0.05), transparent 25%);
    background-attachment: fixed !important;
}

/* Override block container to be wider and remove top padding */
.block-container {
    padding-top: 2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1400px !important;
}

/* Hide Streamlit Chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* === Custom Scrollbar === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2a3350; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00c8ff; }

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid rgba(0, 200, 255, 0.15) !important;
    width: 250px !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
}

/* ── Header Top Bar ────────────────────────────────────────────── */
.top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(123, 47, 247, 0.2);
    margin-bottom: 2rem;
}
.header-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.header-logo span:first-child {
    font-size: 2.2rem;
    filter: drop-shadow(0 0 10px rgba(0,200,255,0.6));
}
.header-search {
    background: rgba(13, 18, 37, 0.8);
    border: 1px solid rgba(0, 200, 255, 0.4);
    border-radius: 20px;
    padding: 0.5rem 1.5rem;
    width: 400px;
    color: #8892b0;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: inset 0 0 10px rgba(0,200,255,0.05);
}
.header-profile {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
}
.online-dot {
    width: 8px;
    height: 8px;
    background-color: #00ff88;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #00ff88;
}

/* ── Glassmorphism Agent Cards ──────────────────────────────────── */
.agent-card {
    background: rgba(13, 18, 37, 0.6);
    border: 1px solid transparent;
    background-clip: padding-box;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    position: relative;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
}

/* Gradient border trick using pseudo element */
.agent-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: -1;
    margin: -1px;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(123, 47, 247, 0.7), rgba(0, 200, 255, 0.7));
    transition: opacity 0.3s ease;
    opacity: 0.5;
}
.agent-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 200, 255, 0.15);
}
.agent-card:hover::before {
    opacity: 1;
}

.ac-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}
.ac-icon {
    font-size: 1.4rem;
    background: linear-gradient(135deg, #7b2ff7, #00c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ac-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #ffffff;
    line-height: 1.2;
}
.ac-subtitle {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8892b0;
}
.ac-body {
    font-size: 0.85rem;
    color: #c0c6d4;
    line-height: 1.6;
    margin-top: 0.8rem;
}

/* ── Central MRI Display ────────────────────────────────────────── */
.mri-center-panel {
    background: rgba(10, 14, 26, 0.8);
    border: 1px solid rgba(0, 200, 255, 0.25);
    border-radius: 24px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    box-shadow: 0 0 50px rgba(0, 200, 255, 0.05);
}
.mri-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #ffffff;
}
.mri-image-wrapper {
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.05);
}
.mri-progress-bar {
    width: 60%;
    height: 4px;
    background: rgba(255,255,255,0.1);
    margin: 0 auto;
    border-radius: 2px;
    overflow: hidden;
}
.mri-progress-fill {
    height: 100%;
    width: 100%;
    background: linear-gradient(90deg, #00c8ff, #7b2ff7);
    animation: scan 3s ease-in-out infinite;
}
@keyframes scan {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* ── Confidence Gauge (CSS Semi-circle) ────────────────────────── */
.confidence-gauge-container {
    text-align: center;
    margin-top: 1rem;
}
.gauge-semi {
    position: relative;
    width: 160px;
    height: 80px;
    margin: 0 auto;
    overflow: hidden;
}
.gauge-semi::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 160px; height: 160px;
    border-radius: 50%;
    border: 12px solid rgba(0, 200, 255, 0.1);
    border-bottom-color: transparent;
    border-right-color: transparent;
    transform: rotate(45deg);
    box-sizing: border-box;
}
.gauge-fill {
    position: absolute;
    top: 0; left: 0;
    width: 160px; height: 160px;
    border-radius: 50%;
    border: 12px solid #00c8ff;
    border-bottom-color: transparent;
    border-right-color: transparent;
    box-sizing: border-box;
    transform: rotate(-135deg); /* Min */
    transition: transform 1.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    filter: drop-shadow(0 0 8px rgba(0,200,255,0.8));
}
.gauge-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    position: absolute;
    bottom: -5px;
    left: 0; right: 0;
}
.warning-text {
    color: #ff9500;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.5rem;
}
.success-text {
    color: #00ff88;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.5rem;
}

/* ── Literature List ───────────────────────────────────────────── */
.lit-list {
    list-style: none;
    padding: 0;
    margin: 0.8rem 0 0;
}
.lit-list li {
    font-size: 0.8rem;
    color: #c0c6d4;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    gap: 0.5rem;
}
.lit-list li:last-child {
    border-bottom: none;
}
.lit-num {
    color: #00c8ff;
    font-weight: 700;
}

/* ── Custom Streamlit Input Styling ────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7 0%, #00c8ff 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(0, 200, 255, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 30px rgba(0, 200, 255, 0.5) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stFileUploader"] {
    background: rgba(13, 18, 37, 0.6) !important;
    border: 1px dashed rgba(0, 200, 255, 0.5) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00c8ff !important;
    background: rgba(13, 18, 37, 0.8) !important;
}
[data-testid="stFileUploader"] label {
    color: #8892b0 !important;
}
"""

def inject_css() -> None:
    """Inject the full design system CSS and Google Fonts into the Streamlit page."""
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
