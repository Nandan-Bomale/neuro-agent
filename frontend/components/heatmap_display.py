"""
frontend/components/heatmap_display.py
----------------------------------------
Renders the centerpiece 3D Brain MRI visualisation.
"""

from __future__ import annotations
import base64
import io
import streamlit as st
from PIL import Image
from typing import Any

def render_heatmap_centerpiece(heatmap_b64: str, scan_filename: str) -> None:
    """Renders the central MRI image with the progress bar underneath."""
    st.markdown('<div class="mri-center-panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="mri-title">3D Brain MRI ({scan_filename})</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mri-image-wrapper">', unsafe_allow_html=True)
    try:
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_img = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")
        st.image(heatmap_img, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not decode heatmap: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Glowing progress bar simulation
    st.markdown(
        """
        <div class="mri-progress-bar">
            <div class="mri-progress-fill"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
