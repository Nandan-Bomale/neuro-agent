"""
frontend/components/heatmap_display.py
----------------------------------------
Side-by-side visualisation: Original MRI slice  |  Grad-CAM Heatmap Overlay.

Supported scan formats
----------------------
  PNG / JPG   — displayed directly via PIL.
  NIfTI       — middle axial slice extracted with nibabel (graceful fallback
                if nibabel not installed or file can't be parsed).

The heatmap is always a base64-encoded PNG string returned by the backend API.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)


# ── NIfTI slice extraction ─────────────────────────────────────────────────────


def _extract_nifti_slice(scan_bytes: bytes, filename: str) -> np.ndarray | None:
    """
    Load a NIfTI file from bytes and return the middle axial slice as a
    normalised float32 numpy array (H × W), ready for matplotlib imshow.

    Args:
        scan_bytes: Raw bytes of the .nii / .nii.gz file.
        filename:   Original filename (used to choose temp-file suffix).

    Returns:
        2-D numpy array (H × W) normalised to [0, 1], or None on failure.
    """
    try:
        import nibabel as nib  # noqa: PLC0415

        suffix = ".nii.gz" if filename.lower().endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(scan_bytes)
            tmp_path = tmp.name

        img = nib.load(tmp_path)
        data: np.ndarray = img.get_fdata(dtype=np.float32)

        # Handle 4-D volumes (time series / multi-channel)
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim == 3:
            mid = data.shape[2] // 2
            data = data[:, :, mid]

        # Normalise to [0, 1]
        lo, hi = data.min(), data.max()
        if hi > lo:
            data = (data - lo) / (hi - lo)
        else:
            data = np.zeros_like(data)

        # Rotate to standard orientation (nibabel is RAS+)
        data = np.rot90(data)

        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

        return data

    except ImportError:
        logger.warning("nibabel not installed — cannot display NIfTI slice natively.")
        return None
    except Exception as exc:
        logger.warning("Failed to extract NIfTI slice: %s", exc)
        return None


def _render_original_scan(scan_bytes: bytes, filename: str) -> None:
    """Display the original scan in the left column."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ext = Path(filename).suffix.lower()
    is_nifti = ext in (".nii", ".gz")

    if is_nifti:
        arr = _extract_nifti_slice(scan_bytes, filename)
        if arr is not None:
            fig, ax = plt.subplots(figsize=(4, 4), facecolor="#050a14")
            ax.imshow(arr, cmap="gray", interpolation="bilinear", aspect="auto")
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            return
        # Fallback message if nibabel unavailable
        st.markdown(
            '<div style="color:#475569;font-size:0.8rem;text-align:center;'
            'padding:2rem 1rem;background:rgba(5,10,20,0.6);border-radius:12px;">'
            '📁 NIfTI slice preview requires <code>nibabel</code>.<br>'
            'Install with: <code>pip install nibabel</code></div>',
            unsafe_allow_html=True,
        )
    else:
        # PNG / JPG — display directly
        try:
            pil_img = Image.open(io.BytesIO(scan_bytes)).convert("RGB")
            st.image(pil_img, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not display scan image: {exc}")


# ── Public component ───────────────────────────────────────────────────────────


def render_heatmap_display(
    scan_bytes: bytes,
    scan_filename: str,
    heatmap_b64: str,
    vision_summary: dict[str, Any],
) -> None:
    """
    Render the side-by-side MRI + Grad-CAM heatmap visualisation section.

    Args:
        scan_bytes:     Raw bytes of the uploaded MRI scan file.
        scan_filename:  Original filename (used for format detection and display).
        heatmap_b64:    Base64-encoded PNG string of the Grad-CAM overlay.
        vision_summary: VisionSummary dict from the API response.
    """
    st.markdown(
        '<div class="section-header">🔬 Scan Visualisation</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2, gap="medium")

    # ── Left: original scan ───────────────────────────────────────────────────
    with col_left:
        st.markdown(
            '<div style="text-align:center;font-size:0.62rem;font-weight:700;'
            'letter-spacing:0.12em;text-transform:uppercase;color:#64748b;'
            'margin-bottom:0.6rem;">Original MRI Scan</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="img-panel">', unsafe_allow_html=True)
        _render_original_scan(scan_bytes, scan_filename)
        st.markdown("</div>", unsafe_allow_html=True)

        # Filename caption
        st.markdown(
            f'<div style="text-align:center;color:#334155;font-size:0.65rem;'
            f'margin-top:0.3rem;">{scan_filename}</div>',
            unsafe_allow_html=True,
        )

    # ── Right: Grad-CAM heatmap ───────────────────────────────────────────────
    with col_right:
        st.markdown(
            '<div style="text-align:center;font-size:0.62rem;font-weight:700;'
            'letter-spacing:0.12em;text-transform:uppercase;color:#64748b;'
            'margin-bottom:0.6rem;">Grad-CAM Heatmap Overlay</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="img-panel">', unsafe_allow_html=True)
        try:
            heatmap_bytes = base64.b64decode(heatmap_b64)
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")
            st.image(heatmap_img, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not decode heatmap: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Slice caption + colourbar legend
        grad_slice = vision_summary.get("gradcam_slice", -1)
        vol_cc = vision_summary.get("tumour_volume_cc", 0.0)
        caption_parts = []
        if grad_slice >= 0:
            caption_parts.append(f"Axial slice {grad_slice}")
        if vol_cc > 0:
            caption_parts.append(f"ROI ≈ {vol_cc:.1f} cm³")
        caption = " · ".join(caption_parts) if caption_parts else "Peak saliency highlighted"

        st.markdown(
            f'<div style="text-align:center;color:#334155;font-size:0.65rem;'
            f'margin-top:0.3rem;">{caption}</div>',
            unsafe_allow_html=True,
        )

    # ── Colourmap legend ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:0.6rem;margin-top:0.8rem;">
            <span style="font-size:0.62rem;color:#475569;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.1em;">Low activation</span>
            <div style="width:160px;height:6px;border-radius:3px;
                        background:linear-gradient(90deg,#000080,#0000ff,#00ffff,
                                                         #00ff00,#ffff00,#ff8000,#ff0000);
                        border:1px solid rgba(99,102,241,0.15);"></div>
            <span style="font-size:0.62rem;color:#475569;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.1em;">High activation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
