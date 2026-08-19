"""
agent.py
--------
ExplainabilityAgent — Generates Grad-CAM heatmap and plain-language explanation.

Wraps the existing GradCAM class from agents/vision_agent/gradcam.py.
This is the last node on the 'pass' branch of the verification gate.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)
AGENT_NAME = "explainability_agent"


class ExplainabilityAgent:
    """Generates Grad-CAM visual explanations for the pipeline output.
    
    Reads:  state["mri_scan_path"]
            state["vision_findings"]
            state["tumor_classification_findings"]
    Writes: state["gradcam_heatmap_path"]
            state["explanation_summary"]
    """

    def __init__(self, output_dir: str = "data/interim/gradcam_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{AGENT_NAME}] Initialized | output_dir={output_dir}")

    def _build_explanation_text(
        self,
        tumor_type:   Optional[str],
        tumor_grade:  Optional[str],
        idh_status:   Optional[str],
        mgmt_status:  Optional[str],
        volume_cc:    Optional[float],
        os_months:    Optional[float],
        resectability: Optional[float],
        risk:         Optional[str],
    ) -> str:
        """Build a plain-language Tumor Board summary from all agent outputs."""
        
        lines = ["=" * 60]
        lines.append("  AUTONOMOUS NEURO-ONCOLOGY TUMOR BOARD — FINAL REPORT")
        lines.append("=" * 60)

        # --- Vision findings ---
        lines.append("\n📍 IMAGING FINDINGS (Vision Agent)")
        if volume_cc is not None:
            lines.append(f"  • Tumor detected with volume of {volume_cc:.1f} cc")
        else:
            lines.append("  • Tumor region identified on 3D MRI")

        # --- Classification ---
        lines.append("\n🔬 PATHOLOGICAL CLASSIFICATION (Classification Agent)")
        if tumor_type:
            lines.append(f"  • Tumor Type  : {tumor_type.replace('_', ' ').title()}")
        if tumor_grade:
            grade_label = tumor_grade.replace("_", " ").upper()
            lines.append(f"  • Tumor Grade : {grade_label}")
            if tumor_grade == "grade_IV":
                lines.append("    (Glioblastoma Multiforme — most aggressive glioma)")
            elif tumor_grade == "grade_III":
                lines.append("    (Anaplastic glioma — high-grade)")
            elif tumor_grade == "grade_II":
                lines.append("    (Low-grade glioma — slower progression)")

        # --- Radiogenomics ---
        lines.append("\n🧬 MOLECULAR PROFILE (Radiogenomics Agent — AI Predicted)")
        if idh_status:
            idh_label = "Mutant ✅ (Favorable)" if idh_status == "mutant" else "Wildtype ⚠️ (Unfavorable)"
            lines.append(f"  • IDH Mutation : {idh_label}")
        if mgmt_status:
            mgmt_label = "Methylated ✅ (Responds to Chemo)" if mgmt_status == "methylated" else "Unmethylated ⚠️ (Chemo-resistant)"
            lines.append(f"  • MGMT Promoter: {mgmt_label}")

        # --- Surgical ---
        lines.append("\n🔪 SURGICAL PLANNING (Surgical Agent)")
        if resectability is not None:
            pct = int(resectability * 100)
            lines.append(f"  • Resectability Score: {pct}%")
            if resectability >= 0.80:
                lines.append("  • Recommendation: Maximal safe resection (>95% EOR)")
            elif resectability >= 0.60:
                lines.append("  • Recommendation: Subtotal resection feasible")
            else:
                lines.append("  • Recommendation: Biopsy only — eloquent area involvement")

        # --- Prognosis ---
        lines.append("\n📊 PROGNOSIS (Prognostic Agent — Evidence-Based)")
        if os_months is not None:
            lines.append(f"  • Estimated Overall Survival    : {os_months:.1f} months")
        if risk:
            lines.append(f"  • Risk Category                 : {risk.upper()}")

        lines.append("\n" + "=" * 60)
        lines.append("  ⚠️  AI-GENERATED REPORT — For clinical decision support only.")
        lines.append("     Final diagnosis must be confirmed by a qualified physician.")
        lines.append("=" * 60)

        return "\n".join(lines)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainability output.
        
        Reads:  mri_scan_path, vision_findings, tumor_classification_findings,
                radiogenomics_findings, surgical_analysis, prognostic_analysis
        Writes: gradcam_heatmap_path, explanation_summary
        """
        t_start = time.perf_counter()
        logger.info(f"[{AGENT_NAME}] run() invoked.")

        # --- Extract all findings ---
        vision  = state.get("vision_findings", {})
        clf     = state.get("tumor_classification_findings", {})
        radio   = state.get("radiogenomics_findings", {})
        surg    = state.get("surgical_analysis", {})
        prog    = state.get("prognostic_analysis", {})
        scan    = state.get("mri_scan_path", "")

        tumor_type   = clf.get("tumor_type")
        tumor_grade  = clf.get("tumor_grade")
        idh_status   = radio.get("idh_mutation_status")
        mgmt_status  = radio.get("mgmt_methylation_status")
        volume_cc    = vision.get("tumour_volume_cc")
        os_months    = prog.get("overall_survival_months")
        resectability = surg.get("resectability_score")
        risk         = prog.get("risk_category")

        # --- Try to generate Grad-CAM heatmap (from existing vision agent) ---
        heatmap_path = state.get("gradcam_heatmap_path")
        if not heatmap_path:
            try:
                from agents.vision_agent.agent import VisionAgent
                from agents.vision_agent.model import build_unet
                from agents.vision_agent.gradcam import GradCAM
                from agents.vision_agent.transforms import get_single_inference_transforms

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_path = "models/vision/best_model.pth"

                if Path(model_path).exists() and scan:
                    net = build_unet(device)
                    ckpt = torch.load(model_path, map_location=device)
                    net.load_state_dict(ckpt.get("model_state", ckpt))
                    net.eval()

                    gcam = GradCAM(net)
                    transforms = get_single_inference_transforms()
                    data_dict = {"image": scan}
                    processed = transforms(data_dict)
                    tensor = processed["image"].unsqueeze(0).to(device)

                    result = gcam.generate(image_tensor=tensor)
                    overlay = result.get("overlay_image")

                    if overlay is not None:
                        from PIL import Image
                        out_name = f"gradcam_{int(time.time())}.jpg"
                        out_path = self.output_dir / out_name
                        Image.fromarray(overlay).save(out_path)
                        heatmap_path = str(out_path)
                        logger.info(f"[{AGENT_NAME}] Grad-CAM saved: {heatmap_path}")
                else:
                    logger.warning(f"[{AGENT_NAME}] Vision model not found. Skipping Grad-CAM.")
                    heatmap_path = "gradcam_not_available"

            except Exception as e:
                logger.warning(f"[{AGENT_NAME}] Grad-CAM generation failed: {e}")
                heatmap_path = "gradcam_not_available"

        # --- Build plain-language summary ---
        summary = self._build_explanation_text(
            tumor_type    = tumor_type,
            tumor_grade   = tumor_grade,
            idh_status    = idh_status,
            mgmt_status   = mgmt_status,
            volume_cc     = volume_cc,
            os_months     = os_months,
            resectability = resectability,
            risk          = risk,
        )

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info(f"[{AGENT_NAME}] Done in {elapsed}s.")
        print(f"\n[{AGENT_NAME}] TUMOR BOARD REPORT GENERATED:\n{summary}")

        return {
            "gradcam_heatmap_path": heatmap_path or "gradcam_not_available",
            "explanation_summary":  summary,
        }
