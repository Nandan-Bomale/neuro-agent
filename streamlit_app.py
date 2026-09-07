"""
streamlit_app.py
----------------
Turnkey 24/7 Independent Cloud Deployment for Streamlit Community Cloud (share.streamlit.io).
Runs the full NeuroAgent multi-agent diagnostic pipeline in the cloud for $0 without credit card.
"""

import os
import sys
from pathlib import Path
import json
import asyncio
import tempfile
import time

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from PIL import Image

# Page Config
st.set_page_config(
    page_title="NeuroAgent ? Clinical Brain MRI Diagnostic Suite",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        margin-bottom: 1rem;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def ensure_model_weights():
    repo_id = "cloudxvi/neuro-agent-models"
    models_to_check = [
        "models/tumor_classifier/type_ensemble_best.pth",
        "models/tumor_classifier/type_ensemble_classes.json",
        "models/tumor_classifier/grade_classifier_best.pth",
        "models/tumor_classifier/grade_classifier_classes.json",
        "models/yolo/weights/detect_best.pt",
        "models/yolo/weights/best.pt",
        "models/bbox_regressor/bbox_model_best.pth",
        "models/bbox_regressor/labels.json",
        "models/radiogenomics/radiogenomics_best.pth"
    ]
    try:
        from huggingface_hub import hf_hub_download
        for rel in models_to_check:
            target = Path(rel)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(repo_id=repo_id, filename=rel, local_dir=".")
    except Exception as exc:
        print(f"Notice: Model check encountered: {exc}")

@st.cache_resource
def load_orchestrator():
    ensure_model_weights()
    from backend.pipeline import OrchestratorPipeline
    pipeline = OrchestratorPipeline()
    pipeline._ensure_graph_loaded()
    return pipeline

def main():
    st.markdown('<div class="main-header">?? NeuroAgent ? Multi-Agent Brain MRI Diagnostic Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous Multi-Agent AI System for Radiological Segmentation, Histological Grading, and Neuro-Oncological Care Planning</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("?? Patient Information")
        patient_name = st.text_input("Patient Identifier", value="PT-2026-0042")
        patient_age = st.slider("Patient Age", min_value=1, max_value=100, value=52)
        patient_sex = st.selectbox("Sex", options=["M", "F", "Other"])
        symptoms = st.multiselect(
            "Presenting Symptoms",
            options=["Headache", "Seizures", "Cognitive Decline", "Visual Disturbances", "Motor Weakness", "Nausea/Vomiting"],
            default=["Headache", "Visual Disturbances"]
        )
        
        st.divider()
        st.header("?? MRI Scan Upload")
        uploaded_file = st.file_uploader(
            "Upload Brain MRI (.jpg, .png, .jpeg)",
            type=["jpg", "png", "jpeg"]
        )

        run_btn = st.button("?? Run Comprehensive Diagnostic Analysis", type="primary", use_container_width=True)

    if not uploaded_file:
        st.info("?? Please upload a brain MRI slice and configure patient details in the sidebar to begin.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### ?? Multi-Agent Vision")
            st.markdown("Automated skull-stripping, anatomical active contouring, and false-positive elimination for non-tumor cases.")
        with col2:
            st.markdown("### ?? Histological Grading")
            st.markdown("Dual-network ensemble classifying Glioma, Meningioma, Pituitary Adenoma, and Normal tissue with WHO Grade.")
        with col3:
            st.markdown("### ?? Care Plan Synthesis")
            st.markdown("Integrated surgical resectability index, midline shift triage, survival prognosis, and clinical trial matching.")
        return

    # Process Scan
    if uploaded_file and run_btn:
        with st.spinner("Initializing Deep Multi-Agent Diagnostic Pipeline..."):
            pipeline = load_orchestrator()

        # Save upload to temp file
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        patient_dict = {
            "age": patient_age,
            "sex": patient_sex,
            "symptoms": symptoms,
            "patient_name": patient_name
        }

        progress_bar = st.progress(0, text="Starting Multi-Agent Execution...")
        
        col_img, col_results = st.columns([1.1, 1.9], gap="large")

        with col_img:
            st.subheader("??? MRI & Radiological Evidence")
            orig_img = Image.open(uploaded_file)
            st.image(orig_img, caption=f"Original Upload: {uploaded_file.name}", use_container_width=True)
            image_placeholder = st.empty()

        with col_results:
            st.subheader("?? Diagnostic Summary & Agent Reasoning")
            status_container = st.container()

        # Run pipeline
        t0 = time.time()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_events():
                events = []
                async for evt in pipeline.stream_run(scan_path=tmp_path, patient_data=patient_dict):
                    events.append(evt)
                return events

            events = loop.run_until_complete(run_events())
            loop.close()

            progress_bar.progress(100, text="Diagnostic Analysis Complete!")
            elapsed = time.time() - t0

            # Parse final state
            final_res = None
            evidence_img_b64 = None

            for evt_raw in events:
                if evt_raw.startswith("data:"):
                    try:
                        evt_data = json.loads(evt_raw[5:].strip())
                        if evt_data.get("image_b64"):
                            evidence_img_b64 = evt_data["image_b64"]
                        if evt_data.get("is_final") or evt_data.get("output_data", {}).get("findings"):
                            final_res = evt_data.get("output_data")
                    except Exception:
                        pass

            if evidence_img_b64:
                import base64
                from io import BytesIO
                dec = base64.b64decode(evidence_img_b64)
                overlay_img = Image.open(BytesIO(dec))
                image_placeholder.image(overlay_img, caption="Identified Tumor Contour & Bounding Box", use_container_width=True)

            with status_container:
                st.success(f"Analysis completed in {elapsed:.2f} seconds.")
                
                # Check tumor classification & vision
                if final_res:
                    st.json(final_res)
                else:
                    st.info("Pipeline completed successfully. Review agent logs below.")

        except Exception as e:
            st.error(f"Error during analysis: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
