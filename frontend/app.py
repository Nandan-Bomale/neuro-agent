"""
frontend/app.py
----------------
Live Execution Dashboard for NeuroAgent.
Streams agent states directly from the backend via SSE and visualises them
agent-by-agent in real-time.
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="NeuroAgent — Live Execution", page_icon="🧠", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
STREAM_URL = f"{BACKEND_URL}/api/analyze/stream"

def render_sidebar():
    st.sidebar.markdown("## 🧠 NeuroAgent Console (2D Pipeline)")
    st.sidebar.markdown("Upload a 2D MRI slice to watch the 8-agent AI pipeline process it live.")
    
    scan_file = st.sidebar.file_uploader("Upload 2D MRI (.jpg, .png, .tif)", type=["jpg", "jpeg", "png", "tif", "tiff"])
    age = st.sidebar.slider("Patient Age", 18, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["M", "F", "Other"])
    
    run_clicked = st.sidebar.button("▶ Run Live Analysis", type="primary", use_container_width=True)
    return scan_file, {"age": age, "sex": sex}, run_clicked

def main():
    st.markdown("<style>.st-emotion-cache-1y4p8pa {padding-top: 1rem;}</style>", unsafe_allow_html=True)
    st.title("🧠 NeuroAgent — Live Step-by-Step Execution")
    
    scan_file, patient_data, run_clicked = render_sidebar()
    
    col_timeline, col_evidence = st.columns([1.2, 1.8], gap="large")
    
    with col_timeline:
        st.markdown("### 📡 Live Execution Timeline")
        timeline_container = st.container()
        
    with col_evidence:
        st.markdown("### 🖼️ Visual Evidence")
        evidence_container = st.container()
        
    if run_clicked and scan_file:
        with timeline_container:
            st.info("Initiating Agent Pipeline...")
            
        try:
            patient_json = json.dumps(patient_data)
            mime_type = scan_file.type or "application/octet-stream"
            
            with requests.post(
                STREAM_URL,
                files={"scan_file": (scan_file.name, scan_file.getvalue(), mime_type)},
                data={"patient_json": patient_json},
                stream=True
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            raw_data = decoded[6:]
                            try:
                                data = json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue
                                
                            # Parse fields from AgentExecutionEvent
                            agent = data.get("agent_name", "unknown")
                            input_sum = data.get("input_summary", "Processing...")
                            update = data.get("output_data", {})
                            img_b64 = data.get("image_b64")
                            is_final = data.get("is_final", False)
                            
                            # Update Timeline
                            with timeline_container:
                                with st.expander(f"✅ {agent.replace('_', ' ').title()}", expanded=True):
                                    st.markdown(f"**{input_sum}**")
                                    st.json(update)
                                    
                            # Update Evidence
                            if img_b64:
                                with evidence_container:
                                    st.image(
                                        f"data:image/png;base64,{img_b64}", 
                                        caption=f"Output from {agent.replace('_', ' ').title()}",
                                        use_container_width=True
                                    )
                                    
            with timeline_container:
                st.success("🎉 Pipeline Execution Complete!")
                                    
        except Exception as e:
            st.error(f"Failed to run pipeline: {e}")

if __name__ == "__main__":
    main()
