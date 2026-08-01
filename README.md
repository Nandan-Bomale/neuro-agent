# 🧠 NeuroAgent
### A Multi-Agent AI System for Brain MRI Diagnosis Support

> **Status**: 🚧 In Development  
> **Team**: Nandan | Guided by Prof. Prashant N  
> **Timeline**: July – October 2026

---

## 📌 Overview

NeuroAgent is a multi-agent AI pipeline for brain MRI analysis. Instead of a single classifier, it builds a team of specialized AI agents that collaborate — the way a real clinical team would — to detect abnormalities, reason over patient history, retrieve medical literature, generate structured reports, and verify confidence before results reach a doctor.

---

## 🏗️ System Architecture

```
Input (MRI Scan + Patient Data)
        │
        ▼
  ┌─────────────┐
  │ Orchestrator │  ← LangGraph
  └──────┬──────┘
         │
   ┌─────┼──────────────────────────────┐
   ▼     ▼                              ▼
Vision  Clinical History            RAG Literature
Agent   Agent                       Agent
   │     │                              │
   └─────┴──────────────────────────────┘
                     │
                     ▼
            Report Generation Agent
                     │
                     ▼
            Verification Agent
                     │
              ┌──────┴──────┐
              ▼             ▼
        ✅ Output     ⚠️ Human Review Flag
              │
              ▼
        Explainability Agent (Grad-CAM)
```

---

## 🤖 Agents

| Agent | Responsibility | Key Tech |
|---|---|---|
| **Vision Agent** | Tumour detection & segmentation | PyTorch, MONAI, U-Net |
| **Clinical History Agent** | Reasoning over patient metadata | LLM + structured prompts |
| **RAG Literature Agent** | Evidence retrieval from PubMed | FAISS/Chroma, PubMedBERT |
| **Report Generation Agent** | Structured radiology report writing | Llama-3.2-3B / Phi-3-mini + LoRA |
| **Verification Agent** | Confidence checking & human review flags | Rule-based + LLM |
| **Explainability Agent** | Grad-CAM heatmap generation | PyTorch hooks |
| **Orchestrator** | Agent coordination & routing | LangGraph |

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Vision Model | PyTorch, MONAI, U-Net |
| Augmentation | Classical (rotation, flip, elastic) + GAN (enhancement) |
| Language Model | Llama-3.2-3B or Phi-3-mini + LoRA (4-bit) |
| RAG / Retrieval | FAISS / Chroma + PubMedBERT embeddings |
| Orchestration | LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker |

---

## 📁 Project Structure

```
neuro-agent/
├── agents/
│   ├── vision_agent/
│   ├── clinical_history_agent/
│   ├── rag_literature_agent/
│   ├── report_generation_agent/
│   ├── verification_agent/
│   └── explainability_agent/
├── orchestrator/
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_store/
├── models/
│   ├── vision/
│   └── llm/
├── frontend/
├── backend/
├── notebooks/
├── tests/
├── docker/
├── docs/
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 📅 Timeline

| Weeks | Phase | Deliverable |
|---|---|---|
| 1–3 | Vision Agent | Tumour segmentation model + Grad-CAM, tested on BraTS |
| 4–5 | Augmentation | Classical pipeline (baseline); GAN attempted in parallel |
| 6–7 | RAG Pipeline | Vector DB from PubMed; literature agent with citations |
| 8–9 | Report Agent | LoRA fine-tuned LLM producing structured reports |
| 10–11 | Orchestration | All agents connected via LangGraph with verification loop |
| 12 | Polish & Demo | End-to-end testing, Streamlit UI, final report, demo |

---

## 📊 Datasets

| Dataset | Purpose |
|---|---|
| BraTS (Brain Tumour Segmentation) | Vision agent training & evaluation |
| PubMed abstracts (NCBI E-utilities) | RAG literature corpus |
| Synthetic clinical profiles | Patient metadata paired with BraTS scans |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or Google Colab
- Docker (optional, for full deployment)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/neuro-agent.git
cd neuro-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Demo

```bash
# Start the backend
cd backend
uvicorn main:app --reload

# Start the frontend (in a new terminal)
cd frontend
streamlit run app.py
```

---

## 🎯 Expected Outcomes

- ✅ End-to-end system: upload MRI scan → receive structured, evidence-cited report with confidence flag
- ✅ Trained vision model with Dice score reported on BraTS held-out test set
- ✅ Fine-tuned report generation model with sample outputs
- ✅ Working RAG pipeline with real, cited PubMed literature
- ✅ Demo video + live walkthrough for evaluation/defence
- ✅ Written project report

---

## 👥 Team

| Name | Role |
|---|---|
| Nandan | Lead Developer |
| [Colleague Name] | Co-Developer |
| Prof. Prashant N | Project Guide |

---

## 📄 License

This project is for academic purposes. All datasets used are publicly available.
