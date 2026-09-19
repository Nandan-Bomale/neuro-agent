---
title: NeuroAgent
emoji: dY 
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# dY  NeuroAgent
### An Autonomous Multi-Agent AI System for Brain Tumor Diagnosis and Care Planning

> **Status**: dYs  Production Prototype  
> **Team**: Nandan  

---

## dY"O Overview

**NeuroAgent** is a distributed, multi-agent AI pipeline for analyzing Brain MRIs. Instead of a single monolithic model (a "black box"), this system digitally recreates a hospital's **Tumor Board**. It uses a sequence of specialized AI agents working together to detect abnormalities, classify tumor types, localize the mass, triage emergencies, and synthesize a comprehensive medical report. 

This multi-agent approach guarantees high accuracy and complete explainability, fostering clinical trust.

---

## dY?-,? System Architecture & The 9 Agents

The system uses a sequential orchestration pipeline where the output of one agent becomes the input of the next.

| # | Agent | Role | Output / Responsibility |
|---|---|---|---|
| **1** | **Vision Agent** | Pre-processor | Skull-stripping, MRI sequence ID (T1, T2, FLAIR) |
| **2** | **Tumor Classification** | Pathologist | Predicts Tumor Type (Glioma, Meningioma, Pituitary) & WHO Grade |
| **3** | **Localization Agent** | Radiologist | Draws bounding boxes and active contour masks around the tumor |
| **4** | **Emergency Agent** | ER Doctor | Flags immediate emergencies (e.g., Midline Shift, Hydrocephalus) |
| **5** | **Surgical Planning** | Neurosurgeon | Evaluates resectability based on tumor coordinates |
| **6** | **Prognostic Agent** | Researcher | Estimates survival timelines & predicts radiogenomic mutations |
| **7** | **Clinical Trials** | Matchmaker | Matches the patient's profile with active experimental medical trials |
| **8** | **Neuro-Oncologist** | Chief Physician | Synthesizes all data into a holistic treatment strategy |
| **9** | **Explainability** | Reporter | Generates structured text logs and Grad-CAM/Bounding Box overlays |

---

## dY>,? Tech Stack

* **AI Models:** PyTorch, Deep CNN Ensembles (ResNet/DenseNet), YOLOv8 (Localization)
* **Orchestration:** LangGraph state-management
* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** Node.js, React (Vite), ReactFlow, TailwindCSS
* **Computer Vision:** OpenCV, MONAI

---

## dYs? Getting Started

Follow these instructions to run the full NeuroAgent system locally on your machine.

### 1. Prerequisites
- **Python 3.10+** 
- **Node.js** (Required for the React frontend)
- **Git**

### 2. Installation
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/Nandan-Bomale/neuro-agent.git
cd neuro-agent

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend_react
npm install
cd ..
```

### 3. Download AI Models
Since the AI weights are too large for GitHub (>100MB), they are hosted on Hugging Face. Run the provided script to automatically download and place them in the correct folders:

```bash
python download_models.py
```

### 4. Running the Application
We have included a 1-click start script that launches both the FastAPI backend and the React frontend simultaneously:

```bash
# On Windows
start.bat
```
*A browser window will automatically open to `http://localhost:8000` with the unified UI.*

---

## dY O Sharing Online (Cloudflare Tunnel)

If you want to access the software on another device (e.g., your phone) or share it with someone over the internet, we have included a 1-click deployment script.

```bash
# On Windows
share_online.bat
```
This script will safely host the application on port `8005` and automatically generate a **100% free, public HTTPS `.trycloudflare.com` link**. Anyone with the link can use your locally running AI pipeline from anywhere in the world.

---

## dY", License & Disclaimer

This project is an **academic research prototype** and is **not FDA-approved**. It is designed as an assistive "second opinion" tool to demonstrate multi-agent architectures in healthcare, and must not be used as a replacement for human clinical diagnosis. All datasets used (e.g., BraTS, Figshare) are publicly available for academic research.
