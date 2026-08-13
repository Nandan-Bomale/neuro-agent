"""
NeuroAgent — LangGraph Graph Definition
=========================================
Defines the full multi-agent pipeline as a compiled LangGraph StateGraph.

Graph topology:

   ┌─────────────────────────────────────────────────────────────────────┐
  │                          START                                      │
  │                            │                                        │
  │                            ▼                                        │
  │                         vision          ← tumour detection first    │
  │                            │                                        │
  │                            ▼                                        │
  │                   tumor_classification  ← type + grade (NEW)        │
  │                            │                                        │
  │                            ▼                                        │
  │                         clinical        ← needs vision+type output  │
  │                            │                                        │
  │                            ▼                                        │
  │                           rag           ← literature retrieval      │
  │                            │                                        │
  │                            ▼                                        │
  │                         report          ← combines all              │
  │                            │                                        │
  │                       verification                                   │
  │                    ┌───────┴───────┐                                │
  │                    ▼               ▼                                │
  │             explainability    human_review   ← conditional edge     │
  │                    ▼               ▼                                │
  │                   END             END                               │
  └─────────────────────────────────────────────────────────────────────┘

Execution order rationale:
  vision runs first — its confidence score and detected regions are required
  by both clinical (to assess history consistency) and rag (to query the right
  literature). clinical runs before rag so its risk-factor analysis can also
  inform the literature query. report combines all three, then verification
  gates the final output.

Usage:
    from orchestrator.graph import compile_graph, run_pipeline

    # Option 1 — low-level (full control)
    graph = compile_graph()
    result = graph.invoke(initial_state)

    # Option 2 — convenience wrapper
    result = run_pipeline(
        mri_scan_path="data/raw/patient_001.nii.gz",
        patient_data={"age": 45, "symptoms": ["headache"]},
    )
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from orchestrator.nodes import (
    clinical_node,
    explainability_node,
    human_review_node,
    rag_node,
    report_node,
    tumor_classification_node,
    verification_node,
    vision_node,
)
from orchestrator.router import (
    ROUTE_EXPLAINABILITY,
    ROUTE_HUMAN_REVIEW,
    route_after_verification,
)
from orchestrator.state import NeuroAgentState, create_initial_state

logger = logging.getLogger(__name__)

# ── Node name constants ────────────────────────────────────────────────────────
# Define node names once — used in both add_node() and add_edge() calls.
# If you rename a node, change it here and nowhere else.

_NODE_VISION = "vision"
_NODE_TUMOR_CLASSIFICATION = "tumor_classification"
_NODE_CLINICAL = "clinical"
_NODE_RAG = "rag"
_NODE_REPORT = "report"
_NODE_VERIFICATION = "verification"
_NODE_EXPLAINABILITY = ROUTE_EXPLAINABILITY    # "explainability"
_NODE_HUMAN_REVIEW = ROUTE_HUMAN_REVIEW        # "human_review"


# ── Graph builder ──────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph (not yet compiled).

    Useful for inspection, testing, or adding checkpointers before compiling.

    Returns:
        An uncompiled StateGraph instance with all nodes and edges registered.
    """
    builder = StateGraph(NeuroAgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    builder.add_node(_NODE_VISION, vision_node)
    builder.add_node(_NODE_TUMOR_CLASSIFICATION, tumor_classification_node)
    builder.add_node(_NODE_CLINICAL, clinical_node)
    builder.add_node(_NODE_RAG, rag_node)
    builder.add_node(_NODE_REPORT, report_node)
    builder.add_node(_NODE_VERIFICATION, verification_node)
    builder.add_node(_NODE_EXPLAINABILITY, explainability_node)
    builder.add_node(_NODE_HUMAN_REVIEW, human_review_node)

    # ── Edges: strictly sequential pipeline ─────────────────────────────────────────
    # vision first, then tumor classification (type+grade), then clinical
    # (which now knows BOTH what was found AND what type it is).
    builder.add_edge(START, _NODE_VISION)
    builder.add_edge(_NODE_VISION, _NODE_TUMOR_CLASSIFICATION)
    builder.add_edge(_NODE_TUMOR_CLASSIFICATION, _NODE_CLINICAL)
    builder.add_edge(_NODE_CLINICAL, _NODE_RAG)
    builder.add_edge(_NODE_RAG, _NODE_REPORT)

    # ── Edges: sequential pipeline after report ────────────────────────────────
    builder.add_edge(_NODE_REPORT, _NODE_VERIFICATION)

    # ── Conditional edge: verification → explainability | human_review ─────────
    builder.add_conditional_edges(
        _NODE_VERIFICATION,
        route_after_verification,          # reads state["verification_status"]
        {
            ROUTE_EXPLAINABILITY: _NODE_EXPLAINABILITY,
            ROUTE_HUMAN_REVIEW: _NODE_HUMAN_REVIEW,
        },
    )

    # ── Terminal edges ─────────────────────────────────────────────────────────
    builder.add_edge(_NODE_EXPLAINABILITY, END)   # pass path
    builder.add_edge(_NODE_HUMAN_REVIEW, END)     # human review path

    logger.debug("NeuroAgent graph built | nodes=%d", len(builder.nodes))
    return builder


def compile_graph():
    """
    Build and compile the LangGraph graph into an executable runnable.

    Returns:
        A compiled LangGraph CompiledGraph ready for .invoke() or .ainvoke().

    Example:
        graph = compile_graph()

        # Synchronous — standard usage
        result = graph.invoke(initial_state)

        # Asynchronous — same sequential order, non-blocking I/O
        import asyncio
        result = asyncio.run(graph.ainvoke(initial_state))
    """
    graph = build_graph().compile()
    logger.info("NeuroAgent pipeline compiled and ready.")
    return graph


# ── Convenience entry point ────────────────────────────────────────────────────


def run_pipeline(
    mri_scan_path: str,
    patient_data: dict[str, Any],
    run_id: str | None = None,
    async_mode: bool = False,
) -> NeuroAgentState:
    """
    End-to-end pipeline runner. Builds state, compiles graph, runs, returns result.

    This is the single function the FastAPI backend calls. It handles all
    LangGraph setup internally — callers don't need to know about StateGraph.

    Args:
        mri_scan_path: Path to the NIfTI MRI scan file.
        patient_data:  Dict of patient metadata (age, symptoms, history, etc.).
        run_id:        Optional run ID for tracing. Auto-generated if omitted.
        async_mode:    If True, use graph.ainvoke() for true parallel execution
                       of vision/clinical/rag nodes. Requires asyncio event loop.
                       If False (default), use synchronous graph.invoke().

    Returns:
        Final NeuroAgentState dict after all nodes have run.
        Key result fields:
          - pipeline_status: "complete" | "human_review_required" | "error"
          - report: structured radiology report (if complete)
          - gradcam_heatmap_path: Grad-CAM image path (if complete)
          - human_review_reason: why flagged (if human_review_required)

    Raises:
        RuntimeError: If any agent is not yet implemented (helpful during dev).
        ValueError:   If state is malformed (propagated from agents).
    """
    _run_id = run_id or str(uuid.uuid4())
    logger.info(
        "run_pipeline() start | run_id=%s | scan=%s",
        _run_id,
        mri_scan_path,
    )

    initial_state = create_initial_state(
        mri_scan_path=mri_scan_path,
        patient_data=patient_data,
        run_id=_run_id,
    )

    graph = compile_graph()

    if async_mode:
        import asyncio
        result = asyncio.run(graph.ainvoke(initial_state))
    else:
        result = graph.invoke(initial_state)

    logger.info(
        "run_pipeline() complete | run_id=%s | status=%s",
        _run_id,
        result.get("pipeline_status"),
    )
    return result
