"""
NeuroAgent Orchestrator
========================
Public API for the LangGraph-based multi-agent pipeline.

Typical usage (from backend/main.py or tests):

    from orchestrator import run_pipeline, NeuroAgentState

    result: NeuroAgentState = run_pipeline(
        mri_scan_path="data/raw/patient_001.nii.gz",
        patient_data={"age": 45, "symptoms": ["headache"], "history": "..."},
    )

    if result["pipeline_status"] == "complete":
        print(result["report"])
    elif result["pipeline_status"] == "human_review_required":
        print("Flagged:", result["human_review_reason"])
"""

# State and router are always safe to import (pure Python, no ML deps)
from orchestrator.router import (
    ROUTE_EXPLAINABILITY,
    ROUTE_HUMAN_REVIEW,
    route_after_verification,
)
from orchestrator.state import NeuroAgentState, create_initial_state

__all__ = [
    # Primary entry point (requires langgraph — imported lazily below)
    "run_pipeline",
    # Graph building (requires langgraph — imported lazily below)
    "build_graph",
    "compile_graph",
    # State
    "NeuroAgentState",
    "create_initial_state",
    # Router
    "route_after_verification",
    "ROUTE_EXPLAINABILITY",
    "ROUTE_HUMAN_REVIEW",
]


def __getattr__(name: str):
    """
    Lazy import for graph functions.
    Defers 'import langgraph' until first use so the orchestrator package
    is importable for state/router/nodes work even before langgraph is installed.
    """
    if name in ("run_pipeline", "build_graph", "compile_graph"):
        from orchestrator.graph import build_graph, compile_graph, run_pipeline  # noqa: PLC0415

        _map = {
            "run_pipeline": run_pipeline,
            "build_graph": build_graph,
            "compile_graph": compile_graph,
        }
        return _map[name]
    raise AttributeError(f"module 'orchestrator' has no attribute {name!r}")
