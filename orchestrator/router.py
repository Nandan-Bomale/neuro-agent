"""
NeuroAgent — Verification Router
==================================
Conditional routing logic for the LangGraph graph.

This module contains the function used as the condition in
graph.add_conditional_edges("verification", route_after_verification, ...).

The router reads verification_status from state and returns the name of
the next node to execute. It contains zero ML logic and zero side effects —
it only reads from state and returns a string.

Decision table:
  verification_status == "pass"          →  ROUTE_EXPLAINABILITY
  verification_status == "human_review"  →  ROUTE_HUMAN_REVIEW
  verification_status missing/unknown    →  ROUTE_HUMAN_REVIEW  (fail-safe)
"""

from __future__ import annotations

import logging
from typing import Final

from orchestrator.state import NeuroAgentState

logger = logging.getLogger(__name__)

# ── Route name constants ───────────────────────────────────────────────────────
# These strings are the node names registered in graph.py.
# If you rename a node in graph.py, update these constants too.

ROUTE_EXPLAINABILITY: Final[str] = "explainability"
ROUTE_HUMAN_REVIEW: Final[str] = "human_review"

# ── Router function ────────────────────────────────────────────────────────────


def route_after_verification(state: NeuroAgentState) -> str:
    """
    Conditional edge function: decides what happens after verification_node.

    Called by LangGraph's conditional edge mechanism. The return value must
    match a key in the routing map passed to add_conditional_edges().

    Routing logic:
      "pass"         → explainability_node  (high-confidence path)
      "human_review" → human_review_node    (low-confidence / flag path)
      anything else  → human_review_node    (fail-safe — never auto-approve)

    Args:
        state: Current shared pipeline state. Reads state["verification_status"].

    Returns:
        Node name string: ROUTE_EXPLAINABILITY or ROUTE_HUMAN_REVIEW.
    """
    status = state.get("verification_status")

    if status == "pass":
        logger.info(
            "[router] Routing to EXPLAINABILITY | confidence cleared threshold"
        )
        return ROUTE_EXPLAINABILITY

    # Covers "human_review", None, or any unexpected value — fail-safe
    if status != "human_review":
        logger.error(
            "[router] Unexpected verification_status=%r — defaulting to human_review "
            "(fail-safe: never auto-approve an unknown state)",
            status,
        )
    else:
        logger.warning(
            "[router] Routing to HUMAN REVIEW | confidence below threshold"
        )

    return ROUTE_HUMAN_REVIEW
