"""
NeuroAgent — Verification Agent
================================
Checks the overall confidence score produced by the Report Generation Agent
(which aggregates Vision Agent confidence) against a configurable threshold.

Decision logic:
  confidence >= CONFIDENCE_THRESHOLD  →  flag = "pass"
  confidence <  CONFIDENCE_THRESHOLD  →  flag = "human_review"

Input  (from shared state dict):
  state["vision_findings"]["confidence"]   — float, primary signal
  state["overall_confidence"]              — float, optional override (set by
                                             Report Agent after aggregation)

Output (merged back into shared state dict):
  {
      "verification_status":   "pass" | "human_review",
      "requires_human_review": bool,
      "verification_notes":    str,          # human-readable reason
      "confidence_threshold":  float,        # threshold used (for audit trail)
  }

Environment variable:
  CONFIDENCE_THRESHOLD  — default 0.75 if not set in .env
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

# Load .env from project root (works regardless of where script is run from)
load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLD = 0.75
_PASS = "pass"
_HUMAN_REVIEW = "human_review"


# ── Agent ──────────────────────────────────────────────────────────────────────


class VerificationAgent:
    """
    Stateless verification agent.

    Reads the confidence score from the shared pipeline state and decides
    whether the case can proceed to the Explainability Agent or must be
    flagged for mandatory human review.

    Usage:
        agent = VerificationAgent()
        result = agent.run(state)
        # result is a dict — merge it into the shared state
    """

    def __init__(self, threshold: float | None = None) -> None:
        """
        Args:
            threshold: Override the CONFIDENCE_THRESHOLD from .env.
                       Useful for tests. If None, reads from environment.
        """
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            raw = os.getenv("CONFIDENCE_THRESHOLD", str(_DEFAULT_THRESHOLD))
            # Strip inline comments that may appear in .env (e.g. "0.75  # comment")
            self.threshold = float(raw.split("#")[0].strip())

        logger.info(
            "VerificationAgent initialised | threshold=%.2f", self.threshold
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute verification against the shared pipeline state.

        Args:
            state: Shared NeuroAgent state dict. Must contain at least one of:
                   - state["overall_confidence"]           (set by Report Agent)
                   - state["vision_findings"]["confidence"]  (Vision Agent raw)

        Returns:
            A dict with verification results. The caller (orchestrator node)
            merges this into the full state.

        Raises:
            ValueError: If no confidence score can be found in state.
        """
        confidence = self._extract_confidence(state)

        logger.info(
            "Verification | confidence=%.4f | threshold=%.2f",
            confidence,
            self.threshold,
        )

        if confidence >= self.threshold:
            status = _PASS
            requires_review = False
            notes = (
                f"Confidence {confidence:.4f} meets threshold {self.threshold:.2f}. "
                "Case cleared for Explainability Agent and final output."
            )
        else:
            status = _HUMAN_REVIEW
            requires_review = True
            notes = (
                f"Confidence {confidence:.4f} is below threshold {self.threshold:.2f}. "
                "Case flagged for mandatory human radiologist review. "
                "Explainability step skipped — do not present this output as final."
            )

        logger.warning("Verification result: %s", status) if requires_review else logger.info(
            "Verification result: %s", status
        )

        return {
            "verification_status": status,
            "requires_human_review": requires_review,
            "verification_notes": notes,
            "confidence_threshold": self.threshold,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_confidence(self, state: dict[str, Any]) -> float:
        """
        Extract the best available confidence score from state.

        Priority order:
          1. state["overall_confidence"]  — Report Agent's aggregated score
          2. state["vision_findings"]["confidence"]  — Vision Agent raw score
          3. Raise ValueError if neither is present

        Args:
            state: Shared pipeline state dict.

        Returns:
            Confidence as a float in [0.0, 1.0].

        Raises:
            ValueError: If no usable confidence score is found in state.
        """
        # Priority 1 — Report Agent's aggregated overall confidence
        if "overall_confidence" in state and state["overall_confidence"] is not None:
            value = float(state["overall_confidence"])
            logger.debug("Using overall_confidence from Report Agent: %.4f", value)
            return self._validate_confidence(value, source="overall_confidence")

        # Priority 2 — Vision Agent's raw confidence
        vision = state.get("vision_findings", {})
        if vision and "confidence" in vision and vision["confidence"] is not None:
            value = float(vision["confidence"])
            logger.debug("Using vision_findings.confidence: %.4f", value)
            return self._validate_confidence(value, source="vision_findings.confidence")

        # Nothing usable found
        raise ValueError(
            "VerificationAgent.run() could not find a confidence score in state. "
            "Expected state['overall_confidence'] (float) or "
            "state['vision_findings']['confidence'] (float). "
            f"Got state keys: {list(state.keys())}"
        )

    @staticmethod
    def _validate_confidence(value: float, source: str) -> float:
        """
        Ensure confidence is in the valid [0.0, 1.0] range.

        Args:
            value:  The confidence value to validate.
            source: Human-readable label for error messages.

        Returns:
            The validated value.

        Raises:
            ValueError: If value is outside [0.0, 1.0].
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"Confidence from '{source}' is out of range: {value}. "
                "Expected a float in [0.0, 1.0]."
            )
        return value
