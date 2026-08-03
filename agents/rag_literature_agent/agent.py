"""
agent.py
--------
RAGLiteratureAgent — the clean interface for the Orchestrator.

Input:  A finding string (e.g. "high-grade glioma detected in right temporal lobe")
Output: Top-k relevant PubMed articles with citations, formatted for the Report Agent

Standard .run() output format (same across all NeuroAgent agents):
{
    "agent_name": "rag_literature_agent",
    "success": True,
    "output": {
        "query_used": "...",
        "num_results": 5,
        "results": [ { rank, score, pmid, title, authors, ... } ],
        "citations": [ "[1] Author et al. Title. Journal (Year). PMID: ... URL" ],
        "citation_block": "Full formatted citation block for the report"
    },
    "confidence": 0.91,
    "error": None
}
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agents.rag_literature_agent.pubmed_fetcher import PubMedFetcher, NEUROAGENT_QUERIES
from agents.rag_literature_agent.embedder import PubMedEmbedder
from agents.rag_literature_agent.vector_store import VectorStore, SearchResult

load_dotenv()
logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))


class RAGLiteratureAgent:
    """
    Retrieval-Augmented Generation agent for medical literature.

    Lifecycle:
        1. On first use, loads the pre-built FAISS vector store from disk.
           If none exists, auto-builds one (takes ~5-10 mins first time).
        2. For each query, retrieves top-k relevant PubMed articles.
        3. Returns structured results + formatted citations for the Report Agent.

    Usage:
        agent = RAGLiteratureAgent()
        result = agent.run("high-grade glioma detected in right temporal lobe")
    """

    def __init__(
        self,
        vector_store_path: Path = VECTOR_STORE_PATH,
        top_k: int = DEFAULT_TOP_K,
        auto_build: bool = True,
    ):
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.auto_build = auto_build
        self._store: Optional[VectorStore] = None

    # ── Initialization ─────────────────────────────────────────────────────────

    def _load_or_build_store(self) -> VectorStore:
        """Load vector store from disk, or build it if it doesn't exist."""
        index_path = self.vector_store_path / "index.faiss"

        if index_path.exists():
            logger.info(f"Loading existing vector store from {self.vector_store_path}")
            return VectorStore.load(self.vector_store_path)

        if self.auto_build:
            logger.warning(
                "No vector store found. Building corpus from PubMed now. "
                "This will take ~5-10 minutes on first run..."
            )
            return self._build_store()

        raise FileNotFoundError(
            f"No vector store at {self.vector_store_path}. "
            "Run: python -m agents.rag_literature_agent.build_corpus"
        )

    def _build_store(self) -> VectorStore:
        """Build the vector store from scratch using PubMed."""
        fetcher = PubMedFetcher()
        articles = fetcher.search_bulk(
            queries=NEUROAGENT_QUERIES,
            max_per_query=30,
        )

        embedder = PubMedEmbedder(show_progress=True)
        embeddings = embedder.embed_articles(articles)

        store = VectorStore(embedder=embedder)
        store.build(articles, embeddings)
        store.save(self.vector_store_path)
        return store

    @property
    def store(self) -> VectorStore:
        """Lazy-load the vector store on first access."""
        if self._store is None:
            self._store = self._load_or_build_store()
        return self._store

    def warmup(self) -> None:
        """Pre-load the vector store. Call this at app startup to avoid delays."""
        _ = self.store
        logger.info("RAGLiteratureAgent warmed up and ready.")

    # ── Core Logic ─────────────────────────────────────────────────────────────

    def _build_query(self, finding: str) -> str:
        """
        Convert a Vision Agent finding into a targeted PubMed search query.
        Enriches the finding with MRI-specific search terms.
        """
        base = finding.strip()
        # Append retrieval-boosting terms if not already present
        if "mri" not in base.lower():
            base += " MRI brain"
        if "treatment" not in base.lower() and "diagnosis" not in base.lower():
            base += " diagnosis treatment prognosis"
        return base

    def _compute_confidence(self, results: list[SearchResult]) -> float:
        """
        Estimate retrieval confidence from the top result scores.
        - High confidence: top result has cosine sim > 0.7
        - Medium: 0.5-0.7
        - Low: < 0.5
        """
        if not results:
            return 0.0
        top_score = results[0].score
        # Normalize to [0, 1] — scores are already cosine similarities
        return round(min(float(top_score), 1.0), 4)

    def _format_citation_block(self, results: list[SearchResult]) -> str:
        """Format all citations into a numbered reference block for the report."""
        if not results:
            return "No supporting literature found."
        lines = ["Supporting Literature:"]
        for r in results:
            lines.append(r.to_citation())
        return "\n".join(lines)

    # ── Main Interface ─────────────────────────────────────────────────────────

    def run(self, finding: str, top_k: Optional[int] = None) -> dict:
        """
        Retrieve relevant literature for a given MRI finding.

        Args:
            finding: Natural language finding from the Vision Agent
                     e.g. "high-grade glioma detected in right temporal lobe,
                           confidence 0.89"
            top_k:   Override default number of results

        Returns:
            Standard agent output dict
        """
        k = top_k or self.top_k

        try:
            query = self._build_query(finding)
            logger.info(f"RAG query: '{query}'")

            results = self.store.search(query, top_k=k, min_score=MIN_SCORE)

            if not results:
                logger.warning("No relevant literature found above score threshold.")
                return self._empty_result(finding)

            confidence = self._compute_confidence(results)
            citation_block = self._format_citation_block(results)

            return {
                "agent_name": "rag_literature_agent",
                "success": True,
                "output": {
                    "query_used": query,
                    "num_results": len(results),
                    "results": [r.to_dict() for r in results],
                    "citations": [r.to_citation() for r in results],
                    "citation_block": citation_block,
                },
                "confidence": confidence,
                "error": None,
            }

        except Exception as e:
            logger.error(f"RAGLiteratureAgent failed: {e}", exc_info=True)
            return {
                "agent_name": "rag_literature_agent",
                "success": False,
                "output": {},
                "confidence": 0.0,
                "error": str(e),
            }

    def _empty_result(self, finding: str) -> dict:
        """Return a graceful empty result when no literature is found."""
        return {
            "agent_name": "rag_literature_agent",
            "success": True,
            "output": {
                "query_used": self._build_query(finding),
                "num_results": 0,
                "results": [],
                "citations": [],
                "citation_block": "No supporting literature found for this finding.",
            },
            "confidence": 0.0,
            "error": None,
        }


# ── Convenience function for the Orchestrator ──────────────────────────────────

def get_rag_agent(
    vector_store_path: Path = VECTOR_STORE_PATH,
    top_k: int = DEFAULT_TOP_K,
) -> RAGLiteratureAgent:
    """Factory function — returns a warmed-up RAGLiteratureAgent."""
    agent = RAGLiteratureAgent(vector_store_path=vector_store_path, top_k=top_k)
    agent.warmup()
    return agent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agent = RAGLiteratureAgent(auto_build=True)

    test_findings = [
        "high-grade glioma detected in right temporal lobe, confidence 0.89",
        "meningioma suspected in left frontal region, confidence 0.74",
        "no significant abnormality detected, confidence 0.92",
    ]

    for finding in test_findings:
        print(f"\n{'='*70}")
        print(f"Finding: {finding}")
        result = agent.run(finding)
        print(f"Success: {result['success']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Results: {result['output'].get('num_results', 0)}")
        if result["output"].get("results"):
            top = result["output"]["results"][0]
            print(f"Top result: {top['title'][:70]}... (score={top['score']})")
        print(f"\nCitation block:\n{result['output'].get('citation_block', '')}")
