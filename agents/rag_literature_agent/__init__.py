"""
RAG Literature Agent package.
Retrieves relevant PubMed literature for a given MRI finding.
"""

from agents.rag_literature_agent.agent import RAGLiteratureAgent, get_rag_agent
from agents.rag_literature_agent.pubmed_fetcher import PubMedFetcher, PubMedArticle
from agents.rag_literature_agent.embedder import PubMedEmbedder
from agents.rag_literature_agent.vector_store import VectorStore, SearchResult

__all__ = [
    "RAGLiteratureAgent",
    "get_rag_agent",
    "PubMedFetcher",
    "PubMedArticle",
    "PubMedEmbedder",
    "VectorStore",
    "SearchResult",
]
