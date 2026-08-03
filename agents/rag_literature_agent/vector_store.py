"""
vector_store.py
---------------
Builds, saves, loads, and searches a FAISS vector store of PubMed articles.

FAISS (Facebook AI Similarity Search) runs entirely on CPU.
- IndexFlatIP: exact inner-product search (cosine sim since embeddings are L2-normalized)
- Fast enough for our corpus size (~500-1000 articles)

Usage:
    store = VectorStore()
    store.build(articles, embeddings)
    store.save("data/vector_store")

    # Later:
    store = VectorStore.load("data/vector_store")
    results = store.search("glioblastoma treatment options", top_k=5)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Union

import faiss
import numpy as np

from agents.rag_literature_agent.pubmed_fetcher import PubMedArticle
from agents.rag_literature_agent.embedder import PubMedEmbedder

logger = logging.getLogger(__name__)


class SearchResult:
    """A single search result with article metadata and similarity score."""

    def __init__(self, article: PubMedArticle, score: float, rank: int):
        self.article = article
        self.score = score      # cosine similarity (0 to 1, higher = more relevant)
        self.rank = rank

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "pmid": self.article.pmid,
            "title": self.article.title,
            "authors": self.article.authors[:3],
            "journal": self.article.journal,
            "year": self.article.year,
            "abstract_excerpt": self.article.abstract[:300] + "..."
            if len(self.article.abstract) > 300
            else self.article.abstract,
            "url": self.article.url,
        }

    def to_citation(self) -> str:
        """Format as a citation string for the radiology report."""
        authors = ", ".join(self.article.authors[:3])
        if len(self.article.authors) > 3:
            authors += " et al."
        return (
            f"[{self.rank}] {authors}. \"{self.article.title}\". "
            f"{self.article.journal} ({self.article.year}). "
            f"PMID: {self.article.pmid}. {self.article.url}"
        )


class VectorStore:
    """
    FAISS-backed vector store for PubMed article retrieval.

    Persistence layout (saved as a directory):
        vector_store/
            index.faiss     — FAISS index binary
            articles.pkl    — list of PubMedArticle objects
            metadata.json   — build info (model, num articles, dim)
    """

    INDEX_FILE = "index.faiss"
    ARTICLES_FILE = "articles.pkl"
    METADATA_FILE = "metadata.json"

    def __init__(self, embedder: PubMedEmbedder | None = None):
        self.embedder = embedder or PubMedEmbedder()
        self._index: faiss.Index | None = None
        self._articles: list[PubMedArticle] = []
        self._metadata: dict = {}

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(
        self,
        articles: list[PubMedArticle],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """
        Build the FAISS index from a list of PubMedArticles.

        Args:
            articles:   List of PubMedArticle objects
            embeddings: Pre-computed embeddings (optional). If None, computes them.
        """
        if not articles:
            raise ValueError("Cannot build vector store with empty article list.")

        if embeddings is None:
            logger.info("Computing embeddings for articles...")
            embeddings = self.embedder.embed_articles(articles)

        assert len(articles) == len(embeddings), (
            f"Mismatch: {len(articles)} articles but {len(embeddings)} embeddings"
        )

        dim = embeddings.shape[1]
        logger.info(
            f"Building FAISS IndexFlatIP with {len(articles)} vectors, dim={dim}"
        )

        # IndexFlatIP = exact inner product search
        # Since embeddings are L2-normalized, this equals cosine similarity
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._articles = articles
        self._metadata = {
            "num_articles": len(articles),
            "embedding_dim": dim,
            "embedding_model": self.embedder.model_name,
        }

        logger.info(f"Vector store built successfully. Total vectors: {self._index.ntotal}")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search the vector store for articles most relevant to a query.

        Args:
            query:     Natural language query string
            top_k:     Number of top results to return
            min_score: Minimum cosine similarity threshold (0.0 to 1.0)

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        if self._index is None:
            raise RuntimeError("Vector store is empty. Call build() or load() first.")

        query_vec = self.embedder.embed_query(query)   # shape: (1, dim)

        scores, indices = self._index.search(query_vec, top_k)
        scores = scores[0]      # flatten
        indices = indices[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx == -1:       # FAISS returns -1 for empty slots
                continue
            if score < min_score:
                continue
            results.append(SearchResult(
                article=self._articles[idx],
                score=float(score),
                rank=rank,
            ))

        logger.info(
            f"Query: '{query[:60]}...' → {len(results)} results "
            f"(top score: {scores[0]:.4f})"
        )
        return results

    def search_batch(
        self, queries: list[str], top_k: int = 3
    ) -> dict[str, list[SearchResult]]:
        """Search multiple queries at once. Returns dict of query → results."""
        return {q: self.search(q, top_k=top_k) for q in queries}

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """Save the vector store to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(directory / self.INDEX_FILE))

        # Save articles
        with open(directory / self.ARTICLES_FILE, "wb") as f:
            pickle.dump(self._articles, f)

        # Save metadata
        with open(directory / self.METADATA_FILE, "w") as f:
            json.dump(self._metadata, f, indent=2)

        logger.info(f"Vector store saved to {directory}")

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        embedder: PubMedEmbedder | None = None,
    ) -> "VectorStore":
        """Load a previously saved vector store from disk."""
        directory = Path(directory)

        if not (directory / cls.INDEX_FILE).exists():
            raise FileNotFoundError(
                f"No FAISS index found at {directory / cls.INDEX_FILE}. "
                "Run build_corpus.py first."
            )

        store = cls(embedder=embedder)

        # Load FAISS index
        store._index = faiss.read_index(str(directory / cls.INDEX_FILE))

        # Load articles
        with open(directory / cls.ARTICLES_FILE, "rb") as f:
            store._articles = pickle.load(f)

        # Load metadata
        with open(directory / cls.METADATA_FILE) as f:
            store._metadata = json.load(f)

        logger.info(
            f"Loaded vector store from {directory}. "
            f"Articles: {store._metadata.get('num_articles', '?')}, "
            f"Dim: {store._metadata.get('embedding_dim', '?')}"
        )
        return store

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._articles) > 0

    @property
    def num_articles(self) -> int:
        return len(self._articles)

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.rag_literature_agent.pubmed_fetcher import PubMedFetcher

    # Build a small test corpus
    fetcher = PubMedFetcher()
    articles = fetcher.search("brain tumor MRI segmentation deep learning", max_results=10)

    embedder = PubMedEmbedder()
    embeddings = embedder.embed_articles(articles)

    store = VectorStore(embedder=embedder)
    store.build(articles, embeddings)

    # Test search
    results = store.search("glioblastoma treatment prognosis", top_k=3)
    for r in results:
        print(f"\n{'='*60}")
        print(f"Rank {r.rank} | Score: {r.score:.4f}")
        print(f"Title: {r.article.title}")
        print(f"Citation: {r.to_citation()}")

    # Save and reload
    store.save("data/vector_store/test")
    reloaded = VectorStore.load("data/vector_store/test")
    print(f"\nReloaded store — articles: {reloaded.num_articles}")
