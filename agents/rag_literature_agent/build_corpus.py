"""
build_corpus.py
---------------
One-time script to build the full PubMed literature corpus for NeuroAgent.

Run this ONCE to:
1. Fetch ~300-500 brain tumor / MRI papers from PubMed
2. Generate PubMedBERT embeddings for each
3. Build and save the FAISS vector index to disk

Usage:
    python -m agents.rag_literature_agent.build_corpus

The saved vector store is loaded at runtime by RAGLiteratureAgent.
Typical runtime: ~5-10 minutes (rate-limited by NCBI API).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from agents.rag_literature_agent.pubmed_fetcher import PubMedFetcher, NEUROAGENT_QUERIES
from agents.rag_literature_agent.embedder import PubMedEmbedder
from agents.rag_literature_agent.vector_store import VectorStore

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = Path(
    os.getenv("VECTOR_STORE_PATH", "data/vector_store")
)
MAX_PER_QUERY = int(os.getenv("MAX_PER_QUERY", "40"))


def main():
    logger.info("=" * 60)
    logger.info("NeuroAgent — Building RAG Literature Corpus")
    logger.info("=" * 60)
    logger.info(f"Queries: {len(NEUROAGENT_QUERIES)}")
    logger.info(f"Max results per query: {MAX_PER_QUERY}")
    logger.info(f"Save path: {VECTOR_STORE_PATH}")

    # Step 1: Fetch articles from PubMed
    logger.info("\n[Step 1/3] Fetching PubMed abstracts...")
    fetcher = PubMedFetcher()
    articles = fetcher.search_bulk(
        queries=NEUROAGENT_QUERIES,
        max_per_query=MAX_PER_QUERY,
    )
    logger.info(f"Total unique articles fetched: {len(articles)}")

    if len(articles) == 0:
        logger.error("No articles fetched. Check your NCBI_API_KEY and internet connection.")
        return

    # Step 2: Generate embeddings
    logger.info("\n[Step 2/3] Generating PubMedBERT embeddings...")
    embedder = PubMedEmbedder(show_progress=True)
    embeddings = embedder.embed_articles(articles)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Step 3: Build and save FAISS index
    logger.info("\n[Step 3/3] Building and saving FAISS vector store...")
    store = VectorStore(embedder=embedder)
    store.build(articles, embeddings)
    store.save(VECTOR_STORE_PATH)

    logger.info("\n" + "=" * 60)
    logger.info("Corpus build complete!")
    logger.info(f"  Articles indexed : {store.num_articles}")
    logger.info(f"  Embedding dim    : {store.metadata.get('embedding_dim')}")
    logger.info(f"  Saved to         : {VECTOR_STORE_PATH}")
    logger.info("=" * 60)

    # Quick sanity check search
    logger.info("\nSanity check — searching for 'glioblastoma MRI treatment'...")
    results = store.search("glioblastoma MRI treatment", top_k=3)
    for r in results:
        logger.info(f"  [{r.rank}] (score={r.score:.3f}) {r.article.title[:70]}...")


if __name__ == "__main__":
    main()
