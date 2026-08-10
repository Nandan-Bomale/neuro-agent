"""
embedder.py
-----------
Generates dense vector embeddings for PubMed articles using PubMedBERT.

Model: pritamdeka/S-PubMedBert-MS-MARCO
- Fine-tuned on MS-MARCO for semantic similarity / retrieval tasks
- Understands medical terminology far better than general-purpose models
- Runs entirely on CPU (~400MB RAM) — no GPU needed

Usage:
    embedder = PubMedEmbedder()
    vectors = embedder.embed_articles(articles)
"""

import os
import logging
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from agents.rag_literature_agent.pubmed_fetcher import PubMedArticle

logger = logging.getLogger(__name__)

# Model name — configurable via .env
DEFAULT_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "pritamdeka/S-PubMedBert-MS-MARCO"
)


class PubMedEmbedder:
    """
    Wraps a SentenceTransformer model to embed PubMed articles.

    - Runs on CPU by default (medical literature retrieval doesn't need GPU)
    - Batches articles for efficiency
    - Caches the loaded model across calls
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device="cpu")
            logger.info(
                f"Embedding model loaded. "
                f"Dimension: {self._model.get_sentence_embedding_dimension()}"
            )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension for this model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw text strings.

        Args:
            texts: List of text strings to embed

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
        )

        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def embed_articles(self, articles: list[PubMedArticle]) -> np.ndarray:
        """
        Embed a list of PubMedArticle objects.
        Uses article.to_text() which includes title + authors + journal + abstract.

        Args:
            articles: List of PubMedArticle objects

        Returns:
            np.ndarray of shape (len(articles), embedding_dim), dtype float32
        """
        texts = [article.to_text() for article in articles]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query for retrieval.

        Args:
            query: Natural language query string

        Returns:
            np.ndarray of shape (1, embedding_dim), dtype float32
        """
        embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        save_path: Union[str, Path],
    ) -> None:
        """Save embeddings to a .npy file for reuse."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embeddings)
        logger.info(f"Saved {len(embeddings)} embeddings to {save_path}")

    def load_embeddings(self, load_path: Union[str, Path]) -> np.ndarray:
        """Load previously saved embeddings from a .npy file."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {load_path}")
        embeddings = np.load(load_path)
        logger.info(f"Loaded embeddings: shape={embeddings.shape}")
        return embeddings.astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    embedder = PubMedEmbedder()

    # Quick test with dummy texts
    sample_texts = [
        "Glioblastoma multiforme is the most aggressive primary brain tumor.",
        "U-Net architecture for biomedical image segmentation achieves high Dice scores.",
        "MRI provides excellent soft tissue contrast for brain tumor detection.",
    ]

    embeddings = embedder.embed_texts(sample_texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding dim: {embedder.embedding_dim}")

    # Test query embedding
    query_vec = embedder.embed_query("brain tumor segmentation deep learning")
    print(f"Query embedding shape: {query_vec.shape}")

    # Cosine similarity (embeddings are L2-normalized, so dot product = cosine sim)
    sims = embeddings @ query_vec.T
    print(f"\nSimilarity scores: {sims.flatten()}")
    print(f"Most similar: '{sample_texts[np.argmax(sims)]}'")
