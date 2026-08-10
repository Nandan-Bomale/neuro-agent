"""
pubmed_fetcher.py
-----------------
Fetches abstracts from PubMed via the NCBI E-utilities API.
Free, no institutional access required — just a free NCBI API key.

Usage:
    fetcher = PubMedFetcher()
    results = fetcher.search("glioblastoma multiforme MRI detection", max_results=50)
"""

import os
import time
import logging
from typing import Optional
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """Represents a single PubMed article with its metadata."""
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    year: str
    url: str

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "url": self.url,
        }

    def to_text(self) -> str:
        """Plain text representation used for embedding."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return (
            f"Title: {self.title}\n"
            f"Authors: {authors_str}\n"
            f"Journal: {self.journal} ({self.year})\n"
            f"Abstract: {self.abstract}"
        )


class PubMedFetcher:
    """
    Fetches PubMed abstracts using the NCBI E-utilities REST API.

    Steps:
        1. esearch  — get list of PMIDs matching a query
        2. efetch   — get full article details for those PMIDs

    Rate limits:
        - Without API key: 3 requests/second
        - With API key:   10 requests/second
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NCBI_API_KEY", "")
        self.delay = 0.11 if self.api_key else 0.34   # respect rate limits
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NeuroAgent/1.0 (academic research)"})

    def _get(self, endpoint: str, params: dict) -> dict:
        """Make a GET request to the NCBI API with retries."""
        if self.api_key:
            params["api_key"] = self.api_key
        params["retmode"] = "json"

        for attempt in range(3):
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/{endpoint}",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                time.sleep(self.delay)
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"NCBI API attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        return {}

    def _search_pmids(self, query: str, max_results: int = 50) -> list[str]:
        """Step 1: Search PubMed and return a list of PMIDs."""
        data = self._get("esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
        })
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"Found {len(pmids)} PMIDs for query: '{query}'")
        return pmids

    def _fetch_details(self, pmids: list[str]) -> list[PubMedArticle]:
        """Step 2: Fetch full article details for a list of PMIDs."""
        if not pmids:
            return []

        # Fetch in batches of 20 to avoid URL length limits
        articles = []
        batch_size = 20
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i: i + batch_size]
            data = self._get("esummary.fcgi", {
                "db": "pubmed",
                "id": ",".join(batch),
            })
            articles.extend(self._parse_summaries(data, batch))

        # Fetch abstracts separately via efetch
        abstracts = self._fetch_abstracts(pmids)

        # Merge abstracts into article objects
        for article in articles:
            article.abstract = abstracts.get(article.pmid, "Abstract not available.")

        return articles

    def _parse_summaries(self, data: dict, pmids: list[str]) -> list[PubMedArticle]:
        """Parse esummary JSON response into PubMedArticle objects."""
        articles = []
        result = data.get("result", {})

        for pmid in pmids:
            if pmid not in result:
                continue
            entry = result[pmid]

            # Authors
            authors = [
                a.get("name", "") for a in entry.get("authors", [])
                if a.get("authtype") == "Author"
            ]

            # Publication year
            pub_date = entry.get("pubdate", "")
            year = pub_date[:4] if pub_date else "Unknown"

            articles.append(PubMedArticle(
                pmid=pmid,
                title=entry.get("title", "No title"),
                abstract="",   # filled in later
                authors=authors,
                journal=entry.get("source", "Unknown Journal"),
                year=year,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            ))

        return articles

    def _fetch_abstracts(self, pmids: list[str]) -> dict[str, str]:
        """Fetch full abstracts via efetch (returns XML, we parse text)."""
        abstracts = {}
        batch_size = 20

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i: i + batch_size]
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "rettype": "abstract",
                        "retmode": "text",
                        **({"api_key": self.api_key} if self.api_key else {}),
                    },
                    timeout=30,
                )
                response.raise_for_status()
                time.sleep(self.delay)

                # Parse plain text response — abstracts are separated by blank lines
                # and preceded by PMID header lines
                text = response.text
                self._parse_abstract_text(text, batch, abstracts)

            except requests.RequestException as e:
                logger.warning(f"Failed to fetch abstracts for batch: {e}")

        return abstracts

    def _parse_abstract_text(
        self, text: str, pmids: list[str], abstracts: dict
    ) -> None:
        """
        Parse the plain-text efetch response.
        Each record starts with '1. ' or '2. ' etc. (numbered by order).
        We match them positionally to the PMIDs.
        """
        # Split on numbered record separators
        import re
        records = re.split(r"\n\n\d+\. ", "\n\n1. " + text.strip())
        records = [r.strip() for r in records if r.strip()]

        for idx, (pmid, record) in enumerate(zip(pmids, records)):
            # Extract the abstract section
            abstract_match = re.search(
                r"(?:ABSTRACT|Abstract)\s*\n(.*?)(?:\n\nPMID:|$)",
                record,
                re.DOTALL,
            )
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                # Clean up extra whitespace
                abstract_text = re.sub(r"\s+", " ", abstract_text)
                abstracts[pmid] = abstract_text
            else:
                # Fallback: take everything after the title block
                lines = record.split("\n")
                # Skip the first 3 lines (usually authors/journal info)
                body = " ".join(lines[3:]).strip()
                abstracts[pmid] = body[:1000] if body else "Abstract not available."

    def search(
        self,
        query: str,
        max_results: int = 50,
        min_abstract_length: int = 100,
    ) -> list[PubMedArticle]:
        """
        Main public method. Search PubMed and return article list.

        Args:
            query:               PubMed search query string
            max_results:         Max number of articles to retrieve
            min_abstract_length: Skip articles with very short abstracts

        Returns:
            List of PubMedArticle objects, filtered and ready for embedding
        """
        logger.info(f"Searching PubMed for: '{query}' (max {max_results} results)")

        pmids = self._search_pmids(query, max_results)
        if not pmids:
            logger.warning("No results found.")
            return []

        articles = self._fetch_details(pmids)

        # Filter out articles with no meaningful abstract
        filtered = [
            a for a in articles
            if len(a.abstract) >= min_abstract_length
        ]
        logger.info(
            f"Returning {len(filtered)}/{len(articles)} articles "
            f"(filtered {len(articles) - len(filtered)} with short abstracts)"
        )
        return filtered

    def search_bulk(
        self, queries: list[str], max_per_query: int = 30
    ) -> list[PubMedArticle]:
        """
        Search multiple queries and return deduplicated results.
        Useful for building a comprehensive literature corpus.
        """
        seen_pmids: set[str] = set()
        all_articles: list[PubMedArticle] = []

        for query in queries:
            articles = self.search(query, max_results=max_per_query)
            for article in articles:
                if article.pmid not in seen_pmids:
                    seen_pmids.add(article.pmid)
                    all_articles.append(article)
            logger.info(
                f"Corpus size after '{query}': {len(all_articles)} unique articles"
            )

        return all_articles


# ── Predefined query set for NeuroAgent literature corpus ─────────────────────
NEUROAGENT_QUERIES = [
    "brain tumor MRI detection deep learning",
    "glioblastoma MRI segmentation U-Net",
    "brain MRI tumor classification convolutional neural network",
    "glioma grading MRI radiomics",
    "meningioma MRI diagnosis treatment",
    "brain metastasis MRI detection",
    "BraTS brain tumor segmentation benchmark",
    "radiology report generation natural language processing",
    "explainable AI medical imaging Grad-CAM",
    "MRI tumor segmentation MONAI deep learning",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fetcher = PubMedFetcher()
    results = fetcher.search("brain tumor MRI U-Net segmentation", max_results=5)

    for article in results:
        print(f"\n{'='*60}")
        print(f"PMID: {article.pmid}")
        print(f"Title: {article.title}")
        print(f"Authors: {', '.join(article.authors[:3])}")
        print(f"Journal: {article.journal} ({article.year})")
        print(f"Abstract: {article.abstract[:200]}...")
        print(f"URL: {article.url}")
