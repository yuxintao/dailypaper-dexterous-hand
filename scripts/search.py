#!/usr/bin/env python3
"""
Daily paper recommendation search engine.
Supports Semantic Scholar API, arXiv, Google Patents, and Chinese patent databases.
"""
import requests
import time
import json
import re
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

# ============================================================
# Configuration
# ============================================================
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
USER_AGENT = "dailypaper-dexterous-hand/1.0 (mailto:researcher@example.com)"
REQUEST_DELAY = 3.1  # seconds between API calls (respect rate limit: 100/5min)

# ============================================================
# Query generation
# ============================================================
def generate_queries():
    """Generate search queries for the 3 rounds."""
    return {
        "round1_exact": [
            "magnetic tactile sensor robot finger localization force",
            "hall effect sensor array tactile skin dexterous manipulation",
            "magnetic field based touch sensing deep learning regression",
        ],
        "round2_complementary": [
            "tactile sensor robot hand force estimation deep learning",
            "soft tactile sensing magnetic localization position prediction",
            "tactile sensor calibration self-supervised transfer learning",
        ],
        "round3_patent": [
            "magnetic tactile sensor robot hand",
            "hall effect array force touch localization",
        ],
    }

# ============================================================
# Semantic Scholar API
# ============================================================
def search_semantic_scholar(query: str, limit: int = 10, year_from: str = "2023") -> list:
    """Search papers on Semantic Scholar."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "offset": 0,
        "year": f"{year_from}-",
        "fields": "title,authors,year,venue,publicationDate,externalIds,abstract,tldr,citationCount,isOpenAccess,openAccessPdf,fieldsOfStudy",
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        print(f"  [WARN] Semantic Scholar search failed for '{query}': {e}")
        return []


def get_paper_details(paper_id: str) -> Optional[dict]:
    """Get detailed info for a specific paper."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}"
    params = {
        "fields": "title,authors,year,venue,publicationDate,externalIds,abstract,tldr,citationCount,references,citations,isOpenAccess,openAccessPdf"
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] Failed to get details for {paper_id}: {e}")
        return None

# ============================================================
# arXiv API (via Semantic Scholar or direct)
# ============================================================
def search_arxiv(query: str, max_results: int = 10) -> list:
    """Search arXiv for recent preprints."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        # Parse Atom XML (simple regex approach)
        entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
        results = []
        for entry in entries:
            title = re.search(r"<title>(.*?)</title>", entry)
            summary = re.search(r"<summary>(.*?)</summary>", entry)
            arxiv_id = re.search(r"<id>.*?/(.*?)</id>", entry)
            published = re.search(r"<published>(.*?)</published>", entry)
            results.append({
                "title": title.group(1).strip() if title else "",
                "abstract": summary.group(1).strip()[:500] if summary else "",
                "arxivId": arxiv_id.group(1) if arxiv_id else "",
                "year": published.group(1)[:4] if published else "",
                "source": "arxiv",
            })
        return results
    except Exception as e:
        print(f"  [WARN] arXiv search failed: {e}")
        return []

# ============================================================
# Patent search (Google Patents + WIPO)
# ============================================================
def search_google_patents(query: str, max_results: int = 5) -> list:
    """Search Google Patents (web scraping, gentle rate)."""
    url = "https://patents.google.com/"
    params = {
        "q": query,
        "num": max_results,
        "language": "EN",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        # Extract patent titles and numbers from HTML (simple regex)
        titles = re.findall(r'<result-title[^>]*>(.*?)</result-title>', resp.text, re.DOTALL)
        patent_ids = re.findall(r'/patent/([A-Z]{2}\d+[A-Z]?\d*)/', resp.text)
        results = []
        for i, title in enumerate(titles[:max_results]):
            results.append({
                "title": re.sub(r'<[^>]+>', '', title).strip(),
                "patentId": patent_ids[i] if i < len(patent_ids) else "",
                "source": "google_patents",
                "year": "",
            })
        return results
    except Exception as e:
        print(f"  [WARN] Google Patents search failed: {e}")
        return []

# ============================================================
# Deduplication
# ============================================================
def deduplicate(papers: list) -> list:
    """Remove duplicate papers by title similarity."""
    seen_titles = set()
    unique = []
    for paper in papers:
        title_key = re.sub(r'[^a-z0-9]', '', paper.get("title", "").lower())[:80]
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(paper)
    return unique

# ============================================================
# Main search orchestrator
# ============================================================
def run_search(verbose: bool = True) -> list:
    """
    Execute the full 3-round search pipeline.
    Returns a list of deduplicated, scored paper dicts.
    """
    all_papers = []
    queries = generate_queries()

    # Round 1: Exact match (Semantic Scholar)
    if verbose:
        print("=== Round 1: Exact Match (Semantic Scholar) ===")
    for q in queries["round1_exact"]:
        if verbose:
            print(f"  Query: {q}")
        papers = search_semantic_scholar(q, limit=8)
        for p in papers:
            p["search_round"] = 1
            p["source"] = "semantic_scholar"
        all_papers.extend(papers)
        time.sleep(REQUEST_DELAY)

    # Round 2: Complementary methods (Semantic Scholar + arXiv)
    if verbose:
        print("\n=== Round 2: Complementary Methods ===")
    for q in queries["round2_complementary"]:
        if verbose:
            print(f"  Query: {q}")
        papers = search_semantic_scholar(q, limit=5)
        for p in papers:
            p["search_round"] = 2
            p["source"] = "semantic_scholar"
        all_papers.extend(papers)
        time.sleep(REQUEST_DELAY)
    # Also check arXiv for very recent preprints
    arxiv_papers = search_arxiv("tactile sensor robot hand learning", max_results=5)
    for p in arxiv_papers:
        p["search_round"] = 2
    all_papers.extend(arxiv_papers)

    # Round 3: Patents
    if verbose:
        print("\n=== Round 3: Patents ===")
    for q in queries["round3_patent"]:
        if verbose:
            print(f"  Query: {q}")
        patents = search_google_patents(q, max_results=3)
        for p in patents:
            p["search_round"] = 3
            p["is_patent"] = True
        all_papers.extend(patents)
        time.sleep(REQUEST_DELAY * 2)

    # Deduplicate
    all_papers = deduplicate(all_papers)
    if verbose:
        print(f"\n=== Total unique papers found: {len(all_papers)} ===")

    return all_papers


if __name__ == "__main__":
    papers = run_search()
    for i, p in enumerate(papers[:15]):
        title = p.get("title", "N/A")[:80]
        year = p.get("year", "N/A")
        citations = p.get("citationCount", "N/A")
        source = p.get("source", "N/A")
        print(f"{i+1:2d}. [{year}] {title}")
        print(f"     Citations: {citations} | Source: {source} | Round: {p.get('search_round','?')}")
        print()
