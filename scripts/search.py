#!/usr/bin/env python3
"""
Daily paper search engine.
Semantic Scholar as primary source, with rate-limit handling and retry logic.
Google Patents scraping removed (unreliable from GitHub Actions IP).
"""
import requests
import time
import re
import sys
import os
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
REQUEST_DELAY = 10.0      # seconds between API calls (GitHub Actions IPs get rate-limited)
RETRY_DELAY = 35.0        # wait when hitting 429
MAX_RETRIES = 2

# ============================================================
# Search queries — tightly scoped to tactile + dexterous hand
# ============================================================
QUERIES = [
    # Round 1: Core magnetic tactile
    ("magnetic tactile sensor + robot hand force localization", 8),
    ("hall effect sensor array + tactile skin manipulation", 8),
    ("magnetic field sensing + dexterous manipulation learning", 5),
    # Round 2: Tactile sensing (broader)
    ("tactile sensor + robot hand + force estimation neural network", 5),
    ("tactile sensing + magnetic localization + position regression", 5),
    ("tactile sensor + deep learning + calibration self-supervised", 5),
    # Round 3: Dexterous hand + sensing
    ("dexterous hand + touch sensing + manipulation 2024 2025", 5),
    ("robot finger + tactile array + force feedback learning", 5),
]


def _api_call(url: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    """Call Semantic Scholar API with retry on 429."""
    headers = {
        "User-Agent": "dailypaper-dexterous-hand/1.0",
        "Accept": "application/json",
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  [429] Rate limited. Waiting {wait:.0f}s (attempt {attempt+1}/{retries+1})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                print(f"  [WARN] Request failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [WARN] Giving up after {retries+1} attempts: {e}")
                return {"data": []}
    return {"data": []}


def search_semantic_scholar(query: str, limit: int = 8) -> list:
    """Search papers on Semantic Scholar."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "offset": 0,
        "year": "2023-",
        "fields": "title,authors,year,venue,publicationDate,externalIds,abstract,tldr,citationCount,isOpenAccess,openAccessPdf",
    }
    data = _api_call(url, params)
    return data.get("data", [])


def run_search(verbose: bool = True) -> list:
    """Execute search across all queries and return deduplicated results."""
    all_papers = []

    for i, (query, limit) in enumerate(QUERIES):
        round_num = 1 if i < 3 else (2 if i < 6 else 3)
        if verbose:
            print(f"  [R{round_num}] Searching: {query}")

        papers = search_semantic_scholar(query, limit=limit)
        for p in papers:
            p["search_round"] = round_num
            p["source"] = "semantic_scholar"

        if verbose:
            print(f"         Found: {len(papers)} papers")

        all_papers.extend(papers)

        if i < len(QUERIES) - 1:
            time.sleep(REQUEST_DELAY)

    # Deduplicate by title
    seen = set()
    unique = []
    for p in all_papers:
        key = re.sub(r'[^a-z0-9]', '', p.get("title", "").lower())[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(p)

    if verbose:
        print(f"\n  Total unique: {len(unique)} papers")
    return unique


if __name__ == "__main__":
    papers = run_search()
    for i, p in enumerate(papers[:20]):
        title = p.get("title", "N/A")[:90]
        year = p.get("year", "?")
        citations = p.get("citationCount", "?")
        print(f"  {i+1:2d}. [{year}|cit:{citations}] {title}")
