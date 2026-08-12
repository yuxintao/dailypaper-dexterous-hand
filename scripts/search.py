#!/usr/bin/env python3
"""
Daily paper search engine.
Sources:
  - Semantic Scholar API (primary, with API key for higher rate limit)
  - arXiv API (fallback, no rate limit, covers all CS/robotics)
Google Patents scraping removed (unreliable from GitHub Actions IP).
"""
import requests
import time
import re
import sys
import os
import xml.etree.ElementTree as ET
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

ARXIV_BASE = "http://export.arxiv.org/api/query"

REQUEST_DELAY = 10.0      # seconds between API calls
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

# arXiv queries — same 3-round structure, arXiv search syntax
ARXIV_QUERIES = [
    # Round 1: Core magnetic tactile
    ("all:magnetic AND all:tactile AND (all:robot OR all:hand OR all:dexterous)", 10),
    ("all:hall AND all:tactile AND all:sensor AND (all:robot OR all:finger)", 10),
    ("all:magnetic AND all:field AND (all:tactile OR all:localization) AND (all:dexterous OR all:manipulation)", 5),
    # Round 2: Tactile sensing (broader)
    ("all:tactile AND all:sensor AND all:force AND all:estimation AND all:neural", 5),
    ("all:magnetic AND all:localization AND (all:tactile OR all:position) AND all:regression", 5),
    ("all:tactile AND all:self-supervised AND (all:calibration OR all:deep OR all:learning)", 5),
    # Round 3: Dexterous hand + sensing
    ("all:dexterous AND all:hand AND all:touch AND all:manipulation", 5),
    ("all:robot AND all:finger AND all:tactile AND all:force AND all:learning", 5),
]

# ============================================================
# Semantic Scholar
# ============================================================
def _api_call(url: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    """Call Semantic Scholar API with retry on 429."""
    headers = {
        "User-Agent": "dailypaper-dexterous-hand/1.0",
        "Accept": "application/json",
    }
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

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
        "fieldsOfStudy": "Computer Science,Engineering",
        "publicationTypes": "Review,JournalArticle,Conference",
        "openAccessPdf": "",
        "fields": "title,authors,year,venue,publicationDate,externalIds,abstract,tldr,citationCount,isOpenAccess,openAccessPdf",
    }
    data = _api_call(url, params)
    return data.get("data", [])


# ============================================================
# arXiv
# ============================================================
def _normalize_title(title: str) -> str:
    """Normalize title for dedup key."""
    return re.sub(r'[^a-z0-9]', '', title.lower())[:80]


def search_arxiv(query: str, max_results: int = 10) -> list:
    """
    Search papers on arXiv API.
    Returns list of paper dicts compatible with Semantic Scholar format.
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    papers = []
    try:
        resp = requests.get(ARXIV_BASE, params=params, timeout=30,
                           headers={"User-Agent": "dailypaper-dexterous-hand/1.0"})
        resp.raise_for_status()

        # Parse Atom XML
        root = ET.fromstring(resp.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        for entry in root.findall("atom:entry", ns):
            try:
                # Extract identifiers
                arxiv_id_full = entry.find("atom:id", ns).text
                # e.g. "http://arxiv.org/abs/2506.15953v1" → "2506.15953"
                arxiv_id = arxiv_id_full.split("/")[-1].split("v")[0] if arxiv_id_full else ""

                # Extract title
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

                # Extract authors
                authors = []
                for author_elem in entry.findall("atom:author", ns):
                    name_elem = author_elem.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append({"name": name_elem.text.strip()})

                # Extract abstract
                summary_elem = entry.find("atom:summary", ns)
                abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""

                # Extract year
                published_elem = entry.find("atom:published", ns)
                year = 0
                if published_elem is not None and published_elem.text:
                    try:
                        year = int(published_elem.text[:4])
                    except ValueError:
                        year = 0

                # Extract categories (as venue proxy)
                categories = []
                for cat_elem in entry.findall("arxiv:primary_category", ns):
                    if cat_elem.get("term"):
                        categories.append(cat_elem.get("term"))
                venue = ", ".join(categories) if categories else "arXiv"

                papers.append({
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "publicationDate": published_elem.text[:10] if published_elem is not None and published_elem.text else "",
                    "externalIds": {"ArXiv": arxiv_id},
                    "abstract": abstract,
                    "tldr": {},
                    "citationCount": 0,
                    "isOpenAccess": True,
                    "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"},
                    "source": "arxiv",
                })
            except Exception as e:
                print(f"  [WARN] Failed to parse arXiv entry: {e}")
                continue

    except requests.exceptions.RequestException as e:
        print(f"  [WARN] arXiv API request failed: {e}")
        return []
    except ET.ParseError as e:
        print(f"  [WARN] arXiv XML parse failed: {e}")
        return []

    return papers


# ============================================================
# Unified search
# ============================================================
def run_search(verbose: bool = True) -> list:
    """
    Execute search across Semantic Scholar AND arXiv.
    Returns deduplicated results (arXiv wins on duplicate titles).
    """
    all_papers = []
    api_has_key = bool(SEMANTIC_SCHOLAR_API_KEY)

    # ---- Semantic Scholar ----
    if verbose:
        key_status = "with API key" if api_has_key else "NO API key (rate limit will be strict)"
        print(f"  [Semantic Scholar] {key_status}")
    for i, (query, limit) in enumerate(QUERIES):
        round_num = 1 if i < 3 else (2 if i < 6 else 3)
        if verbose:
            print(f"  [R{round_num}] SS: {query}")

        papers = search_semantic_scholar(query, limit=limit)
        for p in papers:
            p["search_round"] = round_num
            p["source"] = "semantic_scholar"

        if verbose:
            print(f"         Found: {len(papers)} papers")

        all_papers.extend(papers)

        if i < len(QUERIES) - 1:
            time.sleep(REQUEST_DELAY)

    # ---- arXiv ----
    if verbose:
        print(f"  [arXiv] searching {len(ARXIV_QUERIES)} queries...")
    for i, (query, max_results) in enumerate(ARXIV_QUERIES):
        round_num = 1 if i < 3 else (2 if i < 6 else 3)
        if verbose:
            print(f"  [R{round_num}] arXiv: {query[:80]}...")

        papers = search_arxiv(query, max_results=max_results)
        for p in papers:
            p["search_round"] = round_num
            p["source"] = "arxiv"

        if verbose:
            print(f"         Found: {len(papers)} papers")

        all_papers.extend(papers)

        if i < len(ARXIV_QUERIES) - 1:
            time.sleep(3.0)  # arXiv asks for polite delays

    # ---- Deduplicate ----
    # arXiv wins on duplicate (it has more accurate year info and always has PDF)
    seen = {}
    unique = []
    for p in all_papers:
        key = _normalize_title(p.get("title", ""))
        if not key:
            continue
        if key in seen:
            existing = seen[key]
            # arXiv wins over Semantic Scholar
            if p.get("source") == "arxiv" and existing.get("source") == "semantic_scholar":
                unique.remove(existing)
                unique.append(p)
                seen[key] = p
        else:
            seen[key] = p
            unique.append(p)

    if verbose:
        ss_count = sum(1 for p in unique if p.get("source") == "semantic_scholar")
        arxiv_count = sum(1 for p in unique if p.get("source") == "arxiv")
        print(f"\n  Total unique: {len(unique)} papers (Semantic Scholar: {ss_count}, arXiv: {arxiv_count})")
    return unique


if __name__ == "__main__":
    papers = run_search()
    for i, p in enumerate(papers[:20]):
        title = p.get("title", "N/A")[:90]
        year = p.get("year", "?")
        source = p.get("source", "?")
        citations = p.get("citationCount", "?")
        print(f"  {i+1:2d}. [{source[:4]}|{year}|cit:{citations}] {title}")
