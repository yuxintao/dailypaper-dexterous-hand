#!/usr/bin/env python3
"""
Local PDF sync script.
Run after 'git pull' to download any missing PDFs for recommended papers.
Parses all papers/*.md files, extracts arXiv IDs, downloads PDFs to archive/pdfs/.
"""
import os
import re
import requests
import sys

ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "..", "archive", "pdfs")
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "papers")


def find_arxiv_ids(markdown_text: str) -> list:
    """Extract arXiv IDs from a paper decomposition note."""
    ids = []
    for pattern in [
        r'> 📄 arXiv: \[(\d{4}\.\d{4,5})\]',      # new format
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',       # PDF links
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',       # abs links
        r'ArXiv:\s*(\d{4}\.\d{4,5})',              # plain text reference
    ]:
        found = re.findall(pattern, markdown_text, re.IGNORECASE)
        ids.extend(found)
    return list(set(ids))


def search_arxiv_by_title(title: str) -> str:
    """Fallback: search arXiv API by title to find arXiv ID. Validates title match. Returns '' if not found."""
    import urllib.parse
    import difflib
    try:
        query = title[:120].strip()
        url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&max_results=5"
        resp = requests.get(url, timeout=15,
                           headers={"User-Agent": "dailypaper-sync/1.0"})
        resp.raise_for_status()
        # Parse all entries to find best title match
        entries = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
        best_id = ""
        best_ratio = 0
        for entry in entries:
            m_id = re.search(r'<id>http://arxiv\.org/abs/(\d{4}\.\d{4,5})', entry)
            m_title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            if m_id and m_title:
                found_title = m_title.group(1).strip().replace('\n', ' ')
                ratio = difflib.SequenceMatcher(None, title.lower()[:60], found_title.lower()[:60]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_id = m_id.group(1)
        # Require at least 80% title similarity to accept the match
        if best_ratio >= 0.80:
            return best_id
        else:
            print(f"     ⚠ best match ratio={best_ratio:.2f} < 0.80, skipping")
            return ""
    except Exception:
        pass
    return ""


def find_paper_titles(markdown_text: str) -> list:
    """Extract paper titles from a daily recommendation note."""
    titles = []
    for line in markdown_text.split("\n"):
        line = line.strip()
        # Paper titles start with "# " but not "# 每日" or "## "
        if line.startswith("# ") and "每日" not in line and "推荐" not in line and "评分" not in line:
            titles.append(line[2:].strip())
    return titles


def download_pdf(arxiv_id: str) -> str:
    """Download a single PDF from arXiv. Returns local path or empty string."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    filepath = os.path.join(ARCHIVE_DIR, f"{arxiv_id}.pdf")

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"  ✓ already cached: {arxiv_id}.pdf")
        return filepath

    try:
        print(f"  ↓ downloading: {arxiv_id}.pdf ...")
        resp = requests.get(url, timeout=60,
                           headers={"User-Agent": "dailypaper-sync/1.0"})
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) // 1024
        print(f"  ✓ saved: {arxiv_id}.pdf ({size_kb} KB)")
        return filepath
    except Exception as e:
        print(f"  ✗ failed: {arxiv_id} — {e}")
        return ""


def main():
    today = __import__('datetime').datetime.now()
    year = today.strftime("%Y")
    month = today.strftime("%m")

    output_dir = os.path.join(ARCHIVE_DIR, year, month)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all arXiv IDs from all paper notes
    all_ids = set()
    id_by_title = {}  # title -> arxiv_id mapping
    if not os.path.exists(PAPERS_DIR):
        print("No papers directory found.")
        return

    for root, dirs, files in os.walk(PAPERS_DIR):
        for fname in sorted(files):
            if fname.endswith(".md") and fname[0].isdigit():
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                ids = find_arxiv_ids(content)
                if ids:
                    for aid in ids:
                        all_ids.add(aid)
                else:
                    # Fallback: search arXiv by title for legacy papers
                    titles = find_paper_titles(content)
                    for title in titles:
                        if title not in id_by_title:
                            print(f"  🔍 Searching arXiv for: {title[:60]}...")
                            aid = search_arxiv_by_title(title)
                            if aid:
                                id_by_title[title] = aid
                                all_ids.add(aid)
                                print(f"     → {aid}")
                            else:
                                print(f"     → not found")
                            # Small delay to be polite to arXiv API
                            import time
                            time.sleep(1.0)

    if not all_ids:
        print("No arXiv IDs found in papers.")
        return

    print(f"Found {len(all_ids)} unique arXiv IDs. Syncing PDFs to {output_dir} ...\n")

    downloaded = 0
    for aid in sorted(all_ids):
        # Save to year/month subdirectory matching the paper's date
        if download_pdf(aid):
            downloaded += 1

    print(f"\nDone. {downloaded}/{len(all_ids)} PDFs synced.")


if __name__ == "__main__":
    main()
