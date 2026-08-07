#!/usr/bin/env python3
"""
Daily paper recommendation orchestrator.
Run once per weekday via GitHub Actions.
"""
import sys
import os
import re
from datetime import datetime, timedelta

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(__file__))

from search import run_search
from decompose import decompose_paper

# ============================================================
# PDF download
# ============================================================
def download_pdf(paper: dict, output_dir: str) -> str:
    """Download PDF for a paper if open access. Returns local path or empty string."""
    import requests

    # Try openAccessPdf from Semantic Scholar
    pdf_url = paper.get("openAccessPdf", {}).get("url", "")
    if not pdf_url:
        # Try constructing arXiv PDF URL from externalIds
        arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    if not pdf_url:
        return ""

    # Safe filename
    title = paper.get("title", "paper")
    safe_title = "".join(c for c in title[:60] if c.isascii() and c not in r'\/:*?"<>|').strip().replace(" ", "_")
    filepath = os.path.join(output_dir, f"{safe_title}.pdf")

    if os.path.exists(filepath):
        print(f"     PDF already cached: {os.path.basename(filepath)}")
        return filepath

    try:
        print(f"     Downloading PDF: {pdf_url[:80]}...")
        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "dailypaper-dexterous-hand/1.0"})
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) // 1024
        print(f"     PDF saved: {os.path.basename(filepath)} ({size_kb} KB)")
        return filepath
    except Exception as e:
        print(f"     [WARN] PDF download failed: {e}")
        return ""

# ============================================================
# Keyword filter — papers MUST match at least 1 cluster
# ============================================================
RELEVANCE_KEYWORDS = [
    # Cluster 1: magnetic tactile / hall effect
    r"magnetic.{0,5}tactile|tactile.{0,5}magnetic|hall.{0,5}effect|hall.{0,5}sensor|magnetic.{0,5}field.{0,5}sens|magnetic.{0,5}localization|magnetic.{0,5}flux",
    # Cluster 2: robot hand / dexterous manipulation
    r"dexterous.{0,5}hand|robot.{0,5}hand|robot.{0,5}finger|prosthetic.{0,5}hand|gripper|manipulator|grasp",
    # Cluster 3: sensor array / tactile skin / e-skin
    r"sensor.{0,5}array|tactile.{0,5}skin|e.{0,2}skin|tactile.{0,5}sens|touch.{0,5}sens|soft.{0,5}sensor|flexible.{0,5}sensor",
    # Cluster 4: tactile deep learning
    r"tactile.{0,5}(deep|learn|neural|CNN|MLP|transformer|GNN)|force.{0,5}estimation.{0,5}(neural|learn|deep)|(neural|deep|learn).{0,5}tactile",
    # Cluster 5: magnetic modeling / dipole / simulation
    r"dipole.{0,5}model|magnetic.{0,5}(forward|inverse|inversion|simulation|FEM|finite.{0,5}element)|COMSOL",
    # Cluster 6: hand control / grasp / tactile servo
    r"impedance.{0,5}control|force.{0,5}control|grasp.{0,5}plan|slip.{0,5}detect|tactile.{0,5}servo|force.{0,5}feedback",
    # Cluster 7: calibration / noise / domain adaptation for sensors
    r"sensor.{0,5}(calibrat|noise|drift|domain.{0,5}adapt|transfer)|self.{0,5}supervised.{0,5}(tactile|touch|sensor)",
    # Cluster 8: novel tactile paradigms
    r"elastic.{0,5}wave.{0,5}tactile|ultrasonic.{0,5}tactile|triboelectric|piezoelectric.{0,5}touch|capacitive.{0,5}tactile|fiber.{0,5}optic.{0,5}tactile",
]


def is_relevant(paper: dict) -> bool:
    """Check if a paper is remotely relevant to tactile sensing / dexterous hands."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    if isinstance(abstract, dict):
        abstract = abstract.get("text", "")
    text = (title + " " + str(abstract)).lower()

    for pattern in RELEVANCE_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def is_weekday(d: datetime = None) -> bool:
    """Check if today is a weekday (Mon-Fri)."""
    if d is None:
        d = datetime.now()
    return d.weekday() < 5  # 0=Mon, 4=Fri


def run_daily():
    """Execute the daily recommendation pipeline."""
    today = datetime.now()

    # Skip weekends
    if not is_weekday(today):
        print(f"Today ({today.strftime('%A')}) is not a weekday. Skipping.")
        return

    print(f"=== Daily Paper Recommendation: {today.strftime('%Y-%m-%d')} ===")
    print()

    # Step 1: Search
    print(">>> Phase 1: Searching...")
    all_papers = run_search(verbose=True)

    if not all_papers:
        print("ERROR: No papers found. Check API connectivity.")
        return

    # Step 1.5: Filter for relevance
    relevant = [p for p in all_papers if is_relevant(p)]
    print(f"\n>>> Phase 1.5: Relevance filter: {len(relevant)}/{len(all_papers)} papers pass keyword check")
    for p in all_papers:
        if p not in relevant:
            print(f"  ✗ FILTERED: {p.get('title', 'N/A')[:80]}")

    all_papers = relevant

    if not all_papers:
        print("ERROR: All papers filtered out. No relevant papers today.")
        # Write an empty report so the workflow doesn't crash on no-changes commit
        return

    # Step 2: Score and rank
    print(f"\n>>> Phase 2: Scoring {len(all_papers)} candidates...")
    for p in all_papers:
        score = 0
        year = p.get("year", 0)
        if isinstance(year, str):
            try:
                year = int(year)
            except ValueError:
                year = 0
        if year >= 2025:
            score += 5
        elif year >= 2023:
            score += 3
        elif year > 0:
            score += 1
        citations = p.get("citationCount", 0)
        if isinstance(citations, (int, float)) and citations > 10:
            score += 1
        if p.get("is_patent"):
            score += 1
        p["_score"] = score

    all_papers.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # Step 3: Select top 3
    selected = all_papers[:3]

    # Step 3.5: Download PDFs
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "archive", "pdfs",
                           today.strftime("%Y"), today.strftime("%m"))
    os.makedirs(pdf_dir, exist_ok=True)
    print(f"\n>>> Phase 3.5: Downloading PDFs for {len(selected)} papers...")
    for p in selected:
        p["_pdf_path"] = download_pdf(p, pdf_dir)

    # Step 4: Decompose and write
    print(f"\n>>> Phase 3: Decomposing {len(selected)} recommendations...")
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "papers",
        today.strftime("%Y"), today.strftime("%m")
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, today.strftime("%m%d") + ".md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 每日论文推荐 — {today.strftime('%Y年%m月%d日')}\n\n")
        f.write(f"> 搜索时间：{today.strftime('%H:%M UTC')}\n")
        f.write(f"> 候选论文：{len(all_papers)} 篇（已过滤无关结果）→ 精选 {len(selected)} 篇\n\n")
        f.write("---\n\n")

        for i, paper in enumerate(selected):
            print(f"  [{i+1}/{len(selected)}] Decomposing: {paper.get('title', 'N/A')[:60]}...")
            note = decompose_paper(paper, use_api=True)
            f.write(note)
            f.write("\n\n---\n\n")

    print(f"\n>>> Done! Output: {output_path}")
    for i, p in enumerate(selected):
        print(f"  {i+1}. {p.get('title', 'N/A')[:100]}")


if __name__ == "__main__":
    run_daily()
