#!/usr/bin/env python3
"""
Daily paper recommendation orchestrator.
Run once per weekday via GitHub Actions.
"""
import sys
import os
from datetime import datetime, timedelta

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(__file__))

from search import run_search
from decompose import decompose_paper


def is_weekday(d: datetime = None) -> bool:
    """Check if today is a weekday (Mon-Fri)."""
    if d is None:
        d = datetime.now()
    return d.weekday() < 5  # 0=Mon, 4=Fri


def get_recent_recommendations(days: int = 5) -> list:
    """Get titles of recently recommended papers for diversity check."""
    recent = []
    base_dir = os.path.join(os.path.dirname(__file__), "..", "papers")
    for i in range(1, days + 1):
        d = datetime.now() - timedelta(days=i)
        path = os.path.join(base_dir, d.strftime("%Y"), d.strftime("%m"), d.strftime("%m%d") + ".md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Extract titles (lines starting with "# ")
                for line in content.split("\n"):
                    if line.startswith("# ") and not line.startswith("# 推荐"):
                        recent.append(line[2:].strip())
    return recent


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

    # Step 2: Score and rank (simplified - real implementation would use criteria.yml)
    print(f"\n>>> Phase 2: Scoring {len(all_papers)} candidates...")
    # Basic scoring by year recency + citation count
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
        if isinstance(citations, (int, float)) and citations > 50:
            score += 1
        if p.get("is_patent"):
            score += 1
        p["_score"] = score

    all_papers.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # Step 3: Select top 3 with diversity check
    print(f"\n>>> Phase 3: Selecting Top 3 recommendations...")
    recent_titles = get_recent_recommendations()
    selected = []
    for p in all_papers:
        if len(selected) >= 3:
            break
        title = p.get("title", "")
        # Simple diversity: skip if title looks very similar to recent ones
        too_similar = False
        for rt in recent_titles:
            # Quick overlap check
            words1 = set(title.lower().split()[:6])
            words2 = set(rt.lower().split()[:6])
            if len(words1 & words2) > 4:
                too_similar = True
                break
        if not too_similar:
            selected.append(p)

    # Fill up to 3 if we don't have enough
    for p in all_papers:
        if len(selected) >= 3:
            break
        if p not in selected:
            selected.append(p)

    # Step 4: Decompose and write
    print(f"\n>>> Phase 4: Decomposing and writing recommendations...")
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "papers",
        today.strftime("%Y"), today.strftime("%m")
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, today.strftime("%m%d") + ".md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 每日论文推荐 — {today.strftime('%Y年%m月%d日')}\n\n")
        f.write(f"> 搜索时间：{today.strftime('%H:%M UTC')}\n")
        f.write(f"> 候选论文：{len(all_papers)} 篇 → 精选 3 篇\n\n")
        f.write("---\n\n")

        for i, paper in enumerate(selected):
            print(f"  [{i+1}/3] Decomposing: {paper.get('title', 'N/A')[:60]}...")
            note = decompose_paper(paper, use_api=True)
            f.write(note)
            f.write("\n\n---\n\n")

    print(f"\n>>> Done! Output: {output_path}")
    print(f">>> Selected {len(selected)} papers:")
    for i, p in enumerate(selected):
        print(f"  {i+1}. {p.get('title', 'N/A')[:100]}")


if __name__ == "__main__":
    run_daily()
