#!/usr/bin/env python3
"""
Daily paper recommendation orchestrator.
Run once per weekday via GitHub Actions.

Scoring pipeline:
  1. Search + relevance filter + dedup (as before)
  2. Automatable scoring: timeliness + relevance + patent_bonus
  3. Narrow to top 6 → LLM decomposition (with embedded scoring)
  4. Extract LLM scores + apply diversity penalty → weighted total
  5. Select top 3 → write output
"""
import sys
import os
import re
import math
from datetime import datetime, timedelta
from collections import Counter

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(__file__))

from search import run_search
from decompose import decompose_paper

# ============================================================
# Scoring weights — synced with criteria.yml
# ============================================================
WEIGHTS = {
    "timeliness": 15,
    "innovation": 20,           # LLM
    "relevance": 25,
    "experiment_generation": 15, # LLM
    "reproducibility": 15,       # LLM
    "practicality": 15,          # LLM
    "engineering_feasibility": 5, # LLM
    "diversity": 5,              # computed
}
PATENT_BONUS = 1  # weight=0, direct bonus to final score
LLM_CRITERIA = ["innovation", "experiment_generation", "reproducibility",
                "practicality", "engineering_feasibility"]
LLM_DEFAULT = 2  # default score when LLM parsing fails

# ============================================================
# Keyword filter — papers MUST match at least 1 cluster
# Each regex maps to one of the 8 keyword clusters in criteria.yml
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


def count_cluster_hits(paper: dict) -> int:
    """Count how many keyword clusters a paper hits. Used for relevance scoring."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    if isinstance(abstract, dict):
        abstract = abstract.get("text", "")
    text = (title + " " + str(abstract)).lower()

    hits = 0
    for pattern in RELEVANCE_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            hits += 1
    return hits


def is_weekday(d: datetime = None) -> bool:
    """Check if today is a weekday (Mon-Fri)."""
    if d is None:
        d = datetime.now()
    return d.weekday() < 5


def get_all_recommended_titles() -> set:
    """Read all previously recommended paper titles from markdown files."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "papers")
    titles = set()
    if not os.path.exists(base_dir):
        return titles
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".md") and fname[0].isdigit():
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    for line in content.split("\n"):
                        line = line.strip()
                        if line.startswith("# ") and "每日" not in line and "推荐" not in line:
                            title = re.sub(r"[^a-z0-9]", "", line.lower())[:80]
                            if title:
                                titles.add(title)
                except Exception:
                    pass
    return titles


def is_duplicate(paper: dict, seen_titles: set) -> bool:
    """Check if a paper title is too similar to any previously recommended title."""
    title = paper.get("title", "")
    normalized = re.sub(r"[^a-z0-9]", "", title.lower())[:80]
    if normalized in seen_titles:
        return True
    short = normalized[:40]
    for st in seen_titles:
        if st[:40] == short:
            return True
    return False


# ============================================================
# PDF download
# ============================================================
def download_pdf(paper: dict, output_dir: str) -> str:
    """Download PDF for a paper if open access. Returns local path or empty string."""
    import requests

    pdf_url = paper.get("openAccessPdf", {}).get("url", "")
    if not pdf_url:
        arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    if not pdf_url:
        return ""

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
# Automatable scoring
# ============================================================
def score_timeliness(paper: dict) -> int:
    """Score based on publication year. Range 0-5."""
    year = paper.get("year", 0)
    if isinstance(year, str):
        try:
            year = int(year)
        except ValueError:
            year = 0
    if year >= 2025:
        return 5
    elif year >= 2023:
        return 3
    elif year > 0:
        return 1
    return 0


def score_relevance(paper: dict) -> int:
    """Score based on keyword cluster hits. Range 0-5."""
    hits = count_cluster_hits(paper)
    if hits >= 3:
        return 5
    elif hits >= 2:
        return 3
    elif hits >= 1:
        return 1
    return 0


def compute_automatable_score(paper: dict) -> float:
    """
    Compute score from criteria that don't need LLM:
      timeliness (weight 15) + relevance (weight 25) + patent_bonus
    Returns raw weighted score (not normalized).
    """
    s_tl = score_timeliness(paper)
    s_rel = score_relevance(paper)
    bonus = PATENT_BONUS if paper.get("is_patent") else 0

    score = WEIGHTS["timeliness"] * s_tl + WEIGHTS["relevance"] * s_rel + bonus
    paper["_s_timeliness"] = s_tl
    paper["_s_relevance"] = s_rel
    paper["_cluster_hits"] = count_cluster_hits(paper)

    if paper.get("is_patent"):
        print(f"  [Score-auto] {paper.get('title','')[:50]}... | TL={s_tl} REL={s_rel} PATENT | total={score:.0f}")
    else:
        print(f"  [Score-auto] {paper.get('title','')[:50]}... | TL={s_tl} REL={s_rel} | total={score:.0f}")
    return score


# ============================================================
# Diversity penalty
# ============================================================
def _char_bigrams(text: str) -> set:
    """Extract character bigrams for lightweight similarity check."""
    text = re.sub(r'[^a-z0-9]', '', text.lower())
    return {text[i:i+2] for i in range(len(text)-1)}


def compute_diversity_penalty(paper: dict, recent_titles: list) -> float:
    """
    Compute diversity penalty based on title similarity to recent recommendations.
    Uses Jaccard similarity on character bigrams as a lightweight proxy for embedding cosine similarity.
    Returns 0 (no penalty) to -4 (max penalty), matching criteria.yml thresholds.
    """
    title = paper.get("title", "")
    if not recent_titles or not title:
        return 0.0

    title_bigrams = _char_bigrams(title)
    if len(title_bigrams) < 5:
        return 0.0

    max_sim = 0.0
    for rt in recent_titles:
        rt_bigrams = _char_bigrams(rt)
        if len(rt_bigrams) < 5:
            continue
        intersection = len(title_bigrams & rt_bigrams)
        union = len(title_bigrams | rt_bigrams)
        if union > 0:
            sim = intersection / union
            max_sim = max(max_sim, sim)

    # Map Jaccard to penalty (Jaccard thresholds calibrated to approximate cosine >0.85 / >0.95)
    if max_sim > 0.45:   # proxy for cosine > 0.95
        return -4.0
    elif max_sim > 0.30:  # proxy for cosine > 0.85
        return -2.0
    return 0.0


# ============================================================
# Weighted total
# ============================================================
def compute_weighted_total(paper: dict, recent_titles: list = None) -> float:
    """
    Compute the full 8-dimension weighted score for a paper.
    Requires paper to have been scored by LLM (_llm_scores dict attached).
    """
    total = 0.0
    details = {}

    # Automatable criteria (already computed)
    s_tl = paper.get("_s_timeliness", score_timeliness(paper))
    s_rel = paper.get("_s_relevance", score_relevance(paper))
    total += WEIGHTS["timeliness"] * s_tl
    total += WEIGHTS["relevance"] * s_rel
    details["timeliness"] = s_tl
    details["relevance"] = s_rel

    # LLM-judged criteria
    llm = paper.get("_llm_scores", {})
    for crit in LLM_CRITERIA:
        s = llm.get(crit, LLM_DEFAULT)
        if not isinstance(s, (int, float)) or s < 0:
            s = LLM_DEFAULT
        s = max(0, min(5, s))
        total += WEIGHTS[crit] * s
        details[crit] = s

    # Patent bonus
    if paper.get("is_patent"):
        total += PATENT_BONUS

    # Diversity penalty
    if recent_titles:
        penalty = compute_diversity_penalty(paper, recent_titles)
        total += WEIGHTS["diversity"] * (5 + penalty) / 5.0  # scale to 0-5 range
        details["diversity_penalty"] = penalty
    else:
        total += WEIGHTS["diversity"] * 5  # full marks when no history
        details["diversity_penalty"] = 0

    paper["_weighted_total"] = total
    paper["_score_details"] = details
    return total


# ============================================================
# Recent titles for diversity
# ============================================================
def get_recent_recommended_titles(days: int = 5) -> list:
    """Get titles recommended in the last N working days, most recent first."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "papers")
    titles = []
    if not os.path.exists(base_dir):
        return titles

    today = datetime.now()
    recent_files = []
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".md") and fname[0].isdigit():
                fpath = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    dt = datetime.fromtimestamp(mtime)
                    if (today - dt).days <= days:
                        recent_files.append((dt, fpath))
                except Exception:
                    pass

    recent_files.sort(key=lambda x: x[0], reverse=True)
    for dt, fpath in recent_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("# ") and "每日" not in line and "推荐" not in line:
                    titles.append(line[2:].strip())
        except Exception:
            pass

    return titles


# ============================================================
# Main pipeline
# ============================================================
def run_daily():
    """Execute the daily recommendation pipeline with full 8-dimension scoring."""
    today = datetime.now()

    if not is_weekday(today):
        print(f"Today ({today.strftime('%A')}) is not a weekday. Skipping.")
        return

    print(f"=== Daily Paper Recommendation: {today.strftime('%Y-%m-%d')} ===")
    print()

    # ---- Phase 1: Search ----
    print(">>> Phase 1: Searching...")
    all_papers = run_search(verbose=True)

    if not all_papers:
        print("ERROR: No papers found. Check API connectivity.")
        return

    # ---- Phase 1.5: Relevance filter ----
    relevant = [p for p in all_papers if is_relevant(p)]
    print(f"\n>>> Phase 1.5: Relevance filter: {len(relevant)}/{len(all_papers)} papers pass keyword check")
    for p in all_papers:
        if p not in relevant:
            print(f"  ✗ FILTERED: {p.get('title', 'N/A')[:80]}")

    all_papers = relevant

    if not all_papers:
        print("ERROR: All papers filtered out. No relevant papers today.")
        return

    # ---- Phase 2: Dedup ----
    seen_titles = get_all_recommended_titles()
    print(f"\n>>> Phase 2: Dedup — {len(seen_titles)} previously recommended titles loaded")
    unique_papers = []
    for p in all_papers:
        if is_duplicate(p, seen_titles):
            print(f"  ✗ DUPLICATE: {p.get('title', 'N/A')[:80]}")
        else:
            unique_papers.append(p)
    print(f"  {len(unique_papers)}/{len(all_papers)} papers after dedup")
    all_papers = unique_papers

    if not all_papers:
        print("ERROR: All papers filtered as duplicates. No new papers today.")
        return

    # ---- Phase 3: Automatable scoring + narrow to top 6 ----
    print(f"\n>>> Phase 3: Automatable scoring ({len(all_papers)} candidates)...")
    for p in all_papers:
        compute_automatable_score(p)

    all_papers.sort(key=lambda x: x.get("_s_timeliness", 0) * WEIGHTS["timeliness"]
                               + x.get("_s_relevance", 0) * WEIGHTS["relevance"]
                               + (PATENT_BONUS if x.get("is_patent") else 0),
                    reverse=True)

    NARROW = min(6, len(all_papers))
    candidates = all_papers[:NARROW]
    print(f"  Narrowed to top {NARROW} by automatable scores")

    # ---- Phase 4: LLM Decomposition + scoring ----
    print(f"\n>>> Phase 4: LLM decomposition + scoring for {len(candidates)} candidates...")
    for i, paper in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] Decomposing: {paper.get('title', 'N/A')[:60]}...")
        note, scores = decompose_paper(paper, use_api=True)
        paper["_decompose_note"] = note
        paper["_llm_scores"] = scores
        # Store individual LLM scores for diagnostics
        for crit in LLM_CRITERIA:
            paper[f"_llm_{crit}"] = scores.get(crit, LLM_DEFAULT) if scores else LLM_DEFAULT

    # ---- Phase 5: Diversity penalty + weighted total ----
    print(f"\n>>> Phase 5: Diversity penalty + weighted ranking...")
    recent_titles = get_recent_recommended_titles(days=5)
    print(f"  Loaded {len(recent_titles)} recent titles for diversity check")

    for paper in candidates:
        compute_weighted_total(paper, recent_titles)

    candidates.sort(key=lambda x: x.get("_weighted_total", 0), reverse=True)

    # ---- Phase 6: Select top 3 ----
    selected = candidates[:3]

    # Print scoring summary
    print(f"\n>>> Scoring Summary ({len(candidates)} candidates):")
    print(f"  {'Rank':<5} {'Score':<8} {'Title'}")
    for i, p in enumerate(candidates):
        title = p.get("title", "N/A")[:70]
        total = p.get("_weighted_total", 0)
        details = p.get("_score_details", {})
        marker = "→" if p in selected else " "
        print(f"  {marker} {i+1:<3} {total:>6.0f}   {title}")
        # Print score breakdown
        parts = []
        for crit in ["timeliness", "relevance", "innovation", "experiment_generation",
                      "reproducibility", "practicality", "engineering_feasibility"]:
            s = details.get(crit, "?")
            parts.append(f"{crit[:4]}={s}")
        pen = details.get("diversity_penalty", 0)
        if pen < 0:
            parts.append(f"div={pen:.0f}")
        print(f"       {' ' * 8}  ({' | '.join(parts)})")

    # ---- Phase 7: Download PDFs (top 3 only) ----
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "archive", "pdfs",
                           today.strftime("%Y"), today.strftime("%m"))
    os.makedirs(pdf_dir, exist_ok=True)
    print(f"\n>>> Phase 7: Downloading PDFs for top {len(selected)} papers...")
    for p in selected:
        p["_pdf_path"] = download_pdf(p, pdf_dir)

    # ---- Phase 8: Write output ----
    print(f"\n>>> Phase 8: Writing output...")
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "papers",
        today.strftime("%Y"), today.strftime("%m")
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, today.strftime("%m%d") + ".md")

    # Guard against overwriting a previous run from the same day
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        backup_path = output_path.replace(".md", f"_backup_{today.strftime('%H%M')}.md")
        os.rename(output_path, backup_path)
        print(f"  [INFO] Existing {today.strftime('%m%d')}.md backed up to {os.path.basename(backup_path)}")

    total_candidates = len(all_papers)  # original pool size before narrowing

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 每日论文推荐 — {today.strftime('%Y年%m月%d日')}\n\n")
        f.write(f"> 搜索时间：{today.strftime('%H:%M UTC')}\n")
        f.write(f"> 候选论文：{total_candidates} 篇（已过滤无关结果）→ 评分精选 {len(selected)} 篇\n\n")

        # Add scoring summary
        f.write("## 评分明细\n\n")
        f.write("| # | 论文 | 总分 | TL | REL | INN | EG | REP | PRA | ENG | DIV |\n")
        f.write("|---|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|\n")
        for i, p in enumerate(candidates):
            title = p.get("title", "N/A")[:45]
            total = p.get("_weighted_total", 0)
            d = p.get("_score_details", {})
            marker = "⭐" if p in selected else ""
            f.write(f"| {marker}{i+1} | {title} | {total:.0f} | "
                    f"{d.get('timeliness','?')} | {d.get('relevance','?')} | "
                    f"{d.get('innovation','?')} | {d.get('experiment_generation','?')} | "
                    f"{d.get('reproducibility','?')} | {d.get('practicality','?')} | "
                    f"{d.get('engineering_feasibility','?')} | "
                    f"{d.get('diversity_penalty',0):.0f} |\n")
        f.write("\n> 评分维度：TL=时效性 REL=相关度 INN=创新启发 EG=实验生成 REP=可复现 PRA=实用性 ENG=工程可行 DIV=多样性惩罚\n")
        f.write("---\n\n")

        for i, paper in enumerate(selected):
            note = paper.get("_decompose_note", "")
            if note:
                f.write(note)
            else:
                f.write(f"# {paper.get('title', 'Unknown')}\n\n[LLM decomposition failed]\n\n")
            f.write("\n\n---\n\n")

    print(f"\n>>> Done! Output: {output_path}")
    for i, p in enumerate(selected):
        s = p.get("_weighted_total", 0)
        print(f"  {i+1}. [{s:.0f}] {p.get('title', 'N/A')[:100]}")


if __name__ == "__main__":
    run_daily()
