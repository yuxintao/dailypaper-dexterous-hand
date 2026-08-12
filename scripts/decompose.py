#!/usr/bin/env python3
"""
Paper decomposition using LLM API.
Supports DeepSeek (primary, OpenAI-compatible) and Anthropic Claude (fallback).

Usage:
  Set one of these environment variables:
    DEEPSEEK_API_KEY   (recommended - much cheaper, good Chinese support)
    ANTHROPIC_API_KEY  (fallback)
"""
import os
import json
import re
import sys
from datetime import datetime

# ============================================================
# Configuration - 支持 DeepSeek 和 Anthropic 双 provider
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# 模型选择
DEEPSEEK_MODEL = "deepseek-chat"  # 性价比最高，中文友好
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"  # fallback

# 自动选择 provider
if DEEPSEEK_API_KEY:
    API_PROVIDER = "deepseek"
    API_KEY = DEEPSEEK_API_KEY
elif ANTHROPIC_API_KEY:
    API_PROVIDER = "anthropic"
    API_KEY = ANTHROPIC_API_KEY
else:
    API_PROVIDER = None
    API_KEY = ""

# ============================================================
# 评分标准（与 criteria.yml §experiment_generation / innovation / reproducibility / practicality / engineering_feasibility 对齐）
# ============================================================
SCORING_RUBRIC = """
## 评分规则（严格按以下量表打分，只输出整数）

### innovation（创新启发，0-5分）
- 5: 全新触觉感知范式（弹性波/超声/摩擦电/量子隧道等）
- 4: 同类路线但重大改进（磁触觉+新解耦方法、传感器布局数学优化）
- 3: 同类路线的扎实工程改进（更好的标定、噪声抑制、域自适应）
- 2: 同类路线的微小修改
- 1: 光学/视觉触觉等与磁触觉关联弱的路线
- 0: 与磁触觉灵巧手完全无关
判断核心：论文能否为磁触觉灵巧手算法提供方法层面的创意启发？纯硬件设计、传感器布局、系统集成若无算法贡献，≤2分。

### experiment_generation（实验生成潜力，0-5分）
- 5: 论文提出明确可验证的机制/假设，读完后能直接写出实验脚本
- 4: 论文有具体方法（自回归预测/课程学习/对比预训练），翻译到我们框架需要工作但实验设计清晰
- 2: 仅有概念级启发或部分模块可转化，需要较大的创造性跳跃
- 0: 纯硬件/布局/系统集成，或仅有"触觉很重要"级别的一般性结论
判断核心：读完后我们能不能写出一个 train_4sensor_xxx.m？

### reproducibility（可复现性，0-5分）
- 5: GitHub上有可运行的代码
- 3: 无代码但有完整架构图+超参表
- 1: 仅有文字描述，无参数细节
- 0: 完全无法复现（专有硬件+无任何公开细节）

### practicality（实用性，0-5分）
- 5: 方法可直接在我们的18D磁场数据上验证
- 3: 需要适配（输入维度变化、需要额外传感器）
- 1: 需要我们不具备的特殊硬件
- 0: 完全无法在我们的数据上验证

### engineering_feasibility（工程可行性，0-5分）
- 5: 单GPU<1天，或纯CPU可运行
- 3: 单GPU 1-7天
- 1: 多GPU集群，超出我们能力
- 0: 无法判断
"""

# ============================================================
# Decomposition prompt
# ============================================================
SYSTEM_PROMPT = """你是一位磁触觉灵巧手技术文献分析专家。你的任务是根据论文信息，生成一份结构化的中文技术拆解笔记，并在末尾给出5个维度的自动评分。

## 分析规则
1. 始终用中文撰写（技术术语可保留英文）
2. "一句话核心贡献"严格 ≤50 个汉字
3. "与我们工作的关系"必须和"磁触觉 + MLP/CNN + 位置/力预测"这条技术路线做对比
4. "下一步行动"必须四选一并给出选择理由
5. 如果论文信息不足以做出判断，标注"[信息不足]"而非编造
6. 回复必须是完整的 markdown，以 "# {论文标题}" 开头
7. 末尾的「自动评分」部分必须严格按表格格式输出，每个维度给出整数分数+一句话理由""" + SCORING_RUBRIC

DECOMPOSE_PROMPT = """请根据以下论文信息生成结构化拆解笔记：

## 论文信息
- 标题: {title}
- 作者: {authors}
- 年份: {year}
- 发表处: {venue}
- 摘要: {abstract}
- TLDR: {tldr}
- 引用数: {citations}
- 来源: {source}

## 输出格式（严格遵循）

# {title}

> 推荐日期：{date} | 来源：{source} | 引用数：{citations}
> 推荐理由：[一句话说明为什么值得灵巧手磁触觉团队关注]

## 一句话核心贡献
[≤50汉字，说明这篇论文做了什么新东西]

## 与我们工作的关系
- **相同点**：[与磁触觉 MLP/CNN 位置-力预测路线的共同之处]
- **不同点**：[本质差异]
- **可借鉴的技术/思路**：[具体可用的方法或技巧]
- **可在我们数据上验证的方法**：[是否适合在我们 2/4/6 传感器 demo 上测试，为什么]

## 技术拆解
| 维度 | 内容 |
|------|------|
| 输入 | |
| 输出 | |
| 模型架构 | |
| 关键创新 | |
| 训练数据规模 | |
| 关键指标 | |
| 开源代码 | |
| 计算需求 | |

## 可复现性评估
- [ ] 有开源代码
- [ ] 有完整超参表
- [ ] 有数据集描述
- 复现难度：低 / 中 / 高

## 启发点
- [对我们的具体启发，1-3条；没有则写"暂无明显启发"]

## 下一步行动（四选一）
- [ ] 🟢 在我们的数据上复现
- [ ] 🟡 提取方法融入现有pipeline
- [ ] 🔵 作为related work记录
- [ ] ⚪ 暂不跟进

**选择理由**：[一句话]

## 自动评分
| 维度 | 分数 | 理由 |
|------|:--:|------|
| innovation | 0-5 | [一句话] |
| experiment_generation | 0-5 | [一句话] |
| reproducibility | 0-5 | [一句话] |
| practicality | 0-5 | [一句话] |
| engineering_feasibility | 0-5 | [一句话] |

---
*作者: {authors} | 发表: {year} | {venue}*
"""

# ============================================================
# Score parsing
# ============================================================
def parse_scores(markdown_text: str) -> dict:
    """
    Extract the 5 LLM-judged scores from the decomposition output.
    Returns a dict like {'innovation': 3, 'experiment_generation': 4, ...}
    Returns empty dict if parsing fails.
    """
    scores = {}
    # Look for the scoring table section
    # Pattern: | innovation | 0-5 | ... |
    table_start = markdown_text.find("## 自动评分")
    if table_start < 0:
        return scores

    table_text = markdown_text[table_start:]

    dimension_map = {
        "innovation": "innovation",
        "experiment_generation": "experiment_generation",
        "reproducibility": "reproducibility",
        "practicality": "practicality",
        "engineering_feasibility": "engineering_feasibility",
    }

    for line in table_text.split("\n"):
        line = line.strip()
        if not line.startswith("|") or "---" in line or "维度" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            dim = parts[1].strip()
            score_str = parts[2].strip()
            if dim in dimension_map:
                try:
                    scores[dim] = int(score_str)
                except ValueError:
                    pass

    return scores


# ============================================================
# Decomposition via DeepSeek API (OpenAI-compatible)
# ============================================================
def _call_deepseek(prompt: str) -> str:
    """Call DeepSeek API (OpenAI-compatible)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  [WARN] openai package not installed. pip install openai")
        return ""

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [ERROR] DeepSeek API call failed: {e}")
        return ""


# ============================================================
# Decomposition via Anthropic API
# ============================================================
def _call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude API (fallback)."""
    try:
        import anthropic
    except ImportError:
        print("  [WARN] anthropic package not installed. pip install anthropic")
        return ""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"  [ERROR] Anthropic API call failed: {e}")
        return ""


# ============================================================
# Main decompose function
# ============================================================
def decompose_paper(paper: dict, use_api: bool = True) -> tuple:
    """
    Decompose a paper into a structured markdown note.
    优先使用 DeepSeek API（更便宜，中文更好），fallback 到 Anthropic。

    Returns: (markdown_text, scores_dict)
        scores_dict: {'innovation': int, 'experiment_generation': int, ...} or {}
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Prepare paper info
    title = paper.get("title", "Unknown Title")
    authors_list = paper.get("authors", [])
    if isinstance(authors_list, list):
        authors = ", ".join([a.get("name", "") for a in authors_list[:5]])
    else:
        authors = str(authors_list)
    year = paper.get("year", "?")
    venue = paper.get("venue", paper.get("journal", "Unknown"))
    abstract = paper.get("abstract", "")
    if isinstance(abstract, dict):
        abstract = abstract.get("text", str(abstract))
    abstract = str(abstract)[:1200]
    tldr = paper.get("tldr", {})
    if isinstance(tldr, dict):
        tldr = tldr.get("text", "")
    tldr = str(tldr)[:300] if tldr else "（无）"
    citations = paper.get("citationCount", "?")
    source = paper.get("source", "unknown")

    # Try API decomposition
    if use_api and API_KEY:
        prompt = DECOMPOSE_PROMPT.format(
            title=title, authors=authors, year=year, venue=venue,
            abstract=abstract, tldr=tldr, citations=citations,
            source=source, date=date_str
        )

        result = ""
        if API_PROVIDER == "deepseek":
            print(f"  [API] Calling DeepSeek ({DEEPSEEK_MODEL})...")
            result = _call_deepseek(prompt)
        elif API_PROVIDER == "anthropic":
            print(f"  [API] Calling Anthropic ({ANTHROPIC_MODEL})...")
            result = _call_anthropic(prompt)

        if result:
            scores = parse_scores(result)
            if scores:
                print(f"  [Scores] {scores}")
            else:
                print(f"  [Scores] Failed to parse — using defaults")
            return result, scores
        else:
            print(f"  [WARN] API call returned empty. Falling back to template.")

    # Fallback: return a template for manual filling
    template = _generate_template(title, authors, year, venue, abstract, tldr, citations, source, date_str)
    return template, {}


def _generate_template(title, authors, year, venue, abstract, tldr, citations, source, date_str):
    """Generate a fill-in template (no API available)."""
    return f"""# {title}

> 推荐日期：{date_str} | 来源：{source} | 引用数：{citations}
> 推荐理由：[待填写]

## 一句话核心贡献
[待填写]

## 与我们工作的关系
- **相同点**：[待填写]
- **不同点**：[待填写]
- **可借鉴的技术/思路**：[待填写]
- **可在我们数据上验证的方法**：[待填写]

## 技术拆解
| 维度 | 内容 |
|------|------|
| 输入 | [待填写] |
| 输出 | [待填写] |
| 模型架构 | [待填写] |
| 关键创新 | [待填写] |
| 训练数据规模 | [待填写] |
| 关键指标 | [待填写] |
| 开源代码 | [待填写] |
| 计算需求 | [待填写] |

## 摘要
{abstract[:800] if abstract else '（无摘要）'}

## 可复现性评估
- [ ] 有开源代码
- [ ] 有完整超参表
- [ ] 有数据集描述
- 复现难度：[待评估]

## 启发点
- [待填写]

## 下一步行动（四选一）
- [ ] 🟢 在我们的数据上复现
- [ ] 🟡 提取方法融入现有pipeline
- [ ] 🔵 作为related work记录
- [ ] ⚪ 暂不跟进

**选择理由**：[待填写]

## 自动评分
| 维度 | 分数 | 理由 |
|------|:--:|------|
| innovation | - | [待LLM填写] |
| experiment_generation | - | [待LLM填写] |
| reproducibility | - | [待LLM填写] |
| practicality | - | [待LLM填写] |
| engineering_feasibility | - | [待LLM填写] |

---
*作者: {authors} | 发表: {year} | {venue}*
"""


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print(f"API Provider: {API_PROVIDER or 'None (template mode)'}")
    print(f"DeepSeek API Key: {'✓' if DEEPSEEK_API_KEY else '✗ (set DEEPSEEK_API_KEY env)'}")
    print(f"Anthropic API Key: {'✓' if ANTHROPIC_API_KEY else '✗ (fallback only)'}")

    # Test with a sample paper
    sample = {
        "title": "Test Paper: Magnetic Tactile Sensing for Robot Hands",
        "authors": [{"name": "Smith J"}, {"name": "Wang L"}],
        "year": 2026,
        "venue": "ICRA 2026",
        "abstract": "We present a novel magnetic tactile sensing approach...",
        "tldr": {"text": "Uses dipole model for self-supervised pretraining"},
        "citationCount": 15,
        "source": "semantic_scholar",
    }
    result, scores = decompose_paper(sample)
    print(result[:1000])
    if scores:
        print(f"\nParsed scores: {scores}")
