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
# Decomposition prompt
# ============================================================
SYSTEM_PROMPT = """你是一位磁触觉灵巧手技术文献分析专家。你的任务是根据论文信息，生成一份结构化的中文技术拆解笔记。

## 分析规则
1. 始终用中文撰写（技术术语可保留英文）
2. "一句话核心贡献"严格 ≤50 个汉字
3. "与我们工作的关系"必须和"磁触觉 + MLP/CNN + 位置/力预测"这条技术路线做对比
4. "下一步行动"必须四选一并给出选择理由
5. 如果论文信息不足以做出判断，标注"[信息不足]"而非编造
6. 回复必须是完整的 markdown，以 "# {论文标题}" 开头"""

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

---
*作者: {authors} | 发表: {year} | {venue}*
"""

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
def decompose_paper(paper: dict, use_api: bool = True) -> str:
    """
    Decompose a paper into a structured markdown note.
    优先使用 DeepSeek API（更便宜，中文更好），fallback 到 Anthropic。
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
            return result
        else:
            print(f"  [WARN] API call returned empty. Falling back to template.")

    # Fallback: return a template for manual filling
    return _generate_template(title, authors, year, venue, abstract, tldr, citations, source, date_str)


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
    result = decompose_paper(sample)
    print(result[:1000])
