# 灵巧手磁触觉 — 每日论文推荐

每天自动推荐 3 篇与灵巧手磁触觉感知相关的论文/专利，附带结构化拆解。

## 工作流

```
Semantic Scholar + arXiv + 专利数据库
        ↓
  三轮搜索（精确匹配 → 互补方法 → 专利）
        ↓
  去重 + 多准则打分 + 多样性过滤
        ↓
  LLM 深度拆解 → 结构化笔记
        ↓
  papers/YYYY/MM/MMDD.md + GitHub Issue 通知
```

## 目录结构

```
├── criteria.yml            # 推荐评价准则（可量化规则）
├── knowledge_graph.yml     # 关键词簇、研究组、目标期刊
├── scripts/
│   ├── daily_rec.py        # 每日推荐主脚本
│   ├── search.py           # 搜索引擎
│   └── decompose.py        # LLM 论文拆解
├── papers/                 # 每日推荐笔记
├── verified/               # 已验证（可复现/不可行/已集成）
├── reports/                # 周报 + 月报
└── archive/pdfs/           # 论文PDF本地备份（.gitignore）
```

## 评价准则

| 准则 | 权重 | 评分方式 | 核心判断 |
|------|:--:|------|------|
| 相关度 | 25 | 自动（关键词簇命中数） | 命中几个关键词簇 |
| 创新启发 | 20 | LLM | 是否为灵巧手触觉提供方法启发 |
| 时效性 | 15 | 自动（年份） | 25-26年=5分, 23-24年=3分 |
| 实验生成 | 15 | LLM | 能否催生可验证实验 |
| 可复现性 | 15 | LLM | 是否有开源代码/完整参数 |
| 实用性 | 15 | LLM | 是否可在我们数据上验证 |
| 工程可行 | 5 | LLM | 计算资源是否在我们能力范围内 |
| 多样性 | 5 | 自动（bigram Jaccard） | 与近期推荐的标题相似度 |
| 专利bonus | +1 | 自动 | 每3工作日至少1篇专利 |

## 使用方式

```bash
# 手动运行今日推荐
python scripts/daily_rec.py

# 仅搜索不拆解
python scripts/search.py

# 查看帮助
python scripts/daily_rec.py --help
```

## 工作进度

- [x] 评价准则体系（8维加权 + experiment_generation）
- [x] 知识图谱（8关键词簇+9研究组+5种子论文）
- [x] 搜索引擎 (Semantic Scholar + arXiv 双源, 429重试)
- [x] LLM 拆解 + 自动评分 (DeepSeek 主 / Anthropic fallback)
- [x] GitHub Actions 双 cron 定时触发 (00:00 + 12:00 UTC)
- [x] 交叉日去重 + 多样性惩罚
- [ ] 周报/月报生成
- [ ] 专利来源恢复（Google Patents 移除后暂时缺失）
- [ ] 趋势分析 + 引用网络图
