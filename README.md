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

| 准则 | 权重 | 核心判断 |
|------|:--:|------|
| 相关度 | 30% | 命中几个关键词簇 |
| 时效性 | 20% | 25-26年=5分, 23-24年=3分 |
| 创新启发 | 20% | 是否为灵巧手触觉提供方法启发 |
| 可复现性 | 20% | 是否有开源代码 |
| 实用性 | 10% | 是否可在我们数据上验证 |

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

- [x] 评价准则体系
- [x] 知识图谱
- [x] 搜索引擎 (Semantic Scholar + arXiv + Google Patents + 国内专利)
- [x] LLM 拆解模板
- [ ] Claude API 自动拆解集成
- [ ] GitHub Actions 定时触发
- [ ] 周报/月报生成
- [ ] 趋势分析 + 引用网络图
