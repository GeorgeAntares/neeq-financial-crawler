<p align="center">
  <a href="https://github.com/GeorgeAntares/neeq-financial-data-pipeline/actions/workflows/test.yml"><img src="https://github.com/GeorgeAntares/neeq-financial-data-pipeline/actions/workflows/test.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/Version-0.2.0-blue?logo=git&logoColor=white" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<h1 align="center">NEEQ 财报数据采集与分析流水线</h1>

<p align="center"><a href="README.md">English</a> | <strong>简体中文</strong></p>

<p align="center">
从全国股转系统（可选巨潮资讯网）抓取年报 PDF，解析三大财务报表，导出 CSV，并做公司级财务分析。
</p>

## 功能

- 按**日期范围**搜索新三板年报，避免新公告把页码挤乱
- SQLite 下载状态机：断点续爬、`%PDF-` 校验、`.part` 原子写入
- 过滤摘要、已取消、半年报、季报
- 三级 PDF 解析：pdfplumber → PyMuPDF → RapidOCR
- 文本层起始页定位（跳过管理层分析 / 审计封面，丢掉附注列）
- 导出合并资产负债表、利润表、现金流量表 CSV
- 公司级指标、三类行业画像、杜邦 / PCA、公司四段报告卡，以及「利润为正且 OCF 为负」分类

默认数据源是 **NEEQ**。巨潮请加 `--source cninfo`。

## 环境

- Python 3.9+
- Windows / macOS / Linux

```bash
git clone https://github.com/GeorgeAntares/neeq-financial-data-pipeline.git
cd neeq-financial-data-pipeline
pip install -r requirements.txt
```

`requirements.txt` 含可选的 ML/OCR（`torch`、`shap`、`rapidocr_onnxruntime`）。只爬取和解析的话，`requests`、`pdfplumber`、`pymupdf`、`pandas` 即可。

## 快速开始

```bash
# 下载 2025 年新三板年报 PDF（不解析 CSV）
python main.py --start-date 2025-01-01 --end-date 2025-12-31 --skip-parse

# 只把公告写入 SQLite，不下载
python main.py --year 2025 --skip-download

# 下载并解析
python main.py --year 2025
```

再次执行相同命令时，会跳过 `output/crawl_state.db` 里已是 `downloaded` 的 PDF。

`--resume` 只用于把旧的 `announcements_list.csv` 导入 SQLite：

```bash
python main.py --resume
```

在项目根目录创建 `STOP.txt`，当前这家公司处理完后退出（PowerShell：`New-Item STOP.txt`）。

从磁盘上年报 PDF 重导出 CSV（自动跳过半年报 / 已取消）：

```bash
python reexport_csvs.py
```

## 命令行

| 参数 | 说明 | 默认 |
|------|------|------|
| `--source` | `neeq` 或 `cninfo` | `neeq` |
| `--year` | 公告**发布**年份，不一定等于报告年度 | 去年 |
| `--start-date` / `--end-date` | 查询区间 `YYYY-MM-DD` | 无 |
| `--max-pages` | 搜索页数上限，`0` 不限 | `0` |
| `--start-page` | 搜索结果偏移（故障恢复） | `1` |
| `--skip-parse` | 只下载不解析 | 关 |
| `--skip-download` | 只搜索 / 写元数据 | 关 |
| `--resume` | 导入旧公告 CSV | 关 |

巨潮路径同样校验 PDF 头、原子写入。导出年份从标题推断（2025 年发布的「2024年年度报告」会标成 2024）。

## SQLite 状态库

NEEQ 爬取状态在 `output/crawl_state.db`（WAL）。唯一键：`(source, pdf_url)`。

```
pending → downloading → downloaded
              ↓
            failed
```

启动时会把残留的 `downloading` 重置为 `pending`。

**`announcements`**：`source`、`pdf_url`、`company_code`、`company_name`、`title`、`report_year`、`publish_date`、`status`、`file_path`、`file_size`、`attempts`、`last_error`、`sha256`。

**`crawl_runs`**：查询区间、`discovered_count` / `downloaded_count` / `failed_count`、`started_at` / `finished_at`。

## 解析

```
pdfplumber（文本层）→ PyMuPDF（备用文本）→ RapidOCR（页面截图）
```

定位要求「合并资产负债表 + 货币资金」，利润表 / 现金流量表允许本页只有标题，遇到「YYYY年度财务报表附注」封面即停。会丢掉 `注释31` / `五、32` 这类附注列。解析失败的表直接丢弃，不写假数据。

定位修好后，36 份分层抽样：**至少一张表 100%**，**三张都可用 92%**，利润表 **97%**。剩下多半是图片型页面（本机未装 OCR）或特殊排版。

## 分析与模型

完整报告：[`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md)。指标公式：[`company_metrics_dictionary.md`](company_metrics_dictionary.md)。

新解析器约 255 套 CSV（只取第一张合并表，营收 ≥ 10 万元）。**预处理不是一次筛选**：解析截断、营收门槛、报表质量标记、分位截尾是四层。

- **195** 家。毛利只认营业成本（187 家可算，中位数 26.0%）；负权益 7 家冻结 ROE；资产负债表 158 家勾稽，只有 1 家相对差距 > 1%。净利润缺 70 家是第一张表截断。
- 营运资金按问题完整个案：主口径软件 DSO 179 天（n=24）、制造 105 天（n=80）；去掉 IQR 离群后变成 97 对 102，长账期是右尾。制造 DIO 只略高（113 vs 100）。软件 23% 缺存货行，不是存货为 0。
- 盈余质量：应计/收入中位数 −4.5%（n=109）。利润<0 且 OCF>0 有 21 家，利润>0 且 OCF<0 只有 12 家。IQR 后制造 / 软件应计中位数不动。
- 行业杜邦（有效 ROE 109 家）：制造 4.4%，软件 0.5%，差在净利率。IQR 后仍是制造更高（5.9% vs 2.6%）。冻掉负权益后 ROE 与净利率 Spearman 0.90。
- 公司报告卡（怎么赚钱 / 利润真不真 / 会不会被困住 / 当年营运与现金）。首块补抽：货币资金 173/195、销售商品收现 158/195、营业利润 126/195。最常见红旗是利息保障偏弱（54/107）和制造收现率偏低（21/90）；存贷双高、其他应收、商誉很少触发。
- 附录：12 例现金缺口分类几乎没有信号（logit ROC 0.53）；旧「OCF>0 且现金净增加>0」仍是对照。

```bash
python company_metrics.py --csv-dir output/analysis/_csv_255
python preprocess.py              # 字段体检、缺失机制、IQR、处理前后图
python industry_portrait.py
python dupont_pca.py
python cash_gap_model.py          # 需要 scikit-learn；shap 可选
python report_card.py             # 公司四段报告卡
python financial_analysis.py --csv-dir output/analysis/_csv_255
python ml_financial_health.py     # 对照：OCF>0 且现金净增加>0
python ml_evaluation.py
python shap_analysis.py
python dl_financial_health.py     # 附录 MLP
python csv_to_pdf.py
```

`company_metrics.py` 做报表质量标记（勾稽、毛利口径、负权益、存货 0 vs 缺失、DSO/DIO 帽）。`preprocess.py` 按竞赛顺序做字段体检、缺失机制、IQR 计数和 1%/99% 缩尾对照，PDF/CSV 原件不动。行业画像和杜邦跟同一套口径，但不套八步顺序：分问题完整个案、中位数带 n、IQR / 730 天 / 缩尾稳健对照。`financial_analysis.py` 默认读 `_csv_255`，营收低于 10 万元是样本筛选，毛利率图截到 [-50%, 80%]。图用新文件名（`financial_analysis_clean.png`），避免 Windows 资源管理器仍显示旧的创建时间。

旧脚本仍用 **OCF>0 且现金净增加>0** 加现金流科目（542 家，5 折 ROC-AUC ≈ 0.62）。那是对照实验，不是信用评级。

`output/analysis/` 已 gitignore。仓库里的文字底稿：`preprocess.md`、`industry_portrait.md`、`dupont_pca.md`、`cash_gap_model.md`、`company_report_cards.md`。

## 目录

```
neeq-financial-data-pipeline/
├── main.py
├── neeq_crawler.py
├── cninfo_api.py
├── database.py
├── pdf_parser.py
├── data_exporter.py
├── reexport_csvs.py
├── financial_analysis.py
├── company_metrics.py
├── company_metrics_dictionary.md
├── preprocess.py
├── preprocess.md
├── industry_groups.py
├── industry_portrait.py
├── industry_portrait.md
├── dupont_pca.py
├── dupont_pca.md
├── cash_gap_model.py
├── cash_gap_model.md
├── report_card.py
├── company_report_cards.md
├── ANALYSIS_REPORT.md
├── ANALYSIS_README.md
├── ml_financial_health.py
├── ml_evaluation.py
├── shap_analysis.py
├── dl_financial_health.py
├── csv_to_pdf.py
├── retry_backup_pdfs.py
├── smoke_test.py
├── config.py
├── tests/
├── .github/workflows/test.yml
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

CSV 列为「项目 + 两列金额」（期末/期初或本期/上期）。产物在 `output/`（已 gitignore）：`crawl_state.db`、`pdf/`、`csv/`、`analysis/`、`log/`。

```bash
python -m unittest discover -s tests
```

CI 在 Python 3.11 上跑同一命令（不安 `torch`）。

## 数据源

- [全国股转系统信息披露](https://www.neeq.com.cn/m/disclosure/announcement.html)
- [巨潮资讯网](https://www.cninfo.com.cn/)

## 已知局限

- 少数扫描件 / 图片表仍需 RapidOCR；没装 OCR 时这些表会被跳过
- 附注仍可能混进报表；分析会优先主表行并跳过「其中：」明细
- 磁盘上的 CSV 可能是新旧解析器混着的，需要重导出才会统一
- 分析是单期截面，制造业偏多，没有违约标签
- OCR 大约 1–2 秒/页

## 许可

MIT，见 [LICENSE](LICENSE)。
