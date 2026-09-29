<p align="center">
  <a href="https://github.com/GeorgeAntares/neeq-financial-data-pipeline/actions/workflows/test.yml"><img src="https://github.com/GeorgeAntares/neeq-financial-data-pipeline/actions/workflows/test.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/Version-0.2.0-blue?logo=git&logoColor=white" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<h1 align="center">NEEQ Financial Data Pipeline</h1>

<p align="center"><strong>English</strong> | <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
Crawl NEEQ (and optionally CNINFO) annual-report PDFs, parse the three primary financial statements, export CSV, and run company-level financial analysis.
</p>

## Features

- Search NEEQ annual reports by **date range** (avoids page-drift when new filings appear)
- SQLite download state machine with resume, `%PDF-` validation, and atomic `.part` writes
- Filter out summaries, cancelled filings, half-year and quarterly reports
- Three-tier PDF parser: pdfplumber → PyMuPDF → RapidOCR
- Text-layer start-page detection (skip MD&A / audit covers; drop footnote columns)
- Export consolidated balance sheet, income statement, and cash-flow statement as CSV
- Company-level ratios, three-group industry portraits, DuPont/PCA, and a cash-gap classifier (profit>0 and OCF<0)

Default data source is **NEEQ**. Pass `--source cninfo` for the CNINFO (巨潮) path.

## Requirements

- Python 3.9+
- Windows, macOS, or Linux

```bash
git clone https://github.com/GeorgeAntares/neeq-financial-data-pipeline.git
cd neeq-financial-data-pipeline
pip install -r requirements.txt
```

`requirements.txt` includes optional ML/OCR stacks (`torch`, `shap`, `rapidocr_onnxruntime`). For crawl + parse only, `requests`, `pdfplumber`, `pymupdf`, and `pandas` are enough.

## Quick start

```bash
# Download 2025 NEEQ annual-report PDFs (no CSV parse)
python main.py --start-date 2025-01-01 --end-date 2025-12-31 --skip-parse

# Search into SQLite only
python main.py --year 2025 --skip-download

# Download + parse
python main.py --year 2025
```

Re-running the same command skips PDFs already marked `downloaded` in `output/crawl_state.db`.

`--resume` only imports a legacy `announcements_list.csv` into SQLite:

```bash
python main.py --resume
```

Stop after the current company by creating `STOP.txt` in the repo root (`New-Item STOP.txt` on PowerShell, `touch STOP.txt` elsewhere).

Rebuild CSVs from on-disk annual-report PDFs (skips half-year / cancelled):

```bash
python reexport_csvs.py
```

## CLI

| Flag | Meaning | Default |
|------|---------|---------|
| `--source` | `neeq` or `cninfo` | `neeq` |
| `--year` | Announcement **publish** year (not always the report year) | last calendar year |
| `--start-date` / `--end-date` | Inclusive query window `YYYY-MM-DD` | none |
| `--max-pages` | Cap search pages; `0` = no cap | `0` |
| `--start-page` | Recovery offset into search results | `1` |
| `--skip-parse` | Download PDFs only | off |
| `--skip-download` | Search / persist metadata only | off |
| `--resume` | Import legacy announcement CSV into SQLite | off |

On the CNINFO path, files are still `%PDF-`-checked and written atomically. Export year is inferred from the title (e.g. a 2025 filing titled “2024 Annual Report” is stored as 2024).

## SQLite state

NEEQ crawl state lives in `output/crawl_state.db` (WAL). Unique key: `(source, pdf_url)`.

```
pending → downloading → downloaded
              ↓
            failed
```

Stale `downloading` rows are reset to `pending` on startup.

**`announcements`**: `source`, `pdf_url`, `company_code`, `company_name`, `title`, `report_year`, `publish_date`, `status`, `file_path`, `file_size`, `attempts`, `last_error`, `sha256`.

**`crawl_runs`**: query window, `discovered_count` / `downloaded_count` / `failed_count`, `started_at` / `finished_at`.

## Parsing

```
pdfplumber (text layer) → PyMuPDF (fallback text) → RapidOCR (page images)
```

The locator looks for `合并资产负债表` + `货币资金`, allows income/cash-flow **title-only** pages, and stops at the “YYYY年度财务报表附注” cover. Footnote columns such as `注释31` / `五、32` are dropped. Failed statements are discarded instead of writing fake rows.

On a 36-PDF stratified sample after the locator fix: **100%** had at least one usable statement, **92%** had all three, income statement **97%**. Remaining misses are mostly image-only pages (OCR not installed) or unusual layouts.

## Analysis and models

Full write-up: [`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md). Metric formulas: [`company_metrics_dictionary.md`](company_metrics_dictionary.md).

On the current-parser ~255 CSV set (first consolidated block only, revenue ≥ 100,000 CNY). **Preprocessing is four layers**, not one filter: parse truncation, the revenue floor, statement-quality flags, then winsorizing.

- **195** firms. Gross margin uses 营业成本 only (187 valid, median 26.0%). Negative equity (7 firms) freezes ROE. Balance sheets articulate on 158 firms; one relative gap exceeds 1%. Missing net profit on 70 firms is a truncated first table.
- Working capital is complete-case by question. Headline software DSO 179 days (n=24) vs manufacturing 105 (n=80); dropping IQR outliers flips it to 97 vs 102 — the long cycle is a right tail. Manufacturing DIO is only slightly higher (113 vs 100). Software is missing an inventory *row* in 23% of firms, not reporting inventory of zero.
- Earnings quality: median (NI − OCF)/revenue −4.5% (n=109). Among firms with both items, 21 have NI<0 and OCF>0 vs 12 with NI>0 and OCF<0. Manufacturing / software accrual medians do not move after IQR.
- Industry DuPont (109 valid-ROE firms): manufacturing 4.4%, software 0.5%, driven by net margin. After IQR the gap remains (5.9% vs 2.6%). After freezing negative equity, Spearman(ROE, net margin) is 0.90.
- Appendix: the 12-positive cash-gap classifier has little signal (logit ROC 0.53). The older “OCF>0 and cash increase>0” scripts remain a control.

```bash
python company_metrics.py --csv-dir output/analysis/_csv_255
python preprocess.py              # field exam, missing mechanism, IQR, before/after
python industry_portrait.py
python dupont_pca.py
python cash_gap_model.py          # needs scikit-learn; shap optional
python financial_analysis.py --csv-dir output/analysis/_csv_255
python ml_financial_health.py     # control: OCF>0 and cash increase>0
python ml_evaluation.py
python shap_analysis.py
python dl_financial_health.py     # appendix MLP
python csv_to_pdf.py
```

`company_metrics.py` writes statement-quality flags (articulation, COGS-only gross margin, frozen ROE, inventory 0 vs missing, DSO/DIO caps). `preprocess.py` follows the contest cleaning order (field exam, missing mechanism, IQR counts, 1%/99% winsorize) on a copy of the wide table. Industry portraits and DuPont use the same caliber without the eight-step order: complete-case by question, medians with n, and IQR / 730-day / winsor robustness. `financial_analysis.py` defaults to `_csv_255`; the 100,000 CNY floor is sample screening. Charts use new filenames (`financial_analysis_clean.png`) so Windows Explorer does not keep an old CreationTime.

The older “financial health” scripts still use **OCF>0 and net cash increase>0** with cash-flow line items (542 firms, 5-fold ROC-AUC ≈ 0.62). That is a control experiment, not a credit rating.

Outputs under `output/analysis/` are gitignored. Committed notes: `preprocess.md`, `industry_portrait.md`, `dupont_pca.md`, `cash_gap_model.md`.

## Layout

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

CSV columns: item + two amount columns (`期末余额`/`期初余额` or `本期金额`/`上期金额`). Outputs go under `output/` (gitignored): `crawl_state.db`, `pdf/`, `csv/`, `analysis/`, `log/`.

```bash
python -m unittest discover -s tests
```

CI runs the same command on Python 3.11 (without installing `torch`).

## Data sources

- [NEEQ disclosure](https://www.neeq.com.cn/m/disclosure/announcement.html)
- [CNINFO](https://www.cninfo.com.cn/)

## Limitations

- A few scanned / image-only statements still need RapidOCR; without it those tables are skipped
- Footnotes can still leak into a statement; analysis prefers primary rows and skips `其中：` lines
- On-disk CSVs are a mix of parser generations until you re-export
- Analysis is a single-year cross-section, manufacturing-heavy, with no default labels
- OCR is ~1–2 s/page

## License

MIT. See [LICENSE](LICENSE).
