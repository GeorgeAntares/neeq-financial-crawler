# Changelog / 更新日志

本项目遵循语义化版本（SemVer）与 Keep a Changelog 规范。
All notable changes follow Semantic Versioning and Keep a Changelog.

## [Unreleased]

## [v0.3.0] - 2026-09-29

竞赛口径预处理与分问题分析收口：质量标记、营运资金 / 应计 / 行业杜邦，公司四段报告卡改为一页纸。后续三条轴写在 `ROADMAP.md`，等抉择。未升 1.0.0：仍是单期 195 家，解析也不是全量可用。  
Contest-style preprocess and question-led analysis: quality flags, working-capital / accruals / industry DuPont, and one-pager firm cards. Next three axes live in `ROADMAP.md`. Not 1.0.0: still a single-year 195-firm sample, and the parser is not fully re-exported.

### Changed / 变更

- 报表预处理改为质量标记，而不是再筛一批公司：资产负债表勾稽、毛利只认营业成本、负权益冻结 ROE/乘数、存货 0 与缺行分开、净利润截断 vs 无表、DSO/DIO/DPO 超过 730 天打帽 / Statement preprocessing is now quality flags (BS articulation, COGS-only gross margin, freeze ROE on non-positive equity, inventory 0 vs missing, truncation vs absent statement, DSO/DIO/DPO caps at 730 days)
- 行业分析改为营运资金周期（DSO/DIO/DPO/CCC）和应计利润；12 例「利润为正且 OCF 为负」分类降为附录 / Industry write-up is the working-capital cycle and accruals; the 12-positive cash-gap classifier is an appendix
- 杜邦改为分行业 ROE 分解（净利率 / 周转 / 杠杆）；恒等式和 PCA 作为核对与结构附录 / DuPont is an industry ROE decomposition; the identity check and PCA are supporting
- `financial_analysis.py` 默认 CSV 目录与公司指标库对齐为 `output/analysis/_csv_255` / Descriptive stats default to the same `_csv_255` folder as company metrics
- 行业画像与杜邦按竞赛口径重做：分问题完整个案、每个中位数带 n、IQR / 730 天 / 缩尾稳健对照；分析章节不套用八步顺序。软件 DSO 主口径长于制造，去掉 IQR 后方向会翻 / Industry portraits and DuPont now use contest-style caliber (complete-case by question, n per median, IQR / 730-day / winsor robustness) without forcing the eight-step order. Headline software DSO is longer than manufacturing; the ranking flips after dropping IQR outliers

### Added / 新增

- 后续方向 `ROADMAP.md`：数据 / 问题 / 呈现三条轴，选定后再施工 / `ROADMAP.md` lists three next axes (data, questions, presentation) to choose before more work
- 公司报告卡改为一页纸 tearsheet（`*_card.png`），行业总图 `industry_board.png`；`chart_catalog.py` 按四问归类并把 CSRC 门类从「其他」里拆开。默认不再写单家六张散图。 / Firm cards are one tearsheet; industry has a one-page board; `chart_catalog.py` files charts by the four questions and unpacks CSRC sectors inside 其他. The six-file firm set is no longer written by default.
- 报告卡截面图：单家利润瀑布、现金流三分类、杜邦三因子、同行条形、红旗色块、当期 KPI（`report_charts.py`）；行业画像拆成一问一图，图题写成结论；营运资金柱不再把缺失中位数画成 0。投资/筹资净额并进公司宽表。 / Cross-section charts for firm cards (P&L waterfall, OCF/ICF/FCF, DuPont three-factor, named peers, flag color table) and one-question industry figures with conclusion titles; WC bars no longer 0-fill missing medians. ICF/FCF join the company wide table.
- 首块合并表补抽货币资金、短债、其他应收、商誉、合同负债/预收、营业利润、三费、利息、销售商品收现；公司四段报告卡 `report_card.py`（偿债 / 盈利 / 营运 / 现金流），红旗：存贷双高、其他应收、商誉、本业比、收现率、利息保障 / Extra first-block line items and firm-level four-section report cards (`report_card.py`) with cash-and-debt, other-receivables, goodwill, core-profit, cash-conversion, and interest-cover flags
- `company_metrics_quality.csv` 以及 DSO、DIO、DPO、现金周期、应计/收入列 / Quality-flag summary plus DSO, DIO, DPO, CCC, and accruals-to-revenue
- `preprocess.py`：竞赛顺序的字段体检、缺失机制表、IQR 离群计数、1%/99% 缩尾对照、Z-score/Min-Max 副本和处理前后图 / Contest-style preprocess log: field exam, missing-mechanism table, IQR outlier counts, winsorize comparison, scaling copy, before/after chart
- 行业覆盖表、稳健表与对照图：`industry_coverage.csv` / `industry_sensitivity.csv` / `industry_sensitivity.png`，杜邦 `dupont_coverage.csv` / `dupont_sensitivity.csv` / Coverage and robustness tables for industry portraits and DuPont

## [v0.2.0] - 2026-09-24

解析器定位与附注列修复之后，分析从描述统计扩到公司级指标、行业画像、杜邦/PCA 和现金缺口分类。默认数据源改为 NEEQ，并加上 GitHub Actions。
Parser locator and footnote-column fixes; analysis expands from descriptive stats to company metrics, industry portraits, DuPont/PCA, and a cash-gap classifier. Default source is NEEQ; CI added.

### Added / 新增

- 公司级指标库：`company_metrics.py` 从三大表第一张（合并）表生成一家一行宽表，含毛利率、净利率、杜邦、偿债、营运、OCF 与同比；比率另有 1%/99% 截尾列 / Company-level metrics wide table (`company_metrics.py`) from the first consolidated block of each statement, with DuPont, liquidity, working-capital, OCF, YoY, and winsorized ratio columns
- 指标数据字典 `company_metrics_dictionary.md`（公式与科目来源）/ Data dictionary for those columns
- 行业画像：`industry_portrait.py` 把 PDF 子目录收成制造 / 软件信息 / 其他，输出中位数、箱线图和「利润为正且 OCF 为负」占比 / Industry portraits (manufacturing / software / other) with medians, boxplots, and profit-positive-but-OCF-negative shares
- 杜邦核对 + Spearman 相关阵 + numpy SVD 主成分（规模 / 杠杆 / 现金），见 `dupont_pca.py` / DuPont identity, Spearman correlations, and SVD PCA (scale / leverage / cash)
- 现金缺口分类：`cash_gap_model.py` 用资产负债/利润表比率预测「利润为正且 OCF 为负」，随机森林对照逻辑回归，分层 5 折 + SHAP / Cash-gap classifier (profit>0 and OCF<0) with RF vs logit, stratified 5-fold CV, and SHAP
- 分析报告 `ANALYSIS_REPORT.md`：数据清洗、指标、行业画像、杜邦/PCA、现金缺口分类与局限；README 改为以该报告为封面结论 / Analysis report covering cleaning, metrics, industry portraits, DuPont/PCA, cash-gap classification, and limits

### Fixed / 修复

- 「半年度报告」因包含「年度报告」子串被当成年报下载；现过滤半年报 / 已取消 / 摘要 / 季报 / Half-year titles matched `年度报告` as a substring; they are now excluded with cancelled, summary, and quarterly reports
- 文本层起始页会命中管理层分析 / 审计封面，利润表标题与资产负债表尾巴同页时还会定晚一页；现要求「合并资产负债表+货币资金」，利润表允许标题页，附注封面误判不再中断扫描 / Start-page detector skips MD&A and audit covers; income statement may start on a title-only page
- 解析器将「附注」列（注释31 / 五、32）截成金额列，利润表页还会吞进资产负债表尾巴；现改为压缩空列、丢弃附注、按报表类型认主表 / Drop footnote columns and leftover balance-sheet tables instead of treating them as amounts
- 现金流量表页范围会扫到财务报表附注；现每张表最多 6–8 页，并在「年度财务报表附注」处停止定位 / Cap statement page spans and stop at the notes heading
- 资产负债表表头常用「2025年12月31日」而非「期末余额」，导致主表定位失败、误命中附注页 / Recognise year-end date headers when locating the balance sheet
- 利润表第二轮定位在未赋值 `sorted_pages` 时会 `NameError`，导致整份 PDF 解析失败 / Income-statement fallback used `sorted_pages` before it was assigned, aborting the whole PDF
- 三级解析全部失败时仍可能导出低质量表；现改为显式丢弃（宁可缺数据，不可存假数据）/ Failed three-tier parses are discarded instead of exporting low-quality tables
- 利润表增加「营业收入/营业总收入须带有效数值」校验，减少附注表被当成主表 / Income statements now require a numeric revenue row
- 单份 PDF 解析异常不再中断整批任务；进程失败以非零退出码结束 / One bad PDF no longer stops the batch; crashes exit non-zero
- 分析模块按主表行去重，避免「其中：营业收入」等附注明细重复计数 / Analysis picks primary rows so footnote lines are not double-counted
- 随机森林标准化改为仅在训练集 `fit`；全量预测改用交叉验证，避免乐观偏差 / Scaler fits on train only; full-set predictions use cross-validation

### Changed / 变更

- `financial_analysis.py` 清洗过小营收、优先营业成本、毛利率截尾；图输出为 `financial_analysis_clean.png` / Descriptive stats drop tiny revenue, prefer COGS, clip gross-margin mean; chart is `financial_analysis_clean.png`
- 分析 / ML / DL / SHAP 脚本加上 `if __name__ == '__main__'`，现金流特征抽到 `cashflow_features.py` / Analysis scripts no longer run on import; shared cash-flow features live in `cashflow_features.py`
- DL：去掉写死的 `C:\\torch`；StandardScaler 只在训练集 fit；early stopping 用训练子集验证，不再盯测试集 / Drop hardcoded `C:\\torch`; scaler fits on train; early stopping uses a val split, not the test set
- `reexport_csvs.py` 支持 `--limit` / Support `--limit` for sampled re-exports
- 默认 `--source` 改为 `neeq`；巨潮下载改为 `%PDF-` 头校验、`.part` 原子写入、跳过已有文件，导出年份用标题推断的报告年度 / Default source is NEEQ; CNINFO downloads validate PDF headers, write atomically, skip existing files, and use inferred report year
- 增加 GitHub Actions：`unittest discover`（不安装 torch）/ Add GitHub Actions running unit tests without torch
- README 拆成英文 `README.md` 与简体中文 `README.zh-CN.md`，内容与当前默认 NEEQ / 解析器 / CI 对齐 / Split docs into English `README.md` and Simplified Chinese `README.zh-CN.md`
- `.gitignore` 增加 `.trae/`，不提交 IDE 规则目录 / Ignore `.trae/` IDE metadata

## [v0.1.0] - 2026-08-27 正式版 / Stable Release

### Added / 新增

- 数据分析模块（pandas + matplotlib）：营收、毛利率、现金流统计分析与可视化 / Data analysis module (pandas + matplotlib): revenue, gross margin, and cash flow statistics with visualisation
- 三级 PDF 解析引擎：pdfplumber → PyMuPDF → RapidOCR OCR 回退，解决复杂排版与图片型表格解析失败 / Three-tier PDF parsing engine with RapidOCR fallback for complex layouts and image-based tables
- 机器学习财务健康度分类：RandomForest，542 家企业、7 维现金流特征 / ML financial health classification (RandomForest, 542 companies, 7 cash-flow features)
- 深度学习财务健康度分类：PyTorch MLP（7→64→32→1，早停 + Dropout）/ DL financial health classification (PyTorch MLP with early stopping + Dropout)
- 模型可解释性分析：SHAP 特征重要性 + 方向性财务解读（外部融资依赖为最强负向信号）/ Model interpretability with SHAP: feature importance and directional financial insights
- 严谨化评估：5 折 Stratified 交叉验证 + ROC-AUC / Precision-Recall 指标（AUC ≈ 0.62）/ Rigorous evaluation with 5-fold Stratified Cross-Validation and ROC-AUC / PR metrics (AUC ≈ 0.62)

### Changed / 变更

- README 全面中英双语化，新增数据分析 / 机器学习 / 深度学习 / 可解释性章节 / README fully bilingual with analysis, ML, DL, and interpretability sections
- 依赖更新：新增 rapidocr_onnxruntime、torch、shap / Dependencies updated: rapidocr_onnxruntime, torch, shap

## [v0.1.0-alpha.1] - 2026-08-07

### Added / 新增

- SQLite 持久化与日期范围查询（`crawl_state.db`），解决公告页码漂移问题 / SQLite persistence with date-range querying, eliminating announcement page drift
- 断点续爬：启动时自动恢复中断下载、WAL 模式 + busy_timeout / Resume support with automatic recovery of interrupted downloads
- 端到端冒烟测试 / End-to-end smoke test

### Fixed / 修复

- `_has_complete_csvs` 增加年份参数，修复多年度公告误跳过 / Added year parameter to prevent false skipping of multi-year announcements

## 2026-08-05 项目初始化 / Project Initiation

- 新三板与巨潮资讯网年报爬虫骨架：公告搜索、PDF 下载、CSV 导出 / NEEQ & CNINFO annual report crawler skeleton: announcement search, PDF download, CSV export
