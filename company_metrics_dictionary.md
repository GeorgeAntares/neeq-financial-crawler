# 公司级指标数据字典 / Company Metrics Data Dictionary

生成：`python company_metrics.py --csv-dir output/analysis/_csv_255`  
清洗日志：`python preprocess.py --metrics output/analysis/company_metrics.csv`  
产物：`company_metrics.csv`（一家一年一行）、`company_metrics_quality.csv`、`preprocess_field_exam.csv` / `preprocess_missing.csv` / `preprocess_outliers.csv`

样本默认是新解析器导出的约 255 家 CSV。只读每个文件的**第一张表**（合并报表）；同一 CSV 后半段的母公司表或其它年份表不参与计算。科目缺失记为 NaN，不用后一张表补。

金额单位：人民币元。比率与同比均为小数（`0.218` = 21.8%），流动比率、权益乘数、总资产周转率为倍数。DSO / DIO / DPO / CCC 单位为天。

## 预处理不是一件事

仓库里常说的「数据预处理」其实叠了四层，不要混成一步筛选：

| 层 | 做什么 | 不做什么 |
|----|--------|----------|
| 解析 | PDF→CSV；只取第一张合并表 | 不把母公司净利润拼到合并营收上 |
| 样本筛选 | `revenue >= 100_000`（丢掉附注编号进金额列） | 不按新三板 / 创业板代码再筛，那是样本范围 |
| **报表质量** | 资产负债表勾稽、毛利只认营业成本、负权益冻结 ROE/乘数、存货 0 与缺失分开、净利润截断 vs 无表、DSO/DIO 异常帽 | 不是再丢掉一批公司，而是打标记、无效比率记 NaN |
| 统计稳健 | `*_w` 为 1% / 99% 分位截尾，给画像和模型用 | 核对单家公司看无 `_w` 的原始列 |

竞赛八步（备份、体检、缺失标记、按机制处理缺失、IQR/缩尾、去重、无量纲化、处理前后图）由 `preprocess.py` 落在宽表副本上，日志见 `preprocess.md`。本文件的「清洗」仍指第三层报表质量；营收门槛是第二层。

## 报表质量规则

| 规则 | 说明 |
|------|------|
| 毛利口径 | `cogs` 只取「营业成本」（含「其中：营业成本」）。「营业总成本」进 `total_operating_cost`，**不**拿去算毛利率。`gm_valid` 为真才有 `gross_margin`。 |
| 负权益 | `equity < 0` 或平均权益 ≤ 0 时 `equity_negative`；`roe`、`equity_multiplier`、`dupont_product` 记为 NaN。 |
| 资产负债表勾稽 | `bs_gap = 资产总计 − (负债合计 + 所有者权益)`；相对差距 ≤ 1% 为 `bs_articulation_ok`。权益若是归母口径，勾稽可能对不上（少数股东权益）。 |
| 存货 0 vs 缺失 | `inventory_status`：`positive` / `zero` / `missing`。软件企业存货为 0 是真实结构，缺「存货」行才是缺失。 |
| 净利润截断 | 有营收、第一张利润表却没有净利润行 → `np_truncated`。这是解析截断，不是公司没披露。 |
| 现金流缺失 | `ocf_missing_kind`：`present` / `truncated`（有现金流量表但无经营净额行）/ `no_statement`。 |
| 周转天数帽 | DSO / DIO / DPO > 730 天或为负记 `*_anomalous`，行业中位数不用这些点。 |

## 标识

| 列 | 含义 | 来源 |
|----|------|------|
| `stock_code` | 证券代码 | 文件名 `{code}_{name}_{year}_合并*.csv` |
| `company_name` | 公司简称 | 文件名 |
| `year` | 报告年度 | 文件名 |
| `revenue_item` | 实际命中的营收科目 | 利润表 |
| `cost_source` | `营业成本` 或 `营业总成本` | 利润表 |
| `equity_source` | `total` 或 `parent` | 资产负债表 |

## 水平项（期末 / 本期）

| 列 | 公式 / 口径 | 科目 |
|----|-------------|------|
| `revenue` | 本期营收 | 优先「营业总收入」，否则「一、营业收入」 |
| `revenue_prior` | 上期营收 | 同上，上期金额列 |
| `cogs` | 营业成本（不含期间费用） | 「其中：营业成本」/「减：营业成本」/「营业成本」 |
| `total_operating_cost` | 营业总成本 | 「营业总成本」；不算进毛利 |
| `net_profit` | 本期净利润 | 「四、/五、净利润（净亏损以…）」 |
| `net_profit_prior` | 上期净利润 | 同上，上期金额列 |
| `total_assets` | 期末资产 | 「资产总计」 |
| `total_assets_begin` | 期初资产 | 「资产总计」期初余额 |
| `current_assets` | 期末流动资产 | 「流动资产合计」 |
| `current_liabilities` | 期末流动负债 | 「流动负债合计」 |
| `total_liabilities` | 期末负债 | 「负债合计」 |
| `equity` | 期末所有者权益 | 「所有者权益（或股东权益）合计」 |
| `equity_begin` | 期初所有者权益 | 同上，期初余额 |
| `accounts_receivable` | 期末应收账款 | 「应收账款」（不含票据、应收款项融资） |
| `inventory` | 期末存货 | 「存货」 |
| `accounts_payable` | 期末应付账款 | 「应付账款」 |
| `ocf` | 本期经营现金流净额 | 「经营活动产生的现金流量净额」 |
| `ocf_prior` | 上期经营现金流净额 | 同上，上期金额列 |
| `icf` | 本期投资现金流净额 | 「投资活动产生的现金流量净额」；缺行保持缺失，不记 0 |
| `fcf` | 本期筹资现金流净额 | 「筹资活动产生的现金流量净额」；缺行保持缺失，不记 0 |
| `cash` | 期末货币资金 | 「货币资金」 |
| `st_borrowings` | 期末短期借款 | 「短期借款」 |
| `current_portion_ltd` | 一年内到期的非流动负债 | 「一年内到期的非流动负债」 |
| `st_interest_bearing` | 有息短债 | 短期借款 + 一年内到期；两项都缺则为缺失，不按 0 |
| `other_receivables` | 其他应收款 | 「其他应收款」（不含其中：应收利息） |
| `prepayments` | 预付款项 | 「预付款项」 |
| `goodwill` | 商誉 | 「商誉」；空单元格保持缺失，不记 0 |
| `contract_liabilities` | 合同负债 | 「合同负债」 |
| `advances_from_customers` | 预收 | 「预收款项」/「预收账款」 |
| `customer_advances` | 合同负债 + 预收 | 两项都缺则为缺失 |
| `operating_profit` | 营业利润 | 「三、营业利润（亏损以…）」 |
| `selling_expense` / `admin_expense` / `rd_expense` / `finance_expense` | 销售 / 管理 / 研发 / 财务费用 | 对应科目；「其中」行不算 |
| `interest_expense` | 利息费用 | 优先「其中：利息费用」 |
| `non_operating_income` | 营业外收入 | 「营业外收入」 |
| `sales_cash` | 销售商品收现 | 「销售商品、提供劳务收到的现金」 |
| `avg_assets` | 平均资产 | 期初、期末都有则取平均，否则用期末 |
| `avg_equity` | 平均权益 | 同上 |

## 质量标记

| 列 | 含义 |
|----|------|
| `gm_valid` | 成本来自营业成本，毛利率可解释 |
| `equity_negative` | 期末权益为负或平均权益 ≤ 0 |
| `roe_valid` | 有净利润、有平均权益、且非负权益 |
| `bs_articulation_ok` | 相对勾稽差距 ≤ 1%；缺科目则为空 |
| `bs_gap` / `bs_rel_gap` | 勾稽差额（元）与相对差距 |
| `np_truncated` | 有营收、缺净利润（第一张表截断） |
| `ocf_missing_kind` | `present` / `truncated` / `no_statement` |
| `inventory_status` | `positive` / `zero` / `missing` |
| `dso_anomalous` / `dio_anomalous` / `dpo_anomalous` | 天数 > 730 或 < 0 |
| `goodwill_status` | `positive` / `zero` / `missing`（空单元格是 missing） |
| `flag_cash_debt_high` | 存贷双高：货币资金/资产 ≥ 20% 且 短债/资产 ≥ 20%；科目缺则为空 |
| `flag_other_receivables` | 其他应收/资产 ≥ 10% |
| `flag_goodwill` | 商誉/资产 ≥ 10%（仅商誉有数时评价） |
| `flag_core_profit_off` | 本业比落在 90%–110% 之外 |
| `flag_cash_conversion_low` | 收现率 < 80% |
| `flag_interest_cover_weak` | 利息保障倍数 < 2 |
| `n_red_flags` | 上面六项为真的个数（缺科目不计入） |

## 比率、天数与同比（原始列）

| 列 | 公式 | 备注 |
|----|------|------|
| `gross_margin` | `(revenue - cogs) / revenue` | 仅 `gm_valid` |
| `net_margin` | `net_profit / revenue` | 杜邦净利率 |
| `roe` | `net_profit / avg_equity` | 仅 `roe_valid` |
| `asset_turnover` | `revenue / avg_assets` | 杜邦周转 |
| `equity_multiplier` | `avg_assets / avg_equity` | 仅 `roe_valid` |
| `dupont_product` | `net_margin × asset_turnover × equity_multiplier` | 应与有效 `roe` 相等 |
| `current_ratio` | `current_assets / current_liabilities` | |
| `debt_ratio` | `total_liabilities / total_assets` | 资产负债率 |
| `ar_to_revenue` | `accounts_receivable / revenue` | |
| `inventory_to_revenue` | `inventory / revenue` | 存货为 0 则比率为 0；缺行则缺失 |
| `ocf_to_revenue` | `ocf / revenue` | |
| `accruals_to_revenue` | `(net_profit - ocf) / revenue` | 正值：利润快于经营现金 |
| `dso` | `accounts_receivable / revenue × 365` | 天 |
| `dio` | `inventory / cogs × 365` | 需营业成本；存货 0 → 0 天 |
| `dpo` | `accounts_payable / cogs × 365` | 需营业成本 |
| `operating_cycle` | `dso + dio` | 经营周期 |
| `ccc` | `operating_cycle - dpo` | 现金周期 |
| `ocf_minus_np` | `ocf - net_profit` | 单位：元 |
| `revenue_yoy` | `(revenue - revenue_prior) / \|revenue_prior\|` | 上期为 0 则缺失 |
| `net_profit_yoy` | `(net_profit - net_profit_prior) / \|net_profit_prior\|` | |
| `ocf_yoy` | `(ocf - ocf_prior) / \|ocf_prior\|` | |
| `roa` | `net_profit / avg_assets` | 平均资产 ≤ 0 则缺失 |
| `operating_margin` | `operating_profit / revenue` | |
| `core_profit_ratio` | `operating_profit / (operating_profit + non_operating_income)` | 营业外收入缺行按 0；营业利润缺则本业比缺 |
| `sga_to_revenue` | `(selling + admin) / revenue` | 销售或管理缺一项则缺失 |
| `rd_to_revenue` / `finance_to_revenue` | 研发或财务费用 / 营收 | |
| `cash_to_assets` | `cash / total_assets` | |
| `st_debt_to_assets` | `st_interest_bearing / total_assets` | |
| `cash_ratio` | `cash / current_liabilities` | 现金比率 |
| `quick_ratio` | `(current_assets - inventory) / current_liabilities` | 存货缺行则速动比率缺失 |
| `other_receivables_to_assets` | `other_receivables / total_assets` | |
| `goodwill_to_assets` | `goodwill / total_assets` | |
| `prepayments_to_assets` | `prepayments / total_assets` | |
| `customer_advances_to_revenue` | `customer_advances / revenue` | |
| `cash_conversion` | `sales_cash / revenue` | 收现率 |
| `interest_coverage` | `operating_profit / interest_expense` | 利息费用 ≤ 0 或缺失则为缺 |

`*_w` 是对应列在有效营收样本上的 1% / 99% 分位截尾。核对单家公司请看无 `_w` 的原始列。行业画像和杜邦的**主口径**用带质量标记的原始列（天数再加 730 天帽）；IQR 剔除和 `*_w` 只出现在稳健表里，不替换主中位数。每个问题各自完整个案，不把 195 家硬删成一张表。

期末余额 / 本年流量的天数不是严格的平均余额周转，只用于截面比较。

## 已知缺口

- 不少利润表第一张表在「净利润」行之前被截断，`net_profit` 覆盖会低于营收覆盖。不把母公司净利润拼到合并营收上。
- 现金流量表同样可能缺「经营活动产生的现金流量净额」。
- 单期年报：同比来自表内上期列，不是多年面板。没有三年毛利率/净利率稳定性，也没有审计意见。
- PDF 集合里可能混入非新三板代码；本表不按板块过滤。
- 公司报告卡：`python report_card.py`，底稿 `company_report_cards.md`，单家文件在 `output/analysis/report_cards/`。
