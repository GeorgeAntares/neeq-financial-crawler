"""
Firm-level four-section report cards.

Sections follow the listed-company reading questions:
  1. profitability — how the firm makes money
  2. earnings quality — whether profit is cash-backed
  3. solvency / trap — cash vs short-term debt, other receivables, goodwill
  4. operating + cash this year — the one-year evidence of sustainability

Industry medians are complete-case by field. Missing stays NaN (never filled with 0).
Per-firm markdown lives under output/analysis/report_cards/ (gitignored).
The committed snapshot is coverage + a few example cards.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from company_metrics import (
    CASH_CONVERSION_LOW,
    CASH_HIGH,
    CORE_RATIO_HIGH,
    CORE_RATIO_LOW,
    DAYS_ANOMALY,
    GOODWILL_HIGH,
    INTEREST_COVER_WEAK,
    OTHER_REC_HIGH,
    OUTPUT_DIR_DEFAULT,
    ST_DEBT_HIGH,
    build_company_metrics,
    default_csv_dir,
)
from industry_groups import GROUP_ORDER, PDF_DIR_DEFAULT
from industry_portrait import attach_industry

CARDS_CSV = "company_report_cards.csv"
FLAGS_CSV = "company_report_card_flags.csv"
COVERAGE_CSV = "company_report_card_coverage.csv"
REPORT_NAME = "company_report_cards.md"
CARDS_DIRNAME = "report_cards"
FLAGS_CHART = "company_report_card_flags.png"

FLAG_LABELS = [
    ("flag_cash_debt_high", "存贷双高", f"货币资金/资产≥{CASH_HIGH:.0%} 且 短债/资产≥{ST_DEBT_HIGH:.0%}"),
    ("flag_other_receivables", "其他应收偏高", f"其他应收/资产≥{OTHER_REC_HIGH:.0%}"),
    ("flag_goodwill", "商誉偏高", f"商誉/资产≥{GOODWILL_HIGH:.0%}"),
    ("flag_core_profit_off", "本业比偏离", f"本业比落在 {CORE_RATIO_LOW:.0%}–{CORE_RATIO_HIGH:.0%} 之外"),
    ("flag_cash_conversion_low", "收现率偏低", f"销售商品收现/营收<{CASH_CONVERSION_LOW:.0%}"),
    ("flag_interest_cover_weak", "利息保障偏弱", f"营业利润/利息费用<{INTEREST_COVER_WEAK:.0f}"),
]

COMPARE_FIELDS = [
    "gross_margin",
    "net_margin",
    "operating_margin",
    "roe",
    "asset_turnover",
    "equity_multiplier",
    "core_profit_ratio",
    "sga_to_revenue",
    "accruals_to_revenue",
    "ocf_to_revenue",
    "cash_conversion",
    "current_ratio",
    "cash_ratio",
    "debt_ratio",
    "cash_to_assets",
    "st_debt_to_assets",
    "other_receivables_to_assets",
    "dso",
    "dio",
    "ccc",
]


def _is_true(series):
    if series is None:
        return pd.Series(dtype=bool)
    if getattr(series, "dtype", None) == bool:
        return series.fillna(False)
    return series == True


def industry_medians(labeled, fields=None):
    """Complete-case median and n per industry × field. Days still drop >730 anomalous."""
    fields = fields or COMPARE_FIELDS
    rows = []
    work = labeled.copy()
    if "industry" not in work.columns:
        work["industry"] = "其他"
    groups = list(GROUP_ORDER) + ["合计"]
    for industry in groups:
        part = work if industry == "合计" else work[work["industry"] == industry]
        n_group = len(part)
        for field in fields:
            if field not in part.columns:
                rows.append({"industry": industry, "field": field, "n_group": n_group, "n": 0, "median": np.nan})
                continue
            values = pd.to_numeric(part[field], errors="coerce")
            if field in ("dso", "dio", "dpo", "ccc", "operating_cycle"):
                anomalous = part.get(f"{field}_anomalous")
                if anomalous is not None:
                    values = values.where(_is_true(anomalous) == False)
                else:
                    values = values.where((values <= DAYS_ANOMALY) & (values >= 0))
            finite = values.dropna()
            rows.append(
                {
                    "industry": industry,
                    "field": field,
                    "n_group": n_group,
                    "n": int(len(finite)),
                    "median": float(finite.median()) if len(finite) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def median_lookup(medians, industry, field):
    if medians is None or medians.empty:
        return np.nan, 0
    hit = medians[(medians["industry"] == industry) & (medians["field"] == field)]
    if hit.empty:
        return np.nan, 0
    return hit.iloc[0]["median"], int(hit.iloc[0]["n"])


def coverage_table(labeled):
    """How many firms have the extra line items used by the four questions."""
    n = len(labeled)
    specs = [
        ("cash", "货币资金", "偿债 / 困住"),
        ("st_borrowings", "短期借款", "偿债 / 困住"),
        ("current_portion_ltd", "一年内到期非流动负债", "偿债 / 困住"),
        ("other_receivables", "其他应收款", "偿债 / 困住"),
        ("goodwill", "商誉", "偿债 / 困住"),
        ("operating_profit", "营业利润", "怎么赚钱"),
        ("non_operating_income", "营业外收入", "怎么赚钱"),
        ("selling_expense", "销售费用", "怎么赚钱"),
        ("admin_expense", "管理费用", "怎么赚钱"),
        ("rd_expense", "研发费用", "怎么赚钱"),
        ("interest_expense", "利息费用", "偿债"),
        ("sales_cash", "销售商品收现", "利润真不真"),
        ("icf", "投资活动净额", "营运与现金"),
        ("fcf", "筹资活动净额", "营运与现金"),
        ("contract_liabilities", "合同负债", "利润真不真"),
        ("cash_conversion", "收现率", "利润真不真"),
        ("core_profit_ratio", "本业比", "怎么赚钱"),
        ("interest_coverage", "利息保障倍数", "偿债"),
        ("quick_ratio", "速动比率", "偿债"),
    ]
    rows = []
    for field, label, question in specs:
        if field not in labeled.columns:
            k = 0
        else:
            k = int(pd.to_numeric(labeled[field], errors="coerce").notna().sum())
        rows.append(
            {
                "field": field,
                "label": label,
                "question": question,
                "n": k,
                "n_group": n,
                "share": k / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def flag_summary(labeled):
    rows = []
    n = len(labeled)
    industries = ["合计"] + [g for g in GROUP_ORDER if g in set(labeled.get("industry", pd.Series(dtype=object)))]
    if "industry" not in labeled.columns:
        industries = ["合计"]
    for industry in industries:
        part = labeled if industry == "合计" else labeled[labeled["industry"] == industry]
        n_part = len(part)
        for col, label, rule in FLAG_LABELS:
            if col not in part.columns:
                k = 0
                evaluated = 0
            else:
                values = part[col]
                evaluated = int(values.notna().sum()) if values.dtype != bool else n_part
                k = int(_is_true(values).sum())
            rows.append(
                {
                    "industry": industry,
                    "flag": col,
                    "label": label,
                    "rule": rule,
                    "n_flag": k,
                    "n_evaluated": evaluated,
                    "n_group": n_part,
                    "share_of_group": k / n_part if n_part else np.nan,
                    "share_of_evaluated": k / evaluated if evaluated else np.nan,
                }
            )
    rows.append(
        {
            "industry": "合计",
            "flag": "firms",
            "label": "样本",
            "rule": "",
            "n_flag": n,
            "n_evaluated": n,
            "n_group": n,
            "share_of_group": 1.0,
            "share_of_evaluated": 1.0,
        }
    )
    return pd.DataFrame(rows)


def _pct(value):
    if value is None or pd.isna(value):
        return "缺"
    return f"{float(value) * 100:.1f}%"


def _num(value, digits=2):
    if value is None or pd.isna(value):
        return "缺"
    return f"{float(value):.{digits}f}"


def _days(value):
    if value is None or pd.isna(value):
        return "缺"
    return f"{float(value):.0f} 天"


def _money(value):
    if value is None or pd.isna(value):
        return "缺"
    amount = float(value)
    if abs(amount) >= 1e8:
        return f"{amount / 1e8:.2f} 亿元"
    if abs(amount) >= 1e4:
        return f"{amount / 1e4:.0f} 万元"
    return f"{amount:.0f} 元"


def _cell(row, field, kind="pct", medians=None):
    value = row[field] if field in row.index else np.nan
    if kind == "pct":
        text = _pct(value)
    elif kind == "days":
        text = _days(value)
    elif kind == "num":
        text = _num(value)
    else:
        text = _money(value)
    industry = row["industry"] if "industry" in row.index else "其他"
    if medians is not None:
        med, n_med = median_lookup(medians, industry, field)
        if kind == "pct":
            med_text = _pct(med)
        elif kind == "days":
            med_text = _days(med)
        else:
            med_text = _num(med)
        if med_text != "缺":
            text = f"{text}（行业中位 {med_text}，n={n_med}）"
    return text


def _flag_on(row, name):
    if name not in row.index or pd.isna(row[name]):
        return False
    value = row[name]
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def render_card(row, medians=None):
    """One firm, four sections. Missing prints as 缺."""
    code = row.get("stock_code", "")
    name = row.get("company_name", "")
    year = row.get("year", "")
    industry = row.get("industry", "其他")
    n_flags = row["n_red_flags"] if "n_red_flags" in row.index and pd.notna(row["n_red_flags"]) else 0
    lines = [
        f"# {code} {name} {year}",
        "",
        f"行业：**{industry}**　红旗 {int(n_flags)} 项",
        "",
        f"营收 {_money(row.get('revenue'))}　净利润 {_money(row.get('net_profit'))}　"
        f"OCF {_money(row.get('ocf'))}",
        "",
        "## 1. 怎么赚钱（盈利）",
        "",
        f"- 毛利率：{_cell(row, 'gross_margin', 'pct', medians)}",
        f"- 营业利润率：{_cell(row, 'operating_margin', 'pct', medians)}",
        f"- 净利率：{_cell(row, 'net_margin', 'pct', medians)}",
        f"- ROE：{_cell(row, 'roe', 'pct', medians)}",
        f"- 本业比（营业利润 / (营业利润+营业外收入)）：{_cell(row, 'core_profit_ratio', 'pct', medians)}",
        f"- 销售+管理费用率：{_cell(row, 'sga_to_revenue', 'pct', medians)}",
        f"- 研发费用率：{_cell(row, 'rd_to_revenue', 'pct', medians)}",
        "",
        "## 2. 利润真不真（盈余质量）",
        "",
        f"- 应计/收入：{_cell(row, 'accruals_to_revenue', 'pct', medians)}",
        f"- OCF/收入：{_cell(row, 'ocf_to_revenue', 'pct', medians)}",
        f"- 收现率（销售商品收现/营收）：{_cell(row, 'cash_conversion', 'pct', medians)}",
        f"- 合同负债+预收 / 收入：{_cell(row, 'customer_advances_to_revenue', 'pct', medians)}",
    ]
    np_amt = pd.to_numeric(pd.Series([row.get("net_profit")]), errors="coerce").iloc[0]
    ocf_amt = pd.to_numeric(pd.Series([row.get("ocf")]), errors="coerce").iloc[0]
    if pd.isna(np_amt) or pd.isna(ocf_amt):
        sign_note = "缺净利润或 OCF，不判断符号"
    elif np_amt > 0 and ocf_amt < 0:
        sign_note = "利润为正、OCF 为负"
    elif np_amt < 0 and ocf_amt > 0:
        sign_note = "利润为负、OCF 为正"
    else:
        sign_note = "利润与 OCF 同号"
    bs_ok = row.get("bs_articulation_ok")
    if pd.isna(bs_ok) if not isinstance(bs_ok, (bool, np.bool_)) else False:
        bs_note = "缺科目，未勾稽"
    elif bs_ok in (True, "True", 1):
        bs_note = "相对差距 ≤ 1%"
    else:
        bs_note = "勾稽失败"
    lines += [
        f"- 利润与现金符号：{sign_note}",
        f"- 资产负债表勾稽：{bs_note}",
        "",
        "## 3. 股东会不会被困住（偿债）",
        "",
        f"- 流动比率：{_cell(row, 'current_ratio', 'num', medians)}",
        f"- 速动比率：{_cell(row, 'quick_ratio', 'num', medians)}",
        f"- 现金比率：{_cell(row, 'cash_ratio', 'num', medians)}",
        f"- 资产负债率：{_cell(row, 'debt_ratio', 'pct', medians)}",
        f"- 货币资金/资产：{_cell(row, 'cash_to_assets', 'pct', medians)}",
        f"- 短债/资产：{_cell(row, 'st_debt_to_assets', 'pct', medians)}",
        f"- 其他应收/资产：{_cell(row, 'other_receivables_to_assets', 'pct', medians)}",
        f"- 商誉/资产：{_cell(row, 'goodwill_to_assets', 'pct', medians)}",
        f"- 利息保障倍数：{_cell(row, 'interest_coverage', 'num', medians)}",
        "",
        "## 4. 营运与现金（当年证据）",
        "",
        f"- DSO：{_cell(row, 'dso', 'days', medians)}",
        f"- DIO：{_cell(row, 'dio', 'days', medians)}",
        f"- CCC：{_cell(row, 'ccc', 'days', medians)}",
        f"- 营收同比：{_cell(row, 'revenue_yoy', 'pct', medians)}",
        f"- 净利润同比：{_cell(row, 'net_profit_yoy', 'pct')}",
        f"- OCF 同比：{_cell(row, 'ocf_yoy', 'pct')}",
        "",
        "单期年报没有三年毛利率/净利率波动，可持续性只能看到当年周转和同比。",
        "",
        "同目录一页纸 `*_card.png`（缺科目跳过该柱，不用 0 填）：KPI、利润瀑布、现金流三分类、杜邦对照、营运天数、红旗。",
        "",
        "## 红旗",
        "",
    ]
    any_flag = False
    for col, label, rule in FLAG_LABELS:
        if _flag_on(row, col):
            lines.append(f"- **{label}**：{rule}")
            any_flag = True
    if row.get("equity_negative") in (True, "True", 1):
        lines.append("- **负权益**：ROE / 权益乘数已冻结")
        any_flag = True
    if row.get("np_truncated") in (True, "True", 1):
        lines.append("- **净利润截断**：第一张利润表没有净利润行，保持 NaN")
        any_flag = True
    if not any_flag:
        lines.append("- 无（可评价的红旗均未触发）")
    lines.append("")
    return "\n".join(lines)


def pick_examples(labeled):
    """Most-flagged firm plus one relatively complete card per industry."""
    chosen = []
    seen = set()
    work = labeled.copy()
    if "n_red_flags" in work.columns and not work.empty:
        top = work.sort_values(["n_red_flags", "revenue"], ascending=[False, False]).iloc[0]
        chosen.append(("红旗最多", top))
        seen.add(str(top.get("stock_code")))
    for industry in GROUP_ORDER:
        if "industry" not in work.columns:
            break
        part = work[work["industry"] == industry]
        part = part[~part["stock_code"].astype(str).isin(seen)]
        if part.empty:
            continue
        flags = pd.to_numeric(part.get("n_red_flags"), errors="coerce").fillna(99)
        complete = pd.Series(True, index=part.index)
        if "cash_conversion" in part.columns:
            complete = complete & pd.to_numeric(part["cash_conversion"], errors="coerce").notna()
        if "operating_margin" in part.columns:
            complete = complete & pd.to_numeric(part["operating_margin"], errors="coerce").notna()
        score = flags.where(complete, flags + 50)
        order = pd.DataFrame({"score": score, "revenue": pd.to_numeric(part.get("revenue"), errors="coerce")})
        idx = order.sort_values(["score", "revenue"], ascending=[True, False]).index[0]
        row = part.loc[idx]
        chosen.append((industry, row))
        seen.add(str(row.get("stock_code")))
    return chosen


def render_index(labeled, coverage, flags, medians, examples):
    n = len(labeled)
    lines = [
        "# 公司报告卡",
        "",
        f"样本 {n} 家。每家四段：怎么赚钱、利润真不真、会不会被困住、营运与现金（当年）。",
        "行业中位数按字段完整个案；缺科目写「缺」，不用 0 填。",
        "存贷双高、其他应收、商誉、本业比、收现率、利息保障只在科目齐全时评价。",
        "",
        "复现：`python company_metrics.py` 然后 `python report_card.py`。",
        "单家 markdown 与一页纸图在 `output/analysis/report_cards/`（gitignored）。",
        "每家一张 `*_card.png`：KPI、利润瀑布、现金流三分类、杜邦对照、营运天数、红旗。",
        "",
        "## 科目覆盖",
        "",
        "| 科目 | 问题 | n | 覆盖 |",
        "|------|------|---|------|",
    ]
    for _, row in coverage.iterrows():
        lines.append(
            f"| {row['label']} | {row['question']} | {int(row['n'])}/{int(row['n_group'])} | "
            f"{float(row['share']) * 100:.1f}% |"
        )
    lines += [
        "",
        "## 红旗计数（合计）",
        "",
        "| 红旗 | 触发 | 可评价 | 占样本 | 规则 |",
        "|------|------|--------|--------|------|",
    ]
    total_flags = flags[(flags["industry"] == "合计") & (flags["flag"] != "firms")]
    for _, row in total_flags.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['n_flag'])} | {int(row['n_evaluated'])} | "
            f"{float(row['share_of_group']) * 100:.1f}% | {row['rule']} |"
        )
    if "industry" in labeled.columns and "n_red_flags" in labeled.columns:
        lines += ["", "分行业平均红旗数："]
        for industry in GROUP_ORDER:
            part = labeled[labeled["industry"] == industry]
            if part.empty:
                continue
            mean_flags = pd.to_numeric(part["n_red_flags"], errors="coerce").mean()
            lines.append(f"- {industry}（n={len(part)}）：平均 {mean_flags:.2f} 项")
    lines += ["", "## 例卡", ""]
    for title, row in examples:
        code = row.get("stock_code", "")
        name = row.get("company_name", "")
        lines.append(f"### 例：{title}（{code} {name}）")
        lines.append("")
        card = render_card(row, medians=medians)
        body = "\n".join(card.splitlines()[1:]).strip()
        lines.append(body)
        lines.append("")
    lines += [
        "## 口径",
        "",
        f"- 本业比：营业利润 / (营业利润 + 营业外收入)；营业外收入缺行按 0，营业利润缺则本业比缺。",
        f"- 短债 = 短期借款 + 一年内到期非流动负债；两项都缺则短债缺，不按 0。",
        f"- 速动比率在存货缺行时为缺（存货 0 才减 0）。",
        f"- 利息保障：营业利润 / 利息费用，利息费用 ≤ 0 或缺失则为缺。",
        f"- 收现率：销售商品、提供劳务收到的现金 / 营收。",
        f"- 商誉空单元格保持缺失，不记 0。",
        f"- 投资 / 筹资净额来自首块现金流量表；缺行不记 0，三分类柱只画有数的类。",
        f"- 利润瀑布的「税及其他」是营业利润到净利润的残差（未抽所得税，也没有扣非）。",
        f"- 同行条形取同行业、该指标齐全、对数营收最近的 5–8 家；本公司蓝色，同行灰色。",
        "",
        "## 局限",
        "",
        "- 单年截面，没有三年毛利率/净利率稳定性，也没有审计意见。",
        "- 70 家净利润截断会让盈利段和本业比一起缺。",
        "- 红旗阈值来自常见阅读清单，不是违约模型。",
        "",
    ]
    return "\n".join(lines)


def _flag_rate_title(flags):
    total = flags[(flags["flag"] != "firms") & (flags["industry"] == "合计")]
    if total.empty:
        total = flags[flags["flag"] != "firms"]
    if total.empty:
        return "报告卡红旗占比"
    top = total.sort_values("n_flag", ascending=False).iloc[0]
    return f"最常见红旗是{top['label']}（{int(top['n_flag'])}/{int(top['n_evaluated'])}）"


def plot_flag_rates(flags, output_dir):
    from chart_theme import FIRM, RULE, group_color, restyle, save_fig, set_panel, use_mpl

    _, plt = use_mpl()
    work = flags[(flags["flag"] != "firms") & (flags["industry"].isin(GROUP_ORDER))]
    if work.empty:
        work = flags[(flags["flag"] != "firms") & (flags["industry"] == "合计")]
        if work.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(work["label"], work["share_of_group"] * 100, color=FIRM, linewidth=0)
        restyle(ax)
        set_panel(ax, title=_flag_rate_title(flags), kicker="3  会不会被困住", ylabel=None)
        ax.set_xlabel("占样本 %")
        path = os.path.join(output_dir, FLAGS_CHART)
        return save_fig(fig, path)

    labels = [lab for _, lab, _ in FLAG_LABELS]
    industries = [g for g in GROUP_ORDER if g in set(work["industry"])]
    fig, ax = plt.subplots(figsize=(10.4, 5))
    x = np.arange(len(labels))
    width = 0.24
    for i, industry in enumerate(industries):
        part = work[work["industry"] == industry].set_index("flag")
        vals = []
        for col, _, _ in FLAG_LABELS:
            if col in part.index:
                vals.append(float(part.loc[col, "share_of_group"]) * 100)
            else:
                vals.append(np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=industry, color=group_color(industry), linewidth=0)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("占该行业 %")
    ax.axhline(0, color=RULE, linewidth=0.8)
    restyle(ax)
    ax.legend(loc="upper right")
    set_panel(ax, title=_flag_rate_title(flags), kicker="3  会不会被困住")
    path = os.path.join(output_dir, FLAGS_CHART)
    return save_fig(fig, path)


def write_firm_cards(labeled, medians, cards_dir):
    os.makedirs(cards_dir, exist_ok=True)
    n_written = 0
    for _, row in labeled.iterrows():
        code = str(row.get("stock_code", "unknown"))
        name = str(row.get("company_name", "")).replace("/", "_").replace("\\", "_")
        year = str(row.get("year", ""))
        path = os.path.join(cards_dir, f"{code}_{name}_{year}.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_card(row, medians=medians))
        n_written += 1
    return n_written


def load_or_build_metrics(metrics_path, csv_dir):
    if metrics_path and os.path.isfile(metrics_path):
        return pd.read_csv(metrics_path, encoding="utf-8-sig")
    default_metrics = os.path.join(OUTPUT_DIR_DEFAULT, "company_metrics.csv")
    if os.path.isfile(default_metrics):
        return pd.read_csv(default_metrics, encoding="utf-8-sig")
    return build_company_metrics(csv_dir or default_csv_dir())


def main(metrics_path=None, csv_dir=None, pdf_dir=None, output_dir=None, report_path=None, write_firms=True, write_charts=True):
    output_dir = output_dir or OUTPUT_DIR_DEFAULT
    pdf_dir = pdf_dir or PDF_DIR_DEFAULT
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Firm report cards / 公司报告卡")
    print("=" * 60)

    metrics = load_or_build_metrics(metrics_path, csv_dir)
    if metrics is None or metrics.empty:
        print("No company metrics.")
        return None
    labeled = attach_industry(metrics, pdf_root=pdf_dir)
    medians = industry_medians(labeled)
    coverage = coverage_table(labeled)
    flags = flag_summary(labeled)
    examples = pick_examples(labeled)

    cards_csv = os.path.join(output_dir, CARDS_CSV)
    keep_cols = [
        col
        for col in [
            "stock_code",
            "company_name",
            "year",
            "industry",
            "industry_raw",
            "sector",
            "sector_label",
            "revenue",
            "net_profit",
            "ocf",
            "icf",
            "fcf",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "roe",
            "core_profit_ratio",
            "sga_to_revenue",
            "rd_to_revenue",
            "accruals_to_revenue",
            "ocf_to_revenue",
            "cash_conversion",
            "customer_advances_to_revenue",
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
            "debt_ratio",
            "cash_to_assets",
            "st_debt_to_assets",
            "other_receivables_to_assets",
            "goodwill_to_assets",
            "interest_coverage",
            "dso",
            "dio",
            "ccc",
            "revenue_yoy",
            "n_red_flags",
        ]
        + [col for col, _, _ in FLAG_LABELS]
        if col in labeled.columns
    ]
    labeled[keep_cols].to_csv(cards_csv, index=False, encoding="utf-8-sig")
    cov_path = os.path.join(output_dir, COVERAGE_CSV)
    coverage.to_csv(cov_path, index=False, encoding="utf-8-sig")
    flag_path = os.path.join(output_dir, FLAGS_CSV)
    flags.to_csv(flag_path, index=False, encoding="utf-8-sig")

    report = render_index(labeled, coverage, flags, medians, examples)
    md_path = os.path.join(output_dir, REPORT_NAME)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report)

    n_cards = 0
    cards_dir = os.path.join(output_dir, CARDS_DIRNAME)
    if write_firms:
        n_cards = write_firm_cards(labeled, medians, cards_dir)

    chart = None
    n_chart_firms = 0
    try:
        chart = plot_flag_rates(flags, output_dir)
    except Exception as exc:
        print(f"flag chart skipped: {exc}")
    if write_charts:
        try:
            from report_charts import plot_example_flag_grid, write_firm_charts

            example_codes = [str(row.get("stock_code")) for _, row in examples]
            codes = None if write_firms else example_codes
            n_chart_firms = write_firm_charts(labeled, medians, cards_dir, FLAG_LABELS, codes=codes)
            grid = plot_example_flag_grid(
                examples, os.path.join(output_dir, "company_report_card_flag_grid.png"), FLAG_LABELS
            )
            if grid:
                print(f"wrote {grid}")
            from chart_catalog import organize_charts

            dest = organize_charts(
                output_dir,
                pdf_root=pdf_dir,
                firm_codes=example_codes,
            )
            if dest:
                print(f"wrote chart catalog under {dest}")
        except Exception as exc:
            print(f"firm charts skipped: {exc}")

    print("coverage:")
    for _, row in coverage.iterrows():
        print(f"  {row['label']}: {int(row['n'])}/{int(row['n_group'])} ({row['share'] * 100:.1f}%)")
    print("flags:")
    for _, row in flags[(flags["industry"] == "合计") & (flags["flag"] != "firms")].iterrows():
        print(f"  {row['label']}: {int(row['n_flag'])} / {int(row['n_evaluated'])}")
    print(f"wrote {cards_csv}")
    print(f"wrote {cov_path}")
    print(f"wrote {flag_path}")
    print(f"wrote {md_path}")
    if n_cards:
        print(f"wrote {n_cards} firm cards under {cards_dir}")
    if n_chart_firms:
        print(f"wrote charts for {n_chart_firms} firms under {cards_dir}")
    if chart:
        print(f"wrote {chart}")
    if report_path:
        print(f"wrote {report_path}")
    return labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEEQ firm-level four-section report cards")
    parser.add_argument("--metrics", default=None, help="company_metrics.csv")
    parser.add_argument("--csv-dir", default=None, help="CSV folder if metrics file is missing")
    parser.add_argument("--pdf-dir", default=None, help="PDF folder with industry subdirs")
    parser.add_argument("--output-dir", default=None, help="Charts and tables")
    parser.add_argument("--report", default=None, help="Optional extra markdown path")
    parser.add_argument("--no-firm-files", action="store_true", help="Skip per-firm markdown files")
    parser.add_argument("--no-firm-charts", action="store_true", help="Skip per-firm png charts")
    args = parser.parse_args()
    main(
        metrics_path=args.metrics,
        csv_dir=args.csv_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        write_firms=not args.no_firm_files,
        write_charts=not args.no_firm_charts,
    )
