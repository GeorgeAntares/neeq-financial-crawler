"""
Industry portraits: working-capital cycle and earnings quality.

Groups: 制造 / 软件信息 / 其他, from output/pdf subfolders.

Contest-style *caliber*, not the eight-step order:
  each question uses its own complete cases; medians with n;
  730-day cap is the main estimator; IQR drop and 1%/99% winsor
  are robustness, not a second sample filter.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from company_metrics import OUTPUT_DIR_DEFAULT, build_company_metrics, default_csv_dir
from industry_groups import (
    GROUP_MANUFACTURING,
    GROUP_ORDER,
    GROUP_OTHER,
    GROUP_SOFTWARE,
    PDF_DIR_DEFAULT,
    assign_industries,
    normalize_code,
)
from preprocess import add_iqr_flags

MEDIAN_COLS = [
    "revenue",
    "gross_margin",
    "net_margin",
    "roe",
    "ar_to_revenue",
    "inventory_to_revenue",
    "wc_to_revenue",
    "dso",
    "dio",
    "dpo",
    "operating_cycle",
    "ccc",
    "accruals_to_revenue",
    "ocf_to_revenue",
    "current_ratio",
    "debt_ratio",
    "revenue_yoy",
]

BOXPLOT_COLS = [
    ("dso", "应收账款周转天数 DSO"),
    ("dio", "存货周转天数 DIO"),
    ("accruals_to_revenue", "应计利润 / 收入"),
    ("gross_margin", "毛利率（仅营业成本）"),
]

PORTRAIT_CHART = "industry_portrait.png"
WC_CHART = "industry_wc_cycle.png"
CASH_GAP_CHART = "industry_cash_gap.png"
SENSITIVITY_CHART = "industry_sensitivity.png"
MEDIANS_NAME = "industry_medians.csv"
ASSIGN_NAME = "industry_assignments.csv"
COVERAGE_NAME = "industry_coverage.csv"
SENSITIVITY_NAME = "industry_sensitivity.csv"
ASSOC_NAME = "industry_assoc.csv"
REPORT_NAME = "industry_portrait.md"

SENSITIVITY_COLS = [
    "dso",
    "dio",
    "dpo",
    "operating_cycle",
    "ccc",
    "accruals_to_revenue",
    "gross_margin",
    "net_margin",
    "ar_to_revenue",
    "inventory_to_revenue",
]

TREATMENTS = ("quality", "all", "iqr", "winsor")
TREAT_LABEL = {
    "quality": "主口径",
    "all": "含异常天数",
    "iqr": "去掉 IQR 离群",
    "winsor": "1%/99% 缩尾",
}

ASSOC_PAIRS = [
    ("dso", "accruals_to_revenue", "DSO 与应计/收入"),
    ("dso", "net_margin", "DSO 与净利率"),
    ("ccc", "accruals_to_revenue", "CCC 与应计/收入"),
    ("gross_margin", "accruals_to_revenue", "毛利率与应计/收入"),
]


def attach_industry(metrics, pdf_root=None):
    frame = metrics.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_code)
    assigned = pd.DataFrame(assign_industries(frame["stock_code"], pdf_root=pdf_root))
    assigned = assigned.drop_duplicates(subset=["stock_code"])
    if "industry" in frame.columns:
        frame = frame.drop(columns=["industry", "industry_raw"], errors="ignore")
    return frame.merge(assigned, on="stock_code", how="left")


def add_derived(metrics):
    frame = metrics.copy()
    if "ar_to_revenue" in frame.columns and "inventory_to_revenue" in frame.columns:
        frame["wc_to_revenue"] = (
            pd.to_numeric(frame["ar_to_revenue"], errors="coerce")
            + pd.to_numeric(frame["inventory_to_revenue"], errors="coerce")
        )
    np_ok = pd.to_numeric(frame.get("net_profit"), errors="coerce")
    ocf_ok = pd.to_numeric(frame.get("ocf"), errors="coerce")
    frame["has_np_ocf"] = np_ok.notna() & ocf_ok.notna()
    frame["profit_positive"] = np_ok > 0
    frame["ocf_negative"] = ocf_ok < 0
    frame["profit_pos_ocf_neg"] = frame["has_np_ocf"] & (np_ok > 0) & (ocf_ok < 0)
    frame["profit_neg_ocf_pos"] = frame["has_np_ocf"] & (np_ok < 0) & (ocf_ok > 0)
    ocf_yoy = pd.to_numeric(frame.get("ocf_yoy"), errors="coerce")
    frame["ocf_yoy_abs"] = ocf_yoy.abs()
    return add_iqr_outlier_flags(frame)


def _flag_true(part, name):
    if name not in part.columns:
        return pd.Series(True, index=part.index)
    values = part[name]
    if values.dtype == bool:
        return values
    return values == True


def _flag_false(part, name):
    if name not in part.columns:
        return pd.Series(True, index=part.index)
    values = part[name]
    if values.dtype == bool:
        return ~values
    return values != True


def add_iqr_outlier_flags(frame, columns=None):
    """Global Tukey fences, same as preprocess.py. Skip columns already flagged."""
    cols = columns or [c for c in SENSITIVITY_COLS if c in frame.columns]
    missing = [c for c in cols if f"{c}_iqr_out" not in frame.columns]
    if not missing:
        return frame
    return add_iqr_flags(frame, columns=missing)


def _median_n(part, col, where=None):
    if col not in part.columns:
        return np.nan, 0
    series = pd.to_numeric(part[col], errors="coerce")
    if where is not None:
        series = series.where(where)
    series = series.dropna()
    if series.empty:
        return np.nan, 0
    return float(series.median()), int(len(series))


def _median(part, col, where=None):
    med, _ = _median_n(part, col, where=where)
    return med


def _definition_mask(part, col):
    """Invalid GM / ROE never enter, even in the 'all' robustness column."""
    ok = pd.Series(True, index=part.index)
    if col == "gross_margin":
        ok = ok & _flag_true(part, "gm_valid")
    if col == "roe":
        ok = ok & _flag_true(part, "roe_valid")
    return ok


def _median_mask(part, col):
    """Main estimator: definition flags plus the 730-day economic cap."""
    ok = _definition_mask(part, col)
    if col in ("dso", "operating_cycle"):
        ok = ok & _flag_false(part, "dso_anomalous")
    if col in ("dio", "operating_cycle"):
        ok = ok & _flag_false(part, "dio_anomalous")
    if col in ("dpo", "ccc"):
        ok = ok & _flag_false(part, "dpo_anomalous")
    if col == "ccc":
        ok = ok & _flag_false(part, "dso_anomalous") & _flag_false(part, "dio_anomalous")
    return ok


def _treatment_mask(part, col, treatment):
    quality = _median_mask(part, col)
    if treatment == "all":
        return _definition_mask(part, col)
    if treatment == "iqr":
        flag = f"{col}_iqr_out"
        if flag in part.columns:
            return quality & _flag_false(part, flag)
        return quality
    return quality


def _value_col(part, col, treatment):
    if treatment == "winsor":
        wins = f"{col}_w"
        if wins in part.columns:
            return wins
    return col


def _iqr(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def industry_tables(metrics):
    """Median table, WC cycle, and earnings-quality rates, one row per industry."""
    frame = add_derived(metrics)
    if "industry" not in frame.columns:
        raise ValueError("metrics need an industry column; call attach_industry first")
    frame["industry"] = pd.Categorical(frame["industry"], categories=GROUP_ORDER, ordered=True)

    rows = []
    for group, part in frame.groupby("industry", observed=False):
        labeled = part[part["has_np_ocf"]]
        row = {
            "industry": group,
            "n": int(len(part)),
            "n_np_ocf": int(len(labeled)),
            "n_gm_valid": int(_flag_true(part, "gm_valid").sum()) if "gm_valid" in part.columns else int(part.get("gross_margin", pd.Series(dtype=float)).notna().sum()),
            "n_roe_valid": int(_flag_true(part, "roe_valid").sum()) if "roe_valid" in part.columns else int(part.get("roe", pd.Series(dtype=float)).notna().sum()),
        }
        for col in MEDIAN_COLS:
            med, n_med = _median_n(part, col, where=_median_mask(part, col))
            row[f"median_{col}"] = med
            row[f"n_{col}"] = n_med
        row["iqr_ocf_to_revenue"] = _iqr(part.get("ocf_to_revenue"))
        row["median_ocf_yoy_abs"] = part["ocf_yoy_abs"].median()
        if "inventory_status" in part.columns:
            row["share_inventory_zero"] = float((part["inventory_status"] == "zero").mean())
            row["share_inventory_missing"] = float((part["inventory_status"] == "missing").mean())
        else:
            inv = pd.to_numeric(part.get("inventory_to_revenue"), errors="coerce")
            row["share_inventory_zero"] = float((inv == 0).mean()) if len(part) else np.nan
            row["share_inventory_missing"] = float(inv.isna().mean()) if len(part) else np.nan
        if "dso_anomalous" in part.columns:
            row["share_dso_anomalous"] = float(_flag_true(part, "dso_anomalous").mean())
        else:
            row["share_dso_anomalous"] = np.nan
        if labeled.empty:
            row["share_profit_pos"] = np.nan
            row["share_ocf_neg"] = np.nan
            row["share_profit_pos_ocf_neg"] = np.nan
            row["share_profit_neg_ocf_pos"] = np.nan
        else:
            row["share_profit_pos"] = float(labeled["profit_positive"].mean())
            row["share_ocf_neg"] = float(labeled["ocf_negative"].mean())
            row["share_profit_pos_ocf_neg"] = float(labeled["profit_pos_ocf_neg"].mean())
            row["share_profit_neg_ocf_pos"] = float(labeled["profit_neg_ocf_pos"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return frame, summary


def _numeric_col(part, col):
    if col not in part.columns:
        return pd.Series(np.nan, index=part.index)
    return pd.to_numeric(part[col], errors="coerce")


def _coverage_row(part, industry, question, complete, main, mechanism):
    n = int(len(part))
    n_complete = int(complete.sum()) if n else 0
    n_main = int((complete & main).sum()) if n else 0
    return {
        "industry": industry,
        "question": question,
        "n_group": n,
        "n_complete": n_complete,
        "n_main": n_main,
        "n_missing": n - n_complete,
        "missing_share": (n - n_complete) / n if n else np.nan,
        "mechanism": mechanism,
    }


def question_coverage(frame):
    """Complete-case counts per research question, with missing mechanism."""
    work = add_derived(frame) if "has_np_ocf" not in frame.columns else frame
    groups = [("合计", work)]
    if "industry" in work.columns:
        for group, part in work.groupby("industry", observed=False):
            groups.append((str(group), part))
    rows = []
    for industry, part in groups:
        gm = _numeric_col(part, "gross_margin")
        gm_ok = _flag_true(part, "gm_valid") if "gm_valid" in part.columns else gm.notna()
        n_bad_cost = int((part["cost_source"] == "营业总成本").sum()) if "cost_source" in part.columns else 0
        rows.append(_coverage_row(
            part, industry, "毛利率",
            gm.notna(), gm_ok,
            f"成本不是营业成本则无效；营业总成本口径 {n_bad_cost} 家。",
        ))

        dso = _numeric_col(part, "dso")
        dso_anom = _flag_true(part, "dso_anomalous") if "dso_anomalous" in part.columns else pd.Series(False, index=part.index)
        rows.append(_coverage_row(
            part, industry, "DSO",
            dso.notna(), ~dso_anom,
            f"缺应收账款或收入 {int(dso.isna().sum())} 家；>730 天 {int(dso_anom.sum())} 家不进主中位数。",
        ))

        dio = _numeric_col(part, "dio")
        dio_anom = _flag_true(part, "dio_anomalous") if "dio_anomalous" in part.columns else pd.Series(False, index=part.index)
        n_inv_miss = int((part["inventory_status"] == "missing").sum()) if "inventory_status" in part.columns else int(dio.isna().sum())
        rows.append(_coverage_row(
            part, industry, "DIO",
            dio.notna(), ~dio_anom,
            f"缺存货行或成本不是营业成本（缺存货 {n_inv_miss} 家）；缺行不记 0 天。",
        ))

        dpo = _numeric_col(part, "dpo")
        dpo_anom = _flag_true(part, "dpo_anomalous") if "dpo_anomalous" in part.columns else pd.Series(False, index=part.index)
        rows.append(_coverage_row(
            part, industry, "DPO",
            dpo.notna(), ~dpo_anom,
            f"缺应付账款或营业成本 {int(dpo.isna().sum())} 家。",
        ))

        ccc = _numeric_col(part, "ccc")
        ccc_main = ~dso_anom & ~dio_anom & ~dpo_anom
        rows.append(_coverage_row(
            part, industry, "现金周期 CCC",
            ccc.notna(), ccc_main,
            "三项天数都要有；任一 >730 天则不进主中位数。",
        ))

        acc = _numeric_col(part, "accruals_to_revenue")
        n_np_miss = int(_numeric_col(part, "net_profit").isna().sum())
        n_ocf_miss = int(_numeric_col(part, "ocf").isna().sum())
        rows.append(_coverage_row(
            part, industry, "应计/收入",
            acc.notna(), acc.notna(),
            f"要同时有净利润和 OCF。缺净利润 {n_np_miss} 家，缺 OCF {n_ocf_miss} 家（截断保持 NaN）。",
        ))

        if "has_np_ocf" in part.columns:
            both = part["has_np_ocf"].fillna(False).astype(bool)
        else:
            both = _numeric_col(part, "net_profit").notna() & _numeric_col(part, "ocf").notna()
        rows.append(_coverage_row(
            part, industry, "利润与 OCF 符号",
            both, both,
            "分母是同时有净利润和 OCF 的公司，不把缺项当成符号组合。",
        ))

        roe = _numeric_col(part, "roe")
        roe_ok = _flag_true(part, "roe_valid") if "roe_valid" in part.columns else roe.notna()
        n_eq_neg = int(_flag_true(part, "equity_negative").sum()) if "equity_negative" in part.columns else 0
        rows.append(_coverage_row(
            part, industry, "有效 ROE",
            roe.notna(), roe_ok,
            f"缺净利润或权益非正则冻结。权益非正 {n_eq_neg} 家。",
        ))
    return pd.DataFrame(rows)


def sensitivity_table(frame, columns=None):
    """Long table: industry × field × treatment → median, n."""
    work = add_derived(frame) if "has_np_ocf" not in frame.columns else frame
    if any(f"{c}_iqr_out" not in work.columns for c in (columns or SENSITIVITY_COLS) if c in work.columns):
        work = add_iqr_outlier_flags(work)
    cols = [c for c in (columns or SENSITIVITY_COLS) if c in work.columns]
    groups = [("合计", work)]
    if "industry" in work.columns:
        for group, part in work.groupby("industry", observed=False):
            groups.append((str(group), part))
    rows = []
    for industry, part in groups:
        for col in cols:
            for treatment in TREATMENTS:
                value_col = _value_col(part, col, treatment)
                mask = _treatment_mask(part, col, treatment)
                med, n_med = _median_n(part, value_col, where=mask)
                rows.append(
                    {
                        "industry": industry,
                        "field": col,
                        "treatment": treatment,
                        "treatment_label": TREAT_LABEL[treatment],
                        "median": med,
                        "n": n_med,
                    }
                )
    return pd.DataFrame(rows)


def pairwise_spearman(frame, pairs=None):
    """Rank correlation on complete pairs; no scipy."""
    work = frame
    rows = []
    for left, right, label in (pairs or ASSOC_PAIRS):
        if left not in work.columns or right not in work.columns:
            rows.append({"pair": label, "left": left, "right": right, "n": 0, "spearman": np.nan})
            continue
        mask = _median_mask(work, left) & _median_mask(work, right)
        pair = work.loc[mask, [left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        n_pair = int(len(pair))
        if n_pair < 5:
            rho = np.nan
        else:
            rho = float(pair.rank().corr(method="pearson").loc[left, right])
        rows.append({"pair": label, "left": left, "right": right, "n": n_pair, "spearman": rho})
    return pd.DataFrame(rows)


def _sens_cell(sensitivity, industry, field, treatment):
    if sensitivity is None or sensitivity.empty:
        return np.nan, 0
    hit = sensitivity[
        (sensitivity["industry"] == industry)
        & (sensitivity["field"] == field)
        & (sensitivity["treatment"] == treatment)
    ]
    if hit.empty:
        return np.nan, 0
    return hit.iloc[0]["median"], int(hit.iloc[0]["n"])


def _order_holds(sensitivity, field, higher="软件信息", lower="制造"):
    """Which treatments still have higher > lower on this field."""
    if sensitivity is None or sensitivity.empty:
        return []
    held = []
    for treatment in TREATMENTS:
        hi, _ = _sens_cell(sensitivity, higher, field, treatment)
        lo, _ = _sens_cell(sensitivity, lower, field, treatment)
        if pd.notna(hi) and pd.notna(lo) and hi > lo:
            held.append(TREAT_LABEL[treatment])
    return held


def _pct(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _row(summary, industry):
    hit = summary[summary["industry"] == industry]
    if hit.empty:
        return None
    return hit.iloc[0]


def _days(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.0f} 天"


def _n_suffix(row, col):
    key = f"n_{col}"
    if key not in row.index or pd.isna(row[key]):
        return ""
    return f" (n={int(row[key])})"


def _days_n(row, col):
    return _days(row[f"median_{col}"]) + _n_suffix(row, col)


def _pct_n(row, col):
    return _pct(row[f"median_{col}"]) + _n_suffix(row, col)


def _coverage_md(coverage, industry="合计"):
    if coverage is None or coverage.empty:
        return []
    part = coverage[coverage["industry"] == industry]
    if part.empty:
        return []
    lines = [
        "| 问题 | 完整个案 | 主口径 | 缺失 | 机制 |",
        "|------|----------|--------|------|------|",
    ]
    for _, row in part.iterrows():
        lines.append(
            "| {q} | {c}/{g} | {m} | {miss} | {mech} |".format(
                q=row["question"],
                c=int(row["n_complete"]),
                g=int(row["n_group"]),
                m=int(row["n_main"]),
                miss=_pct(row["missing_share"]),
                mech=row["mechanism"],
            )
        )
    return lines


def _sensitivity_md(sensitivity, fields, industries=None):
    if sensitivity is None or sensitivity.empty:
        return []
    industries = industries or list(GROUP_ORDER)
    lines = [
        "| 指标 | 口径 | " + " | ".join(industries) + " |",
        "|------|------|" + "|".join(["------"] * len(industries)) + "|",
    ]
    is_days = {"dso", "dio", "dpo", "ccc", "operating_cycle"}
    for field in fields:
        for treatment in TREATMENTS:
            cells = []
            for ind in industries:
                med, n_med = _sens_cell(sensitivity, ind, field, treatment)
                if pd.isna(med):
                    cells.append("n/a")
                elif field in is_days:
                    cells.append(f"{med:.0f} 天 (n={n_med})")
                else:
                    cells.append(f"{med * 100:.1f}% (n={n_med})")
            lines.append(f"| {field} | {TREAT_LABEL[treatment]} | " + " | ".join(cells) + " |")
    return lines


def render_report(summary, n_total, coverage=None, sensitivity=None, assoc=None):
    """Question-driven findings. Extra tables are optional so old tests still call this."""
    mfg = _row(summary, GROUP_MANUFACTURING)
    sw = _row(summary, GROUP_SOFTWARE)
    other = _row(summary, GROUP_OTHER)
    lines = [
        "# 行业画像：营运资金与盈余质量",
        "",
        f"样本 {n_total} 家，行业来自 `output/pdf/` 子目录（证监会门类），合并为 **制造 / 软件信息 / 其他**。",
        "`00_待分类` 和 PDF 根目录归入其他，不单独建模。",
        "口径跟预处理同一套：分问题完整个案，不跨公司填净利润；中位数；每个格子写 n；",
        "主估计用质量标记 + 730 天帽，IQR 剔除和 1%/99% 缩尾只做稳健对照。分析章节不套用八步顺序。",
        "",
        "## 样本",
        "",
        "| 行业 | 家数 | 毛利可算 | ROE 可算 | 有净利润且有 OCF |",
        "|------|------|----------|----------|------------------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['industry']} | {int(row['n'])} | {int(row['n_gm_valid'])} | {int(row['n_roe_valid'])} | {int(row['n_np_ocf'])} |"
        )

    lines += ["", "## 营运资金周期", ""]
    lines.append(
        "**问题**：软件的现金是否比制造更紧，紧在应收还是存货？"
    )
    lines.append("")
    lines.append(
        "DSO = 应收 / 收入 × 365；DIO = 存货 / 营业成本 × 365（存货为 0 记 0 天，缺存货行不记 0）；"
        "DPO = 应付 / 营业成本 × 365；经营周期 = DSO + DIO；现金周期 CCC = 经营周期 − DPO。"
        "主口径丢掉 >730 天的点；格子里的 n 是该指标完整个案，不是行业总家数。"
    )
    if coverage is not None:
        lines += ["", "合计覆盖：", ""]
        lines += _coverage_md(coverage, "合计")
    lines += [
        "",
        "| 行业 | DSO | DIO | DPO | 经营周期 | CCC | 应收/收入 | 存货/收入 | 存货为 0 | 存货缺失 |",
        "|------|-----|-----|-----|----------|-----|-----------|-----------|----------|----------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {industry} | {dso} | {dio} | {dpo} | {ocyc} | {ccc} | {ar} | {inv} | {z} | {m} |".format(
                industry=row["industry"],
                dso=_days_n(row, "dso"),
                dio=_days_n(row, "dio"),
                dpo=_days_n(row, "dpo"),
                ocyc=_days_n(row, "operating_cycle"),
                ccc=_days_n(row, "ccc"),
                ar=_pct_n(row, "ar_to_revenue"),
                inv=_pct_n(row, "inventory_to_revenue"),
                z=_pct(row["share_inventory_zero"]),
                m=_pct(row["share_inventory_missing"]),
            )
        )
    if sensitivity is not None and not sensitivity.empty:
        lines += ["", "### 稳健（DSO / DIO / CCC）", ""]
        lines.append("IQR 围栏在全样本上算，与 `preprocess.py` 一致。软件只有二十多家，组内 IQR 不稳定。")
        lines.append("")
        lines += _sensitivity_md(sensitivity, ["dso", "dio", "ccc"])
        dso_holds = _order_holds(sensitivity, "dso", higher=GROUP_SOFTWARE, lower=GROUP_MANUFACTURING)
        if dso_holds:
            lines.append("")
            lines.append("软件 DSO 长于制造的口径：" + "、".join(dso_holds) + "。")
        sw_iqr, n_sw_iqr = _sens_cell(sensitivity, GROUP_SOFTWARE, "dso", "iqr")
        mfg_iqr, n_mfg_iqr = _sens_cell(sensitivity, GROUP_MANUFACTURING, "dso", "iqr")
        sw_q, n_sw_q = _sens_cell(sensitivity, GROUP_SOFTWARE, "dso", "quality")
        if (
            dso_holds
            and TREAT_LABEL["iqr"] not in dso_holds
            and pd.notna(sw_iqr)
            and pd.notna(mfg_iqr)
            and pd.notna(sw_q)
        ):
            lines.append(
                "去掉 IQR 离群后软件 {sw:.0f} 天 (n={nsw})、制造 {mfg:.0f} 天 (n={nm})，"
                "方向翻了：主口径软件 {swq:.0f} 天 (n={nswq}) 是少数长账期公司把中位数拉上去的，"
                "不是全行业都半年回款。".format(
                    sw=sw_iqr, nsw=n_sw_iqr, mfg=mfg_iqr, nm=n_mfg_iqr,
                    swq=sw_q, nswq=n_sw_q,
                )
            )

    lines += [
        "",
        "## 盈余质量",
        "",
        "**问题**：利润是否快于经营现金？现金矛盾是「赚钱没现金」还是「亏损仍有现金」？",
        "",
        "应计利润 / 收入 = (净利润 − OCF) / 收入。正值表示利润快于经营现金。"
        "分母「利润为正且 OCF 为负 / 利润为负且 OCF 为正」是同时有两科目的公司。",
        "",
        "| 行业 | 应计/收入 | 毛利率 | 净利率 | 利润为正 | 利润为正且 OCF 为负 | 利润为负且 OCF 为正 |",
        "|------|-----------|--------|--------|----------|---------------------|---------------------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {industry} | {acc} | {gm} | {nm} | {pp} | {gap} | {rev} |".format(
                industry=row["industry"],
                acc=_pct_n(row, "accruals_to_revenue"),
                gm=_pct_n(row, "gross_margin"),
                nm=_pct_n(row, "net_margin"),
                pp=_pct(row["share_profit_pos"]),
                gap=_pct(row["share_profit_pos_ocf_neg"]),
                rev=_pct(row["share_profit_neg_ocf_pos"]),
            )
        )
    if sensitivity is not None and not sensitivity.empty:
        lines += ["", "### 稳健（应计 / 毛利率）", ""]
        lines += _sensitivity_md(sensitivity, ["accruals_to_revenue", "gross_margin"])
    if assoc is not None and not assoc.empty:
        lines += ["", "应计与营运资金的 Spearman（主口径完整个案）：", ""]
        lines += ["| 配对 | n | Spearman |", "|------|---|----------|"]
        for _, row in assoc.iterrows():
            rho = "n/a" if pd.isna(row["spearman"]) else f"{float(row['spearman']):.3f}"
            lines.append(f"| {row['pair']} | {int(row['n'])} | {rho} |")

    lines += ["", "## 读数", ""]
    if mfg is not None and sw is not None:
        if pd.notna(mfg["median_dso"]) and pd.notna(sw["median_dso"]) and sw["median_dso"] > mfg["median_dso"]:
            dso_note = "软件 DSO 更长，更像项目制回款"
        else:
            dso_note = "制造 DSO 不短于软件"
        if pd.notna(mfg["median_dio"]) and pd.notna(sw["median_dio"]) and mfg["median_dio"] > sw["median_dio"]:
            dio_note = "制造存货周转更慢"
        else:
            dio_note = "软件 DIO 并不低（门类里常有软硬一体）"
        dso_iqr_flip = False
        if sensitivity is not None and not sensitivity.empty:
            dso_iqr_flip = TREAT_LABEL["iqr"] not in _order_holds(
                sensitivity, "dso", higher=GROUP_SOFTWARE, lower=GROUP_MANUFACTURING
            )
        if dso_iqr_flip:
            dso_note = dso_note + "；IQR 剔除后方向会翻，长账期在右尾"
        lines.append(
            "1. **营运资金周期**：DSO 制造 {mfg_dso}、软件 {sw_dso}（{dso_note}）；"
            "DIO 制造 {mfg_dio}、软件 {sw_dio}（{dio_note}）；"
            "现金周期制造 {mfg_ccc}、软件 {sw_ccc}。"
            "存货为 0 的占比软件 {sw_z}、制造 {mfg_z}，缺失行不按 0 天算。".format(
                mfg_dso=_days(mfg["median_dso"]),
                sw_dso=_days(sw["median_dso"]),
                mfg_dio=_days(mfg["median_dio"]),
                sw_dio=_days(sw["median_dio"]),
                mfg_ccc=_days(mfg["median_ccc"]),
                sw_ccc=_days(sw["median_ccc"]),
                sw_z=_pct(sw["share_inventory_zero"]),
                mfg_z=_pct(mfg["share_inventory_zero"]),
                dso_note=dso_note,
                dio_note=dio_note,
            )
        )
        gm_note = "软件毛利率更高" if sw["median_gross_margin"] > mfg["median_gross_margin"] else "制造毛利率更高"
        acc_note = (
            "软件应计更高（利润相对现金更超前）"
            if pd.notna(sw["median_accruals_to_revenue"])
            and pd.notna(mfg["median_accruals_to_revenue"])
            and sw["median_accruals_to_revenue"] > mfg["median_accruals_to_revenue"]
            else "制造应计更高，或软件亏损把应计压下去"
        )
        lines.append(
            "2. **盈余质量**：应计/收入制造 {mfg_acc}、软件 {sw_acc}（{acc_note}）；"
            "毛利率软件 {sw_gm}、制造 {mfg_gm}（{gm_note}，仅营业成本口径）；"
            "净利率软件 {sw_nm}、制造 {mfg_nm}。".format(
                mfg_acc=_pct(mfg["median_accruals_to_revenue"]),
                sw_acc=_pct(sw["median_accruals_to_revenue"]),
                sw_gm=_pct(sw["median_gross_margin"]),
                mfg_gm=_pct(mfg["median_gross_margin"]),
                sw_nm=_pct(sw["median_net_margin"]),
                mfg_nm=_pct(mfg["median_net_margin"]),
                acc_note=acc_note,
                gm_note=gm_note,
            )
        )
        other_gap = _pct(other["share_profit_pos_ocf_neg"]) if other is not None else "n/a"
        other_rev = _pct(other["share_profit_neg_ocf_pos"]) if other is not None else "n/a"
        lines.append(
            "3. **利润与现金符号**：利润为正且 OCF 为负，制造 {mfg_gap}、软件 {sw_gap}、其他 {other_gap}；"
            "利润为负且 OCF 为正更常见（制造 {mfg_rev}、软件 {sw_rev}、其他 {other_rev}）。"
            "软件利润为正的只有 {sw_pp}，二元「现金缺口」标签在软件里几乎用不上，应计利润更合适。".format(
                mfg_gap=_pct(mfg["share_profit_pos_ocf_neg"]),
                sw_gap=_pct(sw["share_profit_pos_ocf_neg"]),
                other_gap=other_gap,
                mfg_rev=_pct(mfg["share_profit_neg_ocf_pos"]),
                sw_rev=_pct(sw["share_profit_neg_ocf_pos"]),
                other_rev=other_rev,
                sw_pp=_pct(sw["share_profit_pos"]),
            )
        )
    lines += [
        "",
        "## 局限",
        "",
        "- 软件样本小，中位数对单家公司敏感。",
        "- 「其他」混了批发、科研、待分类和未进子目录的 PDF，不是一个行业。",
        "- 单期年报；净利润覆盖低于营收，应计和符号组合的分母更小。",
        "- 行业标签来自本地 PDF 文件夹，不是交易所实时行业。",
        "- DSO / DIO 用期末余额 / 本年流量，不是严格的平均余额周转。",
        "- IQR 在全样本上算；组内离群点另看 730 天帽。",
        "",
    ]
    return "\n".join(lines)


def _save_fig(path, fig):
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _boxplot_series(frame, group, col):
    part = frame.loc[frame["industry"] == group]
    series = pd.to_numeric(part[col], errors="coerce")
    series = series.where(_median_mask(part, col))
    return series.dropna().values


def plot_boxplots(metrics, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    frame = metrics.copy()
    frame["industry"] = pd.Categorical(frame["industry"], categories=GROUP_ORDER, ordered=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, title) in zip(axes.ravel(), BOXPLOT_COLS):
        if col not in frame.columns:
            ax.set_visible(False)
            continue
        data = [_boxplot_series(frame, g, col) for g in GROUP_ORDER]
        ax.boxplot(data, tick_labels=GROUP_ORDER, showfliers=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    fig.suptitle("三类行业：营运资金天数与盈余质量", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, PORTRAIT_CHART)
    _save_fig(path, fig)
    return path


def plot_wc_cycle(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    industries = summary["industry"].tolist()
    x = np.arange(len(industries))
    width = 0.2
    dso = summary["median_dso"].fillna(0)
    dio = summary["median_dio"].fillna(0)
    dpo = summary["median_dpo"].fillna(0)
    ccc = summary["median_ccc"].fillna(0)
    axes[0].bar(x - 1.5 * width, dso, width, label="DSO", color="#4c78a8")
    axes[0].bar(x - 0.5 * width, dio, width, label="DIO", color="#f58518")
    axes[0].bar(x + 0.5 * width, dpo, width, label="DPO", color="#54a24b")
    axes[0].bar(x + 1.5 * width, ccc, width, label="CCC", color="#e45756")
    axes[0].set_xticks(x, industries)
    axes[0].set_ylabel("天")
    axes[0].set_title("中位数：营运资金周期", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].axhline(0, color="#999999", linewidth=0.8, linestyle="--")

    width2 = 0.35
    gap = summary["share_profit_pos_ocf_neg"].fillna(0) * 100
    rev = summary["share_profit_neg_ocf_pos"].fillna(0) * 100
    axes[1].bar(x - width2 / 2, gap, width2, label="利润>0 且 OCF<0", color="#e45756")
    axes[1].bar(x + width2 / 2, rev, width2, label="利润<0 且 OCF>0", color="#4c78a8")
    axes[1].set_xticks(x, industries)
    axes[1].set_ylabel("占比 (%)")
    axes[1].set_title("盈余质量：利润与经营现金符号", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, WC_CHART)
    _save_fig(path, fig)
    return path


def plot_sensitivity(sensitivity, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    if sensitivity is None or sensitivity.empty:
        return None
    industries = [g for g in GROUP_ORDER if g in set(sensitivity["industry"])]
    treatments = list(TREATMENTS)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    specs = [("dso", "DSO（天）", False), ("accruals_to_revenue", "应计/收入", True)]
    x = np.arange(len(industries))
    width = 0.18
    for ax, (field, title, as_pct) in zip(axes, specs):
        for i, treatment in enumerate(treatments):
            vals = []
            for ind in industries:
                med, _ = _sens_cell(sensitivity, ind, field, treatment)
                if pd.isna(med):
                    vals.append(0.0)
                else:
                    vals.append(med * 100 if as_pct else med)
            ax.bar(x + (i - 1.5) * width, vals, width, label=TREAT_LABEL[treatment], color=colors[i])
        ax.set_xticks(x, industries)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=7)
    fig.suptitle("稳健：主口径 / 含异常天数 / IQR / 缩尾", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, SENSITIVITY_CHART)
    _save_fig(path, fig)
    return path


def load_or_build_metrics(metrics_path, csv_dir):
    if metrics_path and os.path.isfile(metrics_path):
        return pd.read_csv(metrics_path, encoding="utf-8-sig")
    default_metrics = os.path.join(OUTPUT_DIR_DEFAULT, "company_metrics.csv")
    if os.path.isfile(default_metrics):
        return pd.read_csv(default_metrics, encoding="utf-8-sig")
    return build_company_metrics(csv_dir or default_csv_dir())


def main(metrics_path=None, csv_dir=None, pdf_dir=None, output_dir=None, report_path=None):
    output_dir = output_dir or OUTPUT_DIR_DEFAULT
    pdf_dir = pdf_dir or PDF_DIR_DEFAULT
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Industry portrait / 行业画像")
    print(f"PDF dir: {pdf_dir}")
    print("=" * 60)

    metrics = load_or_build_metrics(metrics_path, csv_dir)
    if metrics is None or metrics.empty:
        print("No company metrics.")
        return None
    labeled = attach_industry(metrics, pdf_root=pdf_dir)
    labeled, summary = industry_tables(labeled)
    coverage = question_coverage(labeled)
    sensitivity = sensitivity_table(labeled)
    assoc = pairwise_spearman(labeled)

    assign_path = os.path.join(output_dir, ASSIGN_NAME)
    labeled[["stock_code", "company_name", "year", "industry", "industry_raw"]].to_csv(
        assign_path, index=False, encoding="utf-8-sig"
    )
    med_path = os.path.join(output_dir, MEDIANS_NAME)
    summary.to_csv(med_path, index=False, encoding="utf-8-sig")
    cov_path = os.path.join(output_dir, COVERAGE_NAME)
    coverage.to_csv(cov_path, index=False, encoding="utf-8-sig")
    sens_path = os.path.join(output_dir, SENSITIVITY_NAME)
    sensitivity.to_csv(sens_path, index=False, encoding="utf-8-sig")
    assoc_path = os.path.join(output_dir, ASSOC_NAME)
    assoc.to_csv(assoc_path, index=False, encoding="utf-8-sig")

    report = render_report(
        summary, n_total=len(labeled),
        coverage=coverage, sensitivity=sensitivity, assoc=assoc,
    )
    md_path = os.path.join(output_dir, REPORT_NAME)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report)

    box_path = plot_boxplots(labeled, output_dir)
    wc_path = plot_wc_cycle(summary, output_dir)
    sens_chart = plot_sensitivity(sensitivity, output_dir)

    print(summary.to_string(index=False))
    print(f"wrote {assign_path}")
    print(f"wrote {med_path}")
    print(f"wrote {cov_path}")
    print(f"wrote {sens_path}")
    print(f"wrote {md_path}")
    print(f"wrote {box_path}")
    print(f"wrote {wc_path}")
    if sens_chart:
        print(f"wrote {sens_chart}")
    if report_path:
        print(f"wrote {report_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEEQ industry portraits")
    parser.add_argument("--metrics", default=None, help="company_metrics.csv")
    parser.add_argument("--csv-dir", default=None, help="CSV folder if metrics file is missing")
    parser.add_argument("--pdf-dir", default=None, help="PDF folder with industry subdirs")
    parser.add_argument("--output-dir", default=None, help="Charts and tables")
    parser.add_argument("--report", default=None, help="Optional extra markdown path")
    args = parser.parse_args()
    main(
        metrics_path=args.metrics,
        csv_dir=args.csv_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        report_path=args.report,
    )
