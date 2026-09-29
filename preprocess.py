"""
Competition-style cleaning / transform pipeline on the company-metrics table.

Order:
  1. backup the working table
  2. field exam
  3. unify missing tokens
  4. missing-value policy (by mechanism, not mean-fill)
  5. outlier detection (IQR + economic caps) and winsorize
  6. dedup / types / drop constant *feature* columns
  7. z-score / min-max (descriptive; models still fit on train only)
  8. before/after charts and a paper log

PDF/CSV under output/ are never overwritten. This stage copies the wide table.
Single-year cross-section: do not interpolate across firms.
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from company_metrics import OUTPUT_DIR_DEFAULT, RATIO_COLS
from financial_analysis import to_numeric_safe

RAW_NAME = "company_metrics_raw.csv"
PREP_NAME = "company_metrics_prep.csv"
EXAM_NAME = "preprocess_field_exam.csv"
MISSING_NAME = "preprocess_missing.csv"
OUTLIER_NAME = "preprocess_outliers.csv"
SCALE_NAME = "preprocess_scaled.csv"
SCALE_STATS_NAME = "preprocess_scale_stats.csv"
LOG_NAME = "preprocess.md"
CHART_NAME = "preprocess_before_after.png"

IQR_K = 1.5
KEY_COLS = ["stock_code", "year"]
ID_KEEP = ["stock_code", "company_name", "year"]

EXAM_COLS = [
    "revenue",
    "cogs",
    "net_profit",
    "total_assets",
    "equity",
    "accounts_receivable",
    "inventory",
    "accounts_payable",
    "ocf",
    "gross_margin",
    "net_margin",
    "roe",
    "current_ratio",
    "debt_ratio",
    "dso",
    "dio",
    "dpo",
    "ccc",
    "accruals_to_revenue",
    "ar_to_revenue",
    "inventory_to_revenue",
    "ocf_to_revenue",
]

SCALE_COLS = [
    "gross_margin",
    "net_margin",
    "current_ratio",
    "debt_ratio",
    "dso",
    "accruals_to_revenue",
    "ar_to_revenue",
    "ocf_to_revenue",
]

PLOT_COLS = [
    ("gross_margin", "毛利率"),
    ("dso", "DSO（天）"),
    ("debt_ratio", "资产负债率"),
    ("accruals_to_revenue", "应计/收入"),
]


def backup_table(src_path, output_dir):
    """Copy the incoming wide table; later steps never write back to PDF/CSV."""
    dest = os.path.join(output_dir, RAW_NAME)
    if os.path.abspath(src_path) != os.path.abspath(dest):
        shutil.copy2(src_path, dest)
    return dest


def unify_missing_frame(frame, columns=None):
    work = frame.copy()
    cols = columns or [c for c in EXAM_COLS if c in work.columns]
    for col in cols:
        work[col] = work[col].map(to_numeric_safe)
    return work


def field_exam(frame, columns=None):
    """n, missing rate, nunique, min / median / max, skew, implausible counts."""
    cols = columns or [c for c in EXAM_COLS if c in frame.columns]
    n = len(frame)
    rows = []
    for col in cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        finite = values.dropna()
        missing = int(values.isna().sum())
        skew = float(finite.skew()) if len(finite) >= 3 else np.nan
        rows.append(
            {
                "field": col,
                "dtype": str(frame[col].dtype),
                "n": n,
                "n_nonnull": int(values.notna().sum()),
                "missing_rate": missing / n if n else np.nan,
                "nunique": int(finite.nunique()),
                "min": float(finite.min()) if len(finite) else np.nan,
                "p25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "p75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "skew": skew,
                "n_negative": int((finite < 0).sum()),
                "n_zero": int((finite == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def missing_policy_table(frame):
    """One row per field: mechanism and action. Mean-fill is not the default."""
    n = max(len(frame), 1)
    specs = [
        (
            "revenue",
            "sample_screen",
            "already_dropped_below_100k",
            "低于 10 万元多半是附注编号，删行属于样本筛选。",
        ),
        (
            "net_profit",
            "parser_truncation",
            "keep_nan",
            "第一张利润表在净利润行前截断。均值填充会把没读到的公司当成平均盈利。",
        ),
        (
            "ocf",
            "truncated_or_no_statement",
            "keep_nan",
            "有现金流量表但缺经营净额行，或根本没有该表。不跨公司插值。",
        ),
        (
            "inventory",
            "row_absent_not_zero",
            "keep_nan_distinct_from_zero",
            "缺存货行不是存货为 0。软件缺行更常见，填 0 会压低 DIO。",
        ),
        (
            "cogs",
            "cost_source",
            "no_fill_from_total_operating_cost",
            "营业总成本含期间费用，不能拿去算毛利。",
        ),
        (
            "gross_margin",
            "invalid_if_not_cogs",
            "set_nan_when_gm_invalid",
            "只有营业成本口径才保留毛利率。",
        ),
        (
            "roe",
            "negative_equity",
            "set_nan",
            "权益非正时 ROE 没有经济含义，冻结而不是填中位数。",
        ),
        (
            "dso",
            "economic_cap",
            "flag_exclude_from_median",
            "天数 > 730 视为异常周转，中位数不用这些点；原始值保留。",
        ),
    ]
    rows = []
    for field, mechanism, action, reason in specs:
        if field not in frame.columns:
            miss = np.nan
        else:
            miss = float(pd.to_numeric(frame[field], errors="coerce").isna().mean())
        extra = ""
        if field == "net_profit" and "np_truncated" in frame.columns:
            extra = f"截断 {int((frame['np_truncated'] == True).sum())} 家。"
        if field == "ocf" and "ocf_missing_kind" in frame.columns:
            extra = (
                f"truncated={int((frame['ocf_missing_kind'] == 'truncated').sum())}，"
                f"no_statement={int((frame['ocf_missing_kind'] == 'no_statement').sum())}。"
            )
        if field == "inventory" and "inventory_status" in frame.columns:
            extra = (
                f"zero={int((frame['inventory_status'] == 'zero').sum())}，"
                f"missing={int((frame['inventory_status'] == 'missing').sum())}。"
            )
        rows.append(
            {
                "field": field,
                "missing_rate": miss,
                "mechanism": mechanism,
                "action": action,
                "reason": (reason + extra).strip(),
                "n": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def iqr_bounds(series, k=IQR_K):
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if len(finite) < 8:
        return np.nan, np.nan
    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def winsorize_iqr(series, k=IQR_K):
    """Clip to Tukey fences. Used as an alternative cap, not the default treatment."""
    values = pd.to_numeric(series, errors="coerce")
    low, high = iqr_bounds(values, k=k)
    if pd.isna(low) or pd.isna(high):
        return values
    return values.clip(lower=low, upper=high)


def outlier_table(frame, columns=None, k=IQR_K):
    """IQR count vs 1%/99% winsorize. Large revenue is kept as a real extreme."""
    cols = columns or [c for c in RATIO_COLS if c in frame.columns]
    rows = []
    n = len(frame)
    for col in cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        finite = values.dropna()
        low, high = iqr_bounds(values, k=k)
        if pd.isna(low):
            n_out = 0
        else:
            n_out = int(((finite < low) | (finite > high)).sum())
        wcol = f"{col}_w"
        treatment = "winsorize_1_99"
        if col in ("dso", "dio", "dpo", "ccc", "operating_cycle"):
            treatment = "economic_cap_730_and_winsorize_1_99"
        rows.append(
            {
                "field": col,
                "n_nonnull": int(values.notna().sum()),
                "iqr_low": low,
                "iqr_high": high,
                "n_iqr_out": n_out,
                "share_iqr_out": n_out / n if n else np.nan,
                "treatment": treatment,
                "has_winsor_col": wcol in frame.columns,
            }
        )
    if "revenue" in frame.columns:
        values = pd.to_numeric(frame["revenue"], errors="coerce")
        low, high = iqr_bounds(values, k=k)
        finite = values.dropna()
        n_out = int(((finite < low) | (finite > high)).sum()) if pd.notna(low) else 0
        rows.append(
            {
                "field": "revenue",
                "n_nonnull": int(values.notna().sum()),
                "iqr_low": low,
                "iqr_high": high,
                "n_iqr_out": n_out,
                "share_iqr_out": n_out / n if n else np.nan,
                "treatment": "keep_real_extreme",
                "has_winsor_col": False,
            }
        )
    return pd.DataFrame(rows)


def add_iqr_flags(frame, columns=None, k=IQR_K):
    work = frame.copy()
    cols = columns or [c for c in RATIO_COLS if c in work.columns]
    for col in cols:
        values = pd.to_numeric(work[col], errors="coerce")
        low, high = iqr_bounds(values, k=k)
        flag = pd.Series(False, index=work.index)
        if pd.notna(low):
            flag = values.notna() & ((values < low) | (values > high))
        work[f"{col}_iqr_out"] = flag
    return work


def drop_duplicate_keys(frame, keys=None):
    keys = list(keys or KEY_COLS)
    present = [k for k in keys if k in frame.columns]
    if not present:
        return frame.copy(), 0
    n_dup = int(frame.duplicated(subset=present).sum())
    return frame.drop_duplicates(subset=present, keep="first").copy(), n_dup


def constant_feature_columns(frame, protect=None):
    protect = set(protect or ID_KEEP)
    names = []
    for col in frame.columns:
        if col in protect:
            continue
        nunique = frame[col].nunique(dropna=False)
        if nunique <= 1:
            names.append(col)
    return names


def zscore_minmax(frame, columns=None):
    """Descriptive scaling on the analysis table. Prediction scripts still fit on train."""
    cols = columns or [c for c in SCALE_COLS if c in frame.columns]
    scaled = frame[ID_KEEP].copy() if set(ID_KEEP).issubset(frame.columns) else pd.DataFrame(index=frame.index)
    stats_rows = []
    for col in cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        mu = float(values.mean()) if values.notna().any() else np.nan
        sd = float(values.std(ddof=0)) if values.notna().any() else np.nan
        lo = float(values.min()) if values.notna().any() else np.nan
        hi = float(values.max()) if values.notna().any() else np.nan
        if pd.notna(sd) and sd > 0:
            scaled[f"{col}_z"] = (values - mu) / sd
        else:
            scaled[f"{col}_z"] = np.nan
        span = hi - lo
        if pd.notna(span) and span > 0:
            scaled[f"{col}_minmax"] = (values - lo) / span
        else:
            scaled[f"{col}_minmax"] = np.nan
        stats_rows.append(
            {
                "field": col,
                "mean": mu,
                "std": sd,
                "min": lo,
                "max": hi,
                "direction": _direction(col),
            }
        )
    return scaled, pd.DataFrame(stats_rows)


def _direction(col):
    cost = {"debt_ratio", "dso", "dio", "dpo", "ccc", "operating_cycle", "accruals_to_revenue"}
    if col in cost:
        return "cost"
    return "benefit"


def apply_pipeline(frame):
    """Run steps 3–7 on a copy. Returns prep frame plus tables."""
    work = unify_missing_frame(frame)
    exam = field_exam(work)
    missing = missing_policy_table(work)
    outliers = outlier_table(work)
    work = add_iqr_flags(work)
    work, n_dup = drop_duplicate_keys(work)
    constants = constant_feature_columns(work)
    scaled, scale_stats = zscore_minmax(work)
    extras = {"n_dup": n_dup, "constant_features": constants}
    return work, exam, missing, outliers, scaled, scale_stats, extras


def _pct(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    if number != 0 and abs(number) < 1e-4:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def render_log(exam, missing, outliers, extras, n_rows):
    lines = [
        "# 数据清洗与预处理",
        "",
        "按竞赛常用顺序：备份 → 字段体检 → 统一缺失标记 → 按机制处理缺失 → 异常值 → 去重/类型 → 无量纲化 → 处理前后对比。",
        "对象是公司级宽表（PDF/CSV 原件不动）。单期截面，不按城市×年份对整张表插值，也不跨公司填净利润。",
        "",
        f"工作副本 {n_rows} 行。主键重复 {extras.get('n_dup', 0)} 条（已按 stock_code+year 去重，保留首行）。",
        "",
        "## 1. 备份",
        "",
        "`output/pdf/` 与 `_csv_255` 不改。本阶段把宽表复制为 `company_metrics_raw.csv`，处理结果写 `company_metrics_prep.csv`。",
        "",
        "## 2. 字段体检",
        "",
        "| 字段 | 缺失率 | 唯一值 | 最小 | 中位数 | 最大 | 偏度 | 负数个数 |",
        "|------|--------|--------|------|--------|------|------|----------|",
    ]
    for _, row in exam.iterrows():
        lines.append(
            "| {f} | {m} | {u} | {mn} | {md} | {mx} | {sk} | {neg} |".format(
                f=row["field"],
                m=_pct(row["missing_rate"]),
                u=int(row["nunique"]),
                mn=_num(row["min"], 3),
                md=_num(row["median"], 3),
                mx=_num(row["max"], 3),
                sk=_num(row["skew"], 2),
                neg=int(row["n_negative"]),
            )
        )
    lines += [
        "",
        "偏度明显的比率用中位数和分位截尾，不用均值描述中心。",
        "",
        "## 3–4. 缺失标记与缺失机制",
        "",
        "`--`、`—`、`未知`、空格等先变成真正的 NaN，再看机制。",
        "",
        "| 字段 | 缺失率 | 机制 | 做法 | 理由 |",
        "|------|--------|------|------|------|",
    ]
    for _, row in missing.iterrows():
        lines.append(
            f"| {row['field']} | {_pct(row['missing_rate'])} | {row['mechanism']} | {row['action']} | {row['reason']} |"
        )
    lines += [
        "",
        "分类信息（存货缺失、成本口径、截断）单独成列，相当于把「缺失」本身留下。",
        "",
        "## 5. 异常值",
        "",
        "IQR（$Q_1-1.5\\,IQR$）用来**计数**离群点。处理默认 1%/99% 缩尾，保留样本量。",
        "DSO/DIO/DPO 另加 730 天经济帽。营收的 IQR 离群当作真实规模差异，不盖帽、不删。",
        "",
        "| 字段 | IQR 下界 | IQR 上界 | 离群家数 | 处理 |",
        "|------|----------|----------|----------|------|",
    ]
    for _, row in outliers.iterrows():
        lines.append(
            "| {f} | {lo} | {hi} | {n} | {t} |".format(
                f=row["field"],
                lo=_num(row["iqr_low"], 3),
                hi=_num(row["iqr_high"], 3),
                n=int(row["n_iqr_out"]),
                t=row["treatment"],
            )
        )
    const = extras.get("constant_features") or []
    const_txt = "、".join(const) if const else "无（标识列如 year 即使取值相同也保留）"
    lines += [
        "",
        "## 6. 去重、类型、常量列",
        "",
        f"- 主键 `stock_code` + `year` 去重。",
        f"- 金额/比率列 `pd.to_numeric(..., errors='coerce')`。",
        f"- 全相同的特征列：{const_txt}。",
        "",
        "## 7. 无量纲化与正向化",
        "",
        "描述用的 Z-score / Min-Max 写在 `preprocess_scaled.csv`（参数来自本表，只供对照）。",
        "真正做预测时，标准化只在**训练折**上 fit，再应用到验证/测试，见 `cash_gap_model.py` 与 `cashflow_features.split_and_scale`。",
        "",
        "- 效益型（越大越好）：毛利率、净利率、ROE、流动比率、OCF/收入。",
        "- 成本型（越小越好）：资产负债率、DSO、DIO、CCC；应计/收入在盈余质量里是「利润相对现金超前」的程度。",
        "- 本报告主分析用原始经济含义的比率和天数，不做熵权/TOPSIS 综合得分。",
        "",
        "行业是无序三类，模型里独热并丢掉「其他」。",
        "",
        "## 8. 处理前后",
        "",
        "图 `output/analysis/preprocess_before_after.png`：毛利率、DSO、资产负债率、应计/收入的原始分布 vs 1%/99% 截尾。",
        "截尾列给画像箱线和模型用；核对单家公司看无 `_w` 的原始列。",
        "",
        "## 不在本阶段做的事",
        "",
        "- 不按新三板/创业板代码再筛（样本范围，不是清洗）。",
        "- 不把母公司净利润拼到合并营收上。",
        "- 不对整张表做线性插值；同比只用表内上期列。",
        "",
    ]
    return "\n".join(lines)


def plot_before_after(frame, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.ravel(), PLOT_COLS):
        raw = pd.to_numeric(frame.get(col), errors="coerce")
        wins = pd.to_numeric(frame.get(f"{col}_w"), errors="coerce")
        data = [raw.dropna().values, wins.dropna().values if wins is not None else np.array([])]
        if len(data[1]) == 0:
            data = [data[0]]
            labels = ["原始"]
        else:
            labels = ["原始", "1%/99% 截尾"]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    fig.suptitle("预处理前后（缩尾保留样本量）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, CHART_NAME)
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def load_metrics(metrics_path):
    path = metrics_path or os.path.join(OUTPUT_DIR_DEFAULT, "company_metrics.csv")
    if not os.path.isfile(path):
        return pd.DataFrame(), path
    return pd.read_csv(path, encoding="utf-8-sig"), path


def main(metrics_path=None, output_dir=None, report_path=None):
    output_dir = output_dir or OUTPUT_DIR_DEFAULT
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Preprocess / 数据清洗与预处理")
    print("=" * 60)

    frame, src = load_metrics(metrics_path)
    if frame.empty:
        print("No company_metrics.csv. Run company_metrics.py first.")
        return None

    backup_table(src, output_dir)
    prep, exam, missing, outliers, scaled, scale_stats, extras = apply_pipeline(frame)

    exam.to_csv(os.path.join(output_dir, EXAM_NAME), index=False, encoding="utf-8-sig")
    missing.to_csv(os.path.join(output_dir, MISSING_NAME), index=False, encoding="utf-8-sig")
    outliers.to_csv(os.path.join(output_dir, OUTLIER_NAME), index=False, encoding="utf-8-sig")
    scaled.to_csv(os.path.join(output_dir, SCALE_NAME), index=False, encoding="utf-8-sig")
    scale_stats.to_csv(os.path.join(output_dir, SCALE_STATS_NAME), index=False, encoding="utf-8-sig")
    prep_path = os.path.join(output_dir, PREP_NAME)
    if os.path.exists(prep_path):
        os.remove(prep_path)
    prep.to_csv(prep_path, index=False, encoding="utf-8-sig")

    log = render_log(exam, missing, outliers, extras, n_rows=len(prep))
    md_path = os.path.join(output_dir, LOG_NAME)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(log)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(log)

    chart = plot_before_after(prep, output_dir)
    print(exam[["field", "missing_rate", "median", "skew"]].to_string(index=False))
    print(f"wrote {os.path.join(output_dir, RAW_NAME)}")
    print(f"wrote {prep_path}")
    print(f"wrote {md_path}")
    print(f"wrote {chart}")
    if report_path:
        print(f"wrote {report_path}")
    return exam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Field exam, missing policy, IQR, scaling log")
    parser.add_argument("--metrics", default=None, help="company_metrics.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report", default=None, help="Optional extra markdown path")
    args = parser.parse_args()
    main(metrics_path=args.metrics, output_dir=args.output_dir, report_path=args.report)
