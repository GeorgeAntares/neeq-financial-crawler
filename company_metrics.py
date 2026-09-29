"""
Company-level financial metrics (one row per stock_code + year).

Reads first statement block only (consolidated). A later ``项目`` header is
treated as the parent-company / extra table and ignored. Missing line items
stay NaN; they are not filled from a later block.

Statement preprocessing lives here as quality flags and valid-ratio rules:
gross margin only from 营业成本, ROE/multiplier frozen on non-positive equity,
BS articulation, inventory 0 vs missing, DSO/DIO/DPO caps, accruals.
Revenue >= 100k is sample screening (footnote IDs), not statement cleaning.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from financial_analysis import MIN_REVENUE_CNY, to_numeric_safe

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_255 = os.path.join(ROOT, "output", "analysis", "_csv_255")
CSV_DIR_DEFAULT = CSV_255 if os.path.isdir(CSV_255) else os.path.join(ROOT, "output", "csv")
OUTPUT_DIR_DEFAULT = os.path.join(ROOT, "output", "analysis")

METRICS_NAME = "company_metrics.csv"
COVERAGE_NAME = "company_metrics_coverage.csv"
QUALITY_NAME = "company_metrics_quality.csv"
WINSOR_LIMITS = (0.01, 0.99)
DAYS_PER_YEAR = 365.0
DAYS_ANOMALY = 730.0
BS_REL_TOL = 0.01

ID_COLS = ["stock_code", "company_name", "year"]

AMOUNT_COLS = [
    "revenue",
    "revenue_prior",
    "cogs",
    "total_operating_cost",
    "net_profit",
    "net_profit_prior",
    "total_assets",
    "total_assets_begin",
    "current_assets",
    "current_liabilities",
    "total_liabilities",
    "equity",
    "equity_begin",
    "accounts_receivable",
    "inventory",
    "accounts_payable",
    "ocf",
    "ocf_prior",
]

RATIO_COLS = [
    "gross_margin",
    "net_margin",
    "roe",
    "asset_turnover",
    "equity_multiplier",
    "current_ratio",
    "debt_ratio",
    "ar_to_revenue",
    "inventory_to_revenue",
    "ocf_to_revenue",
    "accruals_to_revenue",
    "dso",
    "dio",
    "dpo",
    "operating_cycle",
    "ccc",
    "revenue_yoy",
    "net_profit_yoy",
    "ocf_yoy",
]

FLAG_COLS = [
    "gm_valid",
    "equity_negative",
    "roe_valid",
    "bs_articulation_ok",
    "bs_gap",
    "bs_rel_gap",
    "np_truncated",
    "ocf_missing_kind",
    "inventory_status",
    "dso_anomalous",
    "dio_anomalous",
    "dpo_anomalous",
]


def parse_csv_filename(filename):
    """Parse ``{code}_{name}_{year}_合并{type}.csv``."""
    stem = os.path.basename(filename).replace(".csv", "")
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    year = parts[2]
    if not (len(year) == 4 and year.isdigit()):
        return None
    return parts[0], parts[1], year


def iter_first_block(frame):
    """Yield rows of the first statement; stop at a later ``项目`` header."""
    if frame is None or frame.empty or len(frame.columns) < 2:
        return
    item_col = frame.columns[0]
    seen_data = False
    for _, row in frame.iterrows():
        item = str(row[item_col]).strip()
        if item in ("项目", "nan", ""):
            if seen_data and item == "项目":
                return
            continue
        seen_data = True
        yield row


def load_statement(csv_dir, statement_type):
    """Load first-block rows for 利润表 / 资产负债表 / 现金流量表."""
    pattern = os.path.join(csv_dir, f"*{statement_type}*.csv")
    records = []
    for path in glob.glob(pattern):
        parsed = parse_csv_filename(path)
        if parsed is None:
            continue
        stock_code, company_name, year = parsed
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if len(frame.columns) < 2:
            continue
        item_col = frame.columns[0]
        current_col = frame.columns[1]
        prior_col = frame.columns[2] if len(frame.columns) >= 3 else None
        order = 0
        for row in iter_first_block(frame):
            item = str(row[item_col]).strip()
            records.append(
                {
                    "stock_code": stock_code,
                    "company_name": company_name,
                    "year": year,
                    "item": item,
                    "current": row[current_col],
                    "prior": row[prior_col] if prior_col is not None else np.nan,
                    "row_order": order,
                }
            )
            order += 1
    return pd.DataFrame(records)


def score_revenue(item):
    text = str(item).strip()
    if text.startswith("其中"):
        return 0
    if "占" in text and "营业收入" in text:
        return 0
    if "营业总收入" in text:
        return 2
    if text in ("营业收入", "一、营业收入", "一.营业收入") or text.startswith("一、营业收入"):
        return 1
    return 0


def score_cogs(item):
    text = str(item).strip()
    if "营业总成本" in text:
        return 0
    if text.startswith("其中：营业成本") or text.startswith("其中:营业成本"):
        return 2
    if "营业成本" in text:
        return 1
    return 0


def score_total_operating_cost(item):
    text = str(item).strip()
    if "营业总成本" in text:
        return 1
    return 0


def score_net_profit(item):
    text = str(item).strip()
    if "净利润" not in text:
        return 0
    if text.startswith("其中"):
        return 0
    if any(token in text for token in ("持续经营", "终止经营", "归属于", "被合并", "少数股东")):
        return 0
    if text.startswith("四") or text.startswith("五"):
        return 2
    if text == "净利润" or text.startswith("净利润"):
        return 1
    return 0


def score_total_assets(item):
    return 1 if str(item).strip() == "资产总计" else 0


def score_current_assets(item):
    return 1 if str(item).strip() == "流动资产合计" else 0


def score_current_liabilities(item):
    return 1 if str(item).strip() == "流动负债合计" else 0


def score_total_liabilities(item):
    return 1 if str(item).strip() == "负债合计" else 0


def score_equity(item):
    text = str(item).strip()
    if "所有者权益" not in text:
        return 0
    if text.endswith("：") or text.endswith(":"):
        return 0
    if "负债和" in text:
        return 0
    if "归属于" in text:
        return 1
    if "合计" in text or text.endswith("合") or "股东权益" in text:
        return 2
    return 0


def score_accounts_receivable(item):
    text = str(item).strip()
    if text == "应收账款":
        return 1
    return 0


def score_inventory(item):
    return 1 if str(item).strip() == "存货" else 0


def score_accounts_payable(item):
    return 1 if str(item).strip() == "应付账款" else 0


def statement_meta(long_df):
    """Row counts of the first block, one row per company-year."""
    empty = pd.DataFrame(columns=["stock_code", "year", "n_rows"])
    if long_df is None or long_df.empty:
        return empty
    return (
        long_df.groupby(["stock_code", "year"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )


def score_ocf(item):
    text = str(item).strip()
    if "投资" in text or "筹资" in text:
        return 0
    if text == "经营活动产生的现金流量净额":
        return 2
    if text.startswith("经营活动产生的现金流量净"):
        return 1
    return 0


def pick_amount(long_df, score_fn, value_col="current"):
    """One numeric amount per company-year; higher score, then earlier row."""
    empty = pd.DataFrame(columns=ID_COLS + ["item", "value"])
    if long_df is None or long_df.empty:
        return empty
    work = long_df.copy()
    work["score"] = work["item"].map(score_fn)
    work = work[work["score"] > 0].copy()
    if work.empty:
        return empty
    work["value"] = work[value_col].map(to_numeric_safe)
    work = work.dropna(subset=["value"])
    if work.empty:
        return empty
    work = work.sort_values(
        ["stock_code", "year", "score", "row_order"],
        ascending=[True, True, False, True],
    )
    picked = work.drop_duplicates(subset=["stock_code", "year"], keep="first")
    return picked[ID_COLS + ["item", "value"]].reset_index(drop=True)


def _merge_pick(base, picked, value_name, item_name=None):
    keys = ["stock_code", "year"]
    if picked is None or picked.empty:
        if base is None or base.empty:
            cols = ID_COLS + [value_name]
            if item_name:
                cols.append(item_name)
            return pd.DataFrame(columns=cols)
        base = base.copy()
        base[value_name] = np.nan
        if item_name:
            base[item_name] = np.nan
        return base
    keep = keys + ["company_name", "value"]
    if item_name:
        keep = keys + ["company_name", "item", "value"]
    right = picked[keep].rename(columns={"value": value_name})
    if item_name:
        right = right.rename(columns={"item": item_name})
    if base is None or base.empty:
        return right
    merged = base.merge(right.drop(columns=["company_name"]), on=keys, how="outer")
    if "company_name" not in merged.columns:
        merged = merged.merge(right[keys + ["company_name"]], on=keys, how="left")
    else:
        extra = right[keys + ["company_name"]].rename(columns={"company_name": "_name"})
        merged = merged.merge(extra, on=keys, how="left")
        merged["company_name"] = merged["company_name"].fillna(merged["_name"])
        merged = merged.drop(columns=["_name"])
    return merged


def _numeric(value, index=None):
    if value is None:
        if index is None:
            return np.nan
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(value, errors="coerce")


def safe_div(numerator, denominator):
    index = getattr(numerator, "index", None)
    if index is None:
        index = getattr(denominator, "index", None)
    num = _numeric(numerator, index)
    den = _numeric(denominator, index)
    out = num / den
    out = out.mask(den == 0)
    return out


def yoy(current, prior):
    """(current - prior) / |prior|; undefined when prior is 0 or missing."""
    cur = pd.to_numeric(current, errors="coerce")
    if prior is None:
        return pd.Series(np.nan, index=getattr(cur, "index", None))
    old = pd.to_numeric(prior, errors="coerce")
    out = (cur - old) / old.abs()
    out = out.mask(old == 0)
    return out


def winsorize_series(series, limits=WINSOR_LIMITS):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 10:
        return values
    lower = values.quantile(limits[0])
    upper = values.quantile(limits[1])
    return values.clip(lower=lower, upper=upper)


def average_level(end_col, begin_col):
    end = pd.to_numeric(end_col, errors="coerce")
    if begin_col is None:
        return end
    begin = pd.to_numeric(begin_col, errors="coerce")
    both = end.notna() & begin.notna()
    avg = (end + begin) / 2.0
    return avg.where(both, end)


def _merge_counts(wide, counts, name):
    keys = ["stock_code", "year"]
    if counts is None or counts.empty:
        wide[name] = 0
        return wide
    right = counts.rename(columns={"n_rows": name})
    merged = wide.merge(right, on=keys, how="left")
    merged[name] = merged[name].fillna(0).astype(int)
    return merged


def days_anomalous(series, cap=DAYS_ANOMALY):
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & ((values > cap) | (values < 0))


def add_ratios_and_flags(wide):
    """Ratios plus statement-quality flags. Mutates a copy."""
    frame = wide.copy()

    cost_from_cogs = frame["cogs"].notna() if "cogs" in frame.columns else pd.Series(False, index=frame.index)
    if "cogs" not in frame.columns:
        frame["cogs"] = np.nan
    frame["cost_source"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[cost_from_cogs, "cost_source"] = "营业成本"
    total_cost = frame["total_operating_cost"] if "total_operating_cost" in frame.columns else pd.Series(np.nan, index=frame.index)
    frame.loc[~cost_from_cogs & total_cost.notna(), "cost_source"] = "营业总成本"
    frame["gm_valid"] = frame["cost_source"] == "营业成本"

    frame["equity_source"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    if "equity_item" in frame.columns:
        parent_eq = frame["equity_item"].fillna("").str.contains("归属于", na=False)
        frame.loc[parent_eq, "equity_source"] = "parent"
        frame.loc[frame["equity"].notna() & ~parent_eq, "equity_source"] = "total"

    frame["avg_assets"] = average_level(frame["total_assets"], frame.get("total_assets_begin"))
    frame["avg_equity"] = average_level(frame["equity"], frame.get("equity_begin"))

    equity_end = pd.to_numeric(frame["equity"], errors="coerce")
    avg_equity = pd.to_numeric(frame["avg_equity"], errors="coerce")
    frame["equity_negative"] = (equity_end < 0) | (avg_equity <= 0)

    assets = pd.to_numeric(frame.get("total_assets"), errors="coerce")
    liabilities = pd.to_numeric(frame.get("total_liabilities"), errors="coerce")
    frame["bs_gap"] = assets - (liabilities + equity_end)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = frame["bs_gap"].abs() / assets.abs()
    rel = rel.mask(assets.fillna(0) == 0)
    frame["bs_rel_gap"] = rel
    can_check = assets.notna() & liabilities.notna() & equity_end.notna()
    frame["bs_articulation_ok"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[can_check, "bs_articulation_ok"] = (rel.loc[can_check] <= BS_REL_TOL).to_numpy()

    inventory = pd.to_numeric(frame.get("inventory"), errors="coerce")
    status = pd.Series("missing", index=frame.index, dtype="object")
    status = status.mask(inventory == 0, "zero")
    status = status.mask(inventory.notna() & (inventory != 0), "positive")
    frame["inventory_status"] = status

    frame["np_truncated"] = frame["net_profit"].isna()
    cf_rows = pd.to_numeric(frame.get("cashflow_n_rows"), errors="coerce").fillna(0)
    frame["ocf_missing_kind"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[frame["ocf"].notna(), "ocf_missing_kind"] = "present"
    frame.loc[frame["ocf"].isna() & (cf_rows > 0), "ocf_missing_kind"] = "truncated"
    frame.loc[frame["ocf"].isna() & (cf_rows <= 0), "ocf_missing_kind"] = "no_statement"

    frame["gross_margin"] = safe_div(frame["revenue"] - frame["cogs"], frame["revenue"])
    frame.loc[~frame["gm_valid"], "gross_margin"] = np.nan
    frame["net_margin"] = safe_div(frame["net_profit"], frame["revenue"])
    frame["roe"] = safe_div(frame["net_profit"], frame["avg_equity"])
    frame["asset_turnover"] = safe_div(frame["revenue"], frame["avg_assets"])
    frame["equity_multiplier"] = safe_div(frame["avg_assets"], frame["avg_equity"])
    frame.loc[frame["equity_negative"], ["roe", "equity_multiplier"]] = np.nan
    frame["roe_valid"] = frame["net_profit"].notna() & avg_equity.notna() & ~frame["equity_negative"]
    frame["dupont_product"] = frame["net_margin"] * frame["asset_turnover"] * frame["equity_multiplier"]
    frame["current_ratio"] = safe_div(frame.get("current_assets"), frame.get("current_liabilities"))
    frame["debt_ratio"] = safe_div(frame.get("total_liabilities"), frame.get("total_assets"))
    frame["ar_to_revenue"] = safe_div(frame.get("accounts_receivable"), frame["revenue"])
    frame["inventory_to_revenue"] = safe_div(frame.get("inventory"), frame["revenue"])
    frame["ocf_to_revenue"] = safe_div(frame.get("ocf"), frame["revenue"])
    np_amt = pd.to_numeric(frame["net_profit"], errors="coerce")
    ocf_amt = pd.to_numeric(frame["ocf"], errors="coerce")
    frame["ocf_minus_np"] = ocf_amt - np_amt
    frame["accruals_to_revenue"] = safe_div(np_amt - ocf_amt, frame["revenue"])
    frame["dso"] = safe_div(frame["accounts_receivable"], frame["revenue"]) * DAYS_PER_YEAR
    cogs = pd.to_numeric(frame["cogs"], errors="coerce")
    frame["dio"] = safe_div(inventory, cogs) * DAYS_PER_YEAR
    frame.loc[~frame["gm_valid"], "dio"] = np.nan
    payable = pd.to_numeric(frame.get("accounts_payable"), errors="coerce")
    frame["dpo"] = safe_div(payable, cogs) * DAYS_PER_YEAR
    frame.loc[~frame["gm_valid"], "dpo"] = np.nan
    frame["operating_cycle"] = pd.to_numeric(frame["dso"], errors="coerce") + pd.to_numeric(frame["dio"], errors="coerce")
    frame["ccc"] = frame["operating_cycle"] - pd.to_numeric(frame["dpo"], errors="coerce")
    frame["dso_anomalous"] = days_anomalous(frame["dso"])
    frame["dio_anomalous"] = days_anomalous(frame["dio"])
    frame["dpo_anomalous"] = days_anomalous(frame["dpo"])
    frame["revenue_yoy"] = yoy(frame["revenue"], frame.get("revenue_prior"))
    frame["net_profit_yoy"] = yoy(frame["net_profit"], frame.get("net_profit_prior"))
    frame["ocf_yoy"] = yoy(frame["ocf"], frame.get("ocf_prior"))

    for col in RATIO_COLS + ["ocf_minus_np"]:
        frame[f"{col}_w"] = winsorize_series(frame[col])
    return frame


def build_company_metrics(csv_dir, min_revenue=MIN_REVENUE_CNY):
    """Wide table: amounts, quality flags, valid ratios, YoY, winsorized ratios."""
    income = load_statement(csv_dir, "利润表")
    balance = load_statement(csv_dir, "资产负债表")
    cashflow = load_statement(csv_dir, "现金流量表")

    revenue = pick_amount(income, score_revenue)
    revenue_prior = pick_amount(income, score_revenue, value_col="prior")
    cogs = pick_amount(income, score_cogs)
    total_cost = pick_amount(income, score_total_operating_cost)
    net_profit = pick_amount(income, score_net_profit)
    net_profit_prior = pick_amount(income, score_net_profit, value_col="prior")

    total_assets = pick_amount(balance, score_total_assets)
    total_assets_begin = pick_amount(balance, score_total_assets, value_col="prior")
    current_assets = pick_amount(balance, score_current_assets)
    current_liabilities = pick_amount(balance, score_current_liabilities)
    total_liabilities = pick_amount(balance, score_total_liabilities)
    equity = pick_amount(balance, score_equity)
    equity_begin = pick_amount(balance, score_equity, value_col="prior")
    ar = pick_amount(balance, score_accounts_receivable)
    inventory = pick_amount(balance, score_inventory)
    payable = pick_amount(balance, score_accounts_payable)

    ocf = pick_amount(cashflow, score_ocf)
    ocf_prior = pick_amount(cashflow, score_ocf, value_col="prior")

    wide = revenue.rename(columns={"value": "revenue", "item": "revenue_item"})
    wide = _merge_pick(wide, revenue_prior, "revenue_prior")
    wide = _merge_pick(wide, cogs, "cogs", item_name="cogs_item")
    wide = _merge_pick(wide, total_cost, "total_operating_cost")
    wide = _merge_pick(wide, net_profit, "net_profit")
    wide = _merge_pick(wide, net_profit_prior, "net_profit_prior")
    wide = _merge_pick(wide, total_assets, "total_assets")
    wide = _merge_pick(wide, total_assets_begin, "total_assets_begin")
    wide = _merge_pick(wide, current_assets, "current_assets")
    wide = _merge_pick(wide, current_liabilities, "current_liabilities")
    wide = _merge_pick(wide, total_liabilities, "total_liabilities")
    wide = _merge_pick(wide, equity, "equity", item_name="equity_item")
    wide = _merge_pick(wide, equity_begin, "equity_begin")
    wide = _merge_pick(wide, ar, "accounts_receivable")
    wide = _merge_pick(wide, inventory, "inventory")
    wide = _merge_pick(wide, payable, "accounts_payable")
    wide = _merge_pick(wide, ocf, "ocf")
    wide = _merge_pick(wide, ocf_prior, "ocf_prior")
    wide = _merge_counts(wide, statement_meta(income), "income_n_rows")
    wide = _merge_counts(wide, statement_meta(balance), "balance_n_rows")
    wide = _merge_counts(wide, statement_meta(cashflow), "cashflow_n_rows")

    if wide.empty:
        return wide

    wide = wide[wide["revenue"].notna() & (wide["revenue"] >= min_revenue)].copy()
    if wide.empty:
        return wide

    wide = add_ratios_and_flags(wide)

    front = ID_COLS + [
        "revenue_item",
        "cost_source",
        "equity_source",
    ]
    ordered = [col for col in front if col in wide.columns]
    ordered += [col for col in AMOUNT_COLS if col in wide.columns]
    ordered += ["avg_assets", "avg_equity"]
    ordered += FLAG_COLS
    ordered += RATIO_COLS + ["dupont_product", "ocf_minus_np"]
    ordered += [f"{col}_w" for col in RATIO_COLS + ["ocf_minus_np"]]
    rest = [
        col
        for col in wide.columns
        if col not in ordered and col not in ("cogs_item", "equity_item")
    ]
    ordered = [col for col in ordered if col in wide.columns]
    wide = wide[ordered + rest].sort_values(["stock_code", "year"]).reset_index(drop=True)
    return wide


def coverage_table(metrics):
    """Non-null counts for the amounts used by later analysis stages."""
    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=["field", "n", "share"])
    n = len(metrics)
    fields = [
        "revenue",
        "cogs",
        "net_profit",
        "total_assets",
        "equity",
        "current_assets",
        "current_liabilities",
        "accounts_receivable",
        "inventory",
        "accounts_payable",
        "ocf",
        "gross_margin",
        "roe",
        "dso",
        "dio",
        "accruals_to_revenue",
        "revenue_yoy",
    ]
    rows = []
    for field in fields:
        if field not in metrics.columns:
            continue
        k = int(metrics[field].notna().sum())
        rows.append({"field": field, "n": k, "share": k / n})
    return pd.DataFrame(rows)


def quality_summary(metrics):
    """Counts for statement-quality flags (preprocessing, not sample screening)."""
    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=["flag", "n", "share"])
    n = len(metrics)
    rows = [{"flag": "firms", "n": n, "share": 1.0}]

    def add(name, mask):
        k = int(mask.fillna(False).sum())
        rows.append({"flag": name, "n": k, "share": k / n})

    add("gm_valid", metrics["gm_valid"] == True)
    add("gm_from_total_cost", metrics["cost_source"] == "营业总成本")
    add("equity_negative", metrics["equity_negative"] == True)
    add("roe_valid", metrics["roe_valid"] == True)
    checked = metrics["bs_articulation_ok"].notna()
    add("bs_checked", checked)
    add("bs_articulation_fail", checked & (metrics["bs_articulation_ok"] == False))
    add("np_truncated", metrics["np_truncated"] == True)
    add("ocf_truncated", metrics["ocf_missing_kind"] == "truncated")
    add("ocf_no_statement", metrics["ocf_missing_kind"] == "no_statement")
    add("inventory_zero", metrics["inventory_status"] == "zero")
    add("inventory_missing", metrics["inventory_status"] == "missing")
    add("dso_anomalous", metrics["dso_anomalous"] == True)
    add("dio_anomalous", metrics["dio_anomalous"] == True)
    return pd.DataFrame(rows)


def default_csv_dir():
    return CSV_DIR_DEFAULT


def main(csv_dir=None, output_dir=None, min_revenue=MIN_REVENUE_CNY):
    csv_dir = csv_dir or default_csv_dir()
    output_dir = output_dir or OUTPUT_DIR_DEFAULT
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Company metrics / 公司级指标库")
    print(f"CSV dir: {csv_dir}")
    print(f"min revenue: {min_revenue:,.0f} CNY")
    print("=" * 60)

    metrics = build_company_metrics(csv_dir, min_revenue=min_revenue)
    if metrics.empty:
        print("No rows after revenue filter.")
        return metrics

    out_path = os.path.join(output_dir, METRICS_NAME)
    if os.path.exists(out_path):
        os.remove(out_path)
    metrics.to_csv(out_path, index=False, encoding="utf-8-sig")

    coverage = coverage_table(metrics)
    cov_path = os.path.join(output_dir, COVERAGE_NAME)
    if os.path.exists(cov_path):
        os.remove(cov_path)
    coverage.to_csv(cov_path, index=False, encoding="utf-8-sig")

    quality = quality_summary(metrics)
    q_path = os.path.join(output_dir, QUALITY_NAME)
    if os.path.exists(q_path):
        os.remove(q_path)
    quality.to_csv(q_path, index=False, encoding="utf-8-sig")

    print(f"rows: {len(metrics)}  companies: {metrics['stock_code'].nunique()}")
    print("coverage:")
    for _, row in coverage.iterrows():
        print(f"  {row['field']}: {int(row['n'])} ({row['share'] * 100:.1f}%)")
    print("quality flags:")
    for _, row in quality.iterrows():
        print(f"  {row['flag']}: {int(row['n'])} ({row['share'] * 100:.1f}%)")
    gm = metrics.loc[metrics["gm_valid"] == True, "gross_margin"] if "gm_valid" in metrics.columns else metrics.get("gross_margin")
    if gm is not None and gm.notna().any():
        print(f"median gross_margin (gm_valid): {gm.median():.3f}")
    dso_ok = metrics.loc[metrics["dso_anomalous"] != True, "dso"] if "dso" in metrics.columns else None
    if dso_ok is not None and dso_ok.notna().any():
        print(f"median DSO (days, cap {DAYS_ANOMALY:.0f}): {dso_ok.median():.1f}")
    if "accruals_to_revenue" in metrics.columns:
        print(f"median accruals/revenue: {metrics['accruals_to_revenue'].median():.3f}")
    print(f"wrote {out_path}")
    print(f"wrote {cov_path}")
    print(f"wrote {q_path}")
    print("dictionary: company_metrics_dictionary.md")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build company-level financial metrics")
    parser.add_argument("--csv-dir", default=None, help="CSV folder (default output/analysis/_csv_255)")
    parser.add_argument("--output-dir", default=None, help="Output folder")
    parser.add_argument(
        "--min-revenue",
        type=float,
        default=MIN_REVENUE_CNY,
        help="Drop firms below this revenue (CNY)",
    )
    args = parser.parse_args()
    main(csv_dir=args.csv_dir, output_dir=args.output_dir, min_revenue=args.min_revenue)
