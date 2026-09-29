"""
Industry DuPont ROE decomposition, identity check, Spearman, and SVD PCA.

Cover: how net margin, turnover and leverage differ by industry.
Complete-case is by question (valid ROE), not a global drop of the 195-firm table.
IQR drop and winsorized columns are robustness. Identity and PCA stay appendix.
PCA is numpy-only (no sklearn) so CI can run the math without the ML extra.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from company_metrics import OUTPUT_DIR_DEFAULT
from industry_groups import GROUP_MANUFACTURING, GROUP_ORDER, GROUP_SOFTWARE, PDF_DIR_DEFAULT
from industry_portrait import attach_industry
from preprocess import add_iqr_flags

IDENTITY_ATOL = 1e-8

PCA_SPEC = [
    ("log_revenue", "log营收", "规模"),
    ("log_assets", "log资产", "规模"),
    ("debt_ratio_w", "资产负债率", "杠杆"),
    ("equity_multiplier_w", "权益乘数", "杠杆"),
    ("current_ratio_w", "流动比率", "杠杆"),
    ("ar_to_revenue_w", "应收/收入", "现金"),
    ("inventory_to_revenue_w", "存货/收入", "现金"),
    ("ocf_to_revenue_w", "OCF/收入", "现金"),
]
PCA_COLS = [row[0] for row in PCA_SPEC]
PCA_LABELS = {row[0]: row[1] for row in PCA_SPEC}
PCA_THEME = {row[0]: row[2] for row in PCA_SPEC}

CORR_COLS = [
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
    "log_revenue",
]
CORR_LABELS = {
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    "roe": "ROE",
    "asset_turnover": "周转",
    "equity_multiplier": "乘数",
    "current_ratio": "流动比率",
    "debt_ratio": "资产负债率",
    "ar_to_revenue": "应收/收入",
    "inventory_to_revenue": "存货/收入",
    "ocf_to_revenue": "OCF/收入",
    "accruals_to_revenue": "应计/收入",
    "dso": "DSO",
    "log_revenue": "log营收",
}

CHART_NAME = "dupont_pca.png"
FACTORS_CHART = "dupont_factors.png"
SPEARMAN_CHART = "dupont_spearman.png"
REPORT_NAME = "dupont_pca.md"
IDENTITY_NAME = "dupont_check.csv"
INDUSTRY_DUPONT_NAME = "dupont_industry.csv"
CORR_NAME = "correlation_spearman.csv"
LOADINGS_NAME = "pca_loadings.csv"
VARIANCE_NAME = "pca_variance.csv"
COVERAGE_NAME = "dupont_coverage.csv"
SENSITIVITY_NAME = "dupont_sensitivity.csv"

DUPONT_TREATMENTS = ("quality", "iqr", "winsor")
DUPONT_TREAT_LABEL = {
    "quality": "主口径",
    "iqr": "去掉 ROE 的 IQR 离群",
    "winsor": "1%/99% 缩尾列",
}


def add_size_logs(metrics):
    frame = metrics.copy()
    revenue = pd.to_numeric(frame.get("revenue"), errors="coerce")
    assets = pd.to_numeric(frame.get("avg_assets"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_revenue"] = np.where(revenue > 0, np.log10(revenue), np.nan)
        frame["log_assets"] = np.where(assets > 0, np.log10(assets), np.nan)
    return frame


def industry_dupont_table(metrics, colmap=None, extra_mask=None):
    """Median NM / turnover / leverage / ROE by industry on valid-ROE rows."""
    mapping = {
        "roe": "roe",
        "net_margin": "net_margin",
        "asset_turnover": "asset_turnover",
        "equity_multiplier": "equity_multiplier",
        "dupont_product": "dupont_product",
    }
    if colmap:
        mapping.update(colmap)
    need = [mapping["roe"], mapping["net_margin"], mapping["asset_turnover"], mapping["equity_multiplier"]]
    work = metrics.copy()
    if extra_mask is not None:
        mask = extra_mask.reindex(work.index).fillna(False).astype(bool)
        work = work.loc[mask].copy()
    if "industry" not in work.columns:
        work["industry"] = "其他"
    for col in need:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = np.nan
    if "roe_valid" in work.columns:
        work = work[work["roe_valid"] == True]
    present = [c for c in need if c in work.columns]
    work = work.dropna(subset=present)
    rows = []
    groups = list(GROUP_ORDER)
    extra = [g for g in work["industry"].dropna().unique() if g not in groups]
    for group in groups + extra:
        part = work[work["industry"] == group]
        empty = {
            "industry": group,
            "n": 0,
            "median_net_margin": np.nan,
            "median_asset_turnover": np.nan,
            "median_equity_multiplier": np.nan,
            "median_roe": np.nan,
            "product_of_medians": np.nan,
            "median_dupont_product": np.nan,
        }
        if part.empty:
            rows.append(empty)
            continue
        nm = float(part[mapping["net_margin"]].median())
        at = float(part[mapping["asset_turnover"]].median())
        em = float(part[mapping["equity_multiplier"]].median())
        roe = float(part[mapping["roe"]].median())
        product = nm * at * em
        prod_col = mapping["dupont_product"]
        med_prod = float(part[prod_col].median()) if prod_col in part.columns else product
        rows.append(
            {
                "industry": group,
                "n": int(len(part)),
                "median_net_margin": nm,
                "median_asset_turnover": at,
                "median_equity_multiplier": em,
                "median_roe": roe,
                "product_of_medians": product,
                "median_dupont_product": med_prod,
            }
        )
    return pd.DataFrame(rows)


def dupont_coverage(metrics):
    """Why a firm is in or out of the DuPont complete case."""
    n = int(len(metrics))
    np_ok = pd.to_numeric(metrics.get("net_profit"), errors="coerce").notna() if n else pd.Series(dtype=bool)
    n_np = int(np_ok.sum()) if n else 0
    n_trunc = int((metrics["np_truncated"] == True).sum()) if "np_truncated" in metrics.columns else int((~np_ok).sum()) if n else 0
    n_eq_neg = int((metrics["equity_negative"] == True).sum()) if "equity_negative" in metrics.columns else 0
    if "roe_valid" in metrics.columns:
        n_roe = int((metrics["roe_valid"] == True).sum())
    else:
        n_roe = int(pd.to_numeric(metrics.get("roe"), errors="coerce").notna().sum()) if n else 0
    four = metrics.copy()
    for col in ("roe", "net_margin", "asset_turnover", "equity_multiplier"):
        if col in four.columns:
            four[col] = pd.to_numeric(four[col], errors="coerce")
        else:
            four[col] = np.nan
    if "roe_valid" in four.columns:
        four = four[four["roe_valid"] == True]
    four = four.dropna(subset=["roe", "net_margin", "asset_turnover", "equity_multiplier"])
    n_four = int(len(four))
    n_at = int(pd.to_numeric(metrics.get("asset_turnover"), errors="coerce").notna().sum()) if n else 0
    return pd.DataFrame(
        [
            {"item": "有效营收", "n": n, "mechanism": "样本筛选：营收 ≥ 10 万元。"},
            {"item": "有净利润", "n": n_np, "mechanism": f"缺的主要是第一张利润表截断（{n_trunc} 家），保持 NaN。"},
            {"item": "权益非正冻结", "n": n_eq_neg, "mechanism": "ROE / 乘数没有经济含义，不进杜邦。"},
            {"item": "有效 ROE", "n": n_roe, "mechanism": "有净利润、有平均权益、且非负权益。"},
            {"item": "有总资产周转", "n": n_at, "mechanism": "缺平均资产则周转缺失。"},
            {"item": "四项齐全", "n": n_four, "mechanism": "净利率、周转、乘数、ROE 同时非空，用于恒等式。"},
        ]
    )


def industry_dupont_sensitivity(metrics):
    """Valid-ROE medians vs drop ROE IQR outliers vs winsorized columns."""
    work = metrics.copy()
    if "roe_iqr_out" not in work.columns:
        cols = [c for c in ("roe", "net_margin") if c in work.columns]
        if cols:
            work = add_iqr_flags(work, columns=cols)
    rows = []
    quality = industry_dupont_table(work)
    quality["treatment"] = "quality"
    quality["treatment_label"] = DUPONT_TREAT_LABEL["quality"]
    rows.append(quality)

    if "roe_iqr_out" in work.columns:
        keep = work["roe_iqr_out"] != True
        iqr = industry_dupont_table(work, extra_mask=keep)
    else:
        iqr = quality.copy()
    iqr["treatment"] = "iqr"
    iqr["treatment_label"] = DUPONT_TREAT_LABEL["iqr"]
    rows.append(iqr)

    wins_cols = ["roe_w", "net_margin_w", "asset_turnover_w", "equity_multiplier_w"]
    if all(c in work.columns for c in wins_cols):
        wins = industry_dupont_table(
            work,
            colmap={
                "roe": "roe_w",
                "net_margin": "net_margin_w",
                "asset_turnover": "asset_turnover_w",
                "equity_multiplier": "equity_multiplier_w",
            },
        )
    else:
        wins = quality.copy()
    wins["treatment"] = "winsor"
    wins["treatment_label"] = DUPONT_TREAT_LABEL["winsor"]
    rows.append(wins)
    return pd.concat(rows, ignore_index=True)


def dupont_identity(metrics, atol=IDENTITY_ATOL):
    """ROE should equal net_margin × turnover × leverage when all four exist."""
    if not {"roe", "dupont_product"}.issubset(metrics.columns):
        return {"n": 0, "max_abs_gap": np.nan, "median_abs_gap": np.nan, "n_match": 0}
    work = metrics[["roe", "dupont_product"]].apply(pd.to_numeric, errors="coerce").dropna()
    if work.empty:
        return {"n": 0, "max_abs_gap": np.nan, "median_abs_gap": np.nan, "n_match": 0}
    gap = (work["dupont_product"] - work["roe"]).abs()
    return {
        "n": int(len(work)),
        "max_abs_gap": float(gap.max()),
        "median_abs_gap": float(gap.median()),
        "n_match": int((gap <= atol).sum()),
    }


def spearman_matrix(frame, columns):
    """Pairwise Spearman via ranks + Pearson; no scipy required."""
    cols = [col for col in columns if col in frame.columns]
    ranks = frame[cols].apply(pd.to_numeric, errors="coerce").rank()
    return ranks.corr(method="pearson")


def dupont_factor_assoc(metrics):
    """Spearman of ROE with each DuPont factor (complete rows)."""
    cols = ["roe", "net_margin", "asset_turnover", "equity_multiplier"]
    work = metrics[cols].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    for factor in cols[1:]:
        rows.append(
            {
                "factor": factor,
                "n": int(len(work)),
                "spearman_with_roe": float(spearman_matrix(work, ["roe", factor]).loc["roe", factor])
                if len(work) >= 3
                else np.nan,
                "median": float(work[factor].median()) if len(work) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def log_variance_shares(metrics):
    """
    For firms with positive NM, turnover, leverage and ROE:
    log(ROE) = log(NM) + log(AT) + log(EM).
    Report Var(log factor) / Var(log ROE); they need not sum to 1 (covariances).
    """
    work = metrics[["roe", "net_margin", "asset_turnover", "equity_multiplier"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    work = work[
        (work["roe"] > 0)
        & (work["net_margin"] > 0)
        & (work["asset_turnover"] > 0)
        & (work["equity_multiplier"] > 0)
    ]
    if len(work) < 5:
        return pd.DataFrame(columns=["factor", "n", "var_share"])
    log_roe = np.log(work["roe"])
    denom = float(log_roe.var(ddof=1))
    mapping = {
        "net_margin": np.log(work["net_margin"]),
        "asset_turnover": np.log(work["asset_turnover"]),
        "equity_multiplier": np.log(work["equity_multiplier"]),
    }
    rows = []
    for name, series in mapping.items():
        share = float(series.var(ddof=1) / denom) if denom else np.nan
        rows.append({"factor": name, "n": int(len(work)), "var_share": share})
    return pd.DataFrame(rows)


def standardize(matrix):
    """Column mean 0, std 1 (population std, 0-std columns stay 0)."""
    data = np.asarray(matrix, dtype=float)
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=0)
    scaled = np.zeros_like(data)
    nonzero = std > 0
    scaled[:, nonzero] = (data[:, nonzero] - mean[nonzero]) / std[nonzero]
    return scaled, mean, std


def fit_pca(matrix):
    """
    SVD PCA on an already-standardized n×p matrix.

    Loadings are V (p×k); scores are U S; variance uses S²/(n-1) like sklearn.
    """
    data = np.asarray(matrix, dtype=float)
    n_obs, n_feat = data.shape
    if n_obs < 2 or n_feat < 1:
        raise ValueError("PCA needs at least 2 rows")
    u, singular, vt = np.linalg.svd(data, full_matrices=False)
    explained = (singular ** 2) / max(n_obs - 1, 1)
    total = explained.sum()
    ratio = explained / total if total else np.zeros_like(explained)
    loadings = vt.T
    scores = u * singular
    return loadings, ratio, scores


def theme_for_loadings(loadings_col, feature_names):
    """Theme of the largest absolute loading (the variable people actually read)."""
    values = np.abs(np.asarray(loadings_col, dtype=float))
    return PCA_THEME[feature_names[int(np.argmax(values))]]


def pca_tables(metrics):
    frame = add_size_logs(metrics)
    complete = frame.dropna(subset=PCA_COLS).copy()
    if len(complete) < 8:
        empty_load = pd.DataFrame(columns=["feature", "label", "theme"] + [f"PC{i}" for i in range(1, 4)])
        empty_var = pd.DataFrame(columns=["component", "explained_ratio", "cumulative", "theme"])
        return complete, empty_load, empty_var
    scaled, _, _ = standardize(complete[PCA_COLS].to_numpy(dtype=float))
    loadings, ratio, _ = fit_pca(scaled)
    n_pc = loadings.shape[1]
    load_df = pd.DataFrame({"feature": PCA_COLS, "label": [PCA_LABELS[c] for c in PCA_COLS],
                            "theme": [PCA_THEME[c] for c in PCA_COLS]})
    for i in range(n_pc):
        load_df[f"PC{i + 1}"] = loadings[:, i]
    var_rows = []
    running = 0.0
    for i, share in enumerate(ratio):
        running += float(share)
        var_rows.append(
            {
                "component": f"PC{i + 1}",
                "explained_ratio": float(share),
                "cumulative": running,
                "theme": theme_for_loadings(loadings[:, i], PCA_COLS) if i < 3 else "",
            }
        )
    return complete, load_df, pd.DataFrame(var_rows)


def _pct(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value, digits=4):
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    if number != 0 and abs(number) < 1e-6:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def render_report(
    identity, assoc, var_shares, corr, load_df, var_df, n_metrics, n_pca,
    industry_df=None, coverage=None, sensitivity=None,
):
    lines = [
        "# 行业杜邦分解",
        "",
        f"样本来自 `company_metrics.csv`（{n_metrics} 家有效营收）。",
        "杜邦：ROE = 净利率 × 总资产周转 × 权益乘数（资产/权益用期初期末平均）。",
        "这是第三个分析问题，完整个案按有效 ROE 取，不把 195 家硬删成一张表。",
        "权益非正的公司不进入 ROE 和乘数。中位数之积不等于乘积的中位数，两列都报。",
        "IQR 剔除和缩尾列只做稳健，不改主口径。",
        "",
    ]
    if coverage is not None and not coverage.empty:
        lines += [
            "## 谁进杜邦样本",
            "",
            "| 口径 | 家数 | 机制 |",
            "|------|------|------|",
        ]
        for _, row in coverage.iterrows():
            lines.append(f"| {row['item']} | {int(row['n'])} | {row['mechanism']} |")
        lines.append("")
    lines += [
        "## 分行业 ROE",
        "",
        "| 行业 | 家数 | 净利率 | 总资产周转 | 权益乘数 | 中位数 ROE | 中位数之积 |",
        "|------|------|--------|------------|----------|------------|------------|",
    ]
    if industry_df is not None and not industry_df.empty:
        for _, row in industry_df.iterrows():
            lines.append(
                "| {ind} | {n} | {nm} | {at} | {em} | {roe} | {prod} |".format(
                    ind=row["industry"],
                    n=int(row["n"]),
                    nm=_pct(row["median_net_margin"]),
                    at=_num(row["median_asset_turnover"], 3),
                    em=_num(row["median_equity_multiplier"], 3),
                    roe=_pct(row["median_roe"]),
                    prod=_pct(row["product_of_medians"]),
                )
            )
        mfg = industry_df[industry_df["industry"] == "制造"]
        sw = industry_df[industry_df["industry"] == "软件信息"]
        if not mfg.empty and not sw.empty and int(mfg.iloc[0]["n"]) and int(sw.iloc[0]["n"]):
            m, s = mfg.iloc[0], sw.iloc[0]
            drivers = []
            if s["median_net_margin"] < m["median_net_margin"]:
                drivers.append("软件净利率更低")
            else:
                drivers.append("软件净利率不低于制造")
            if s["median_asset_turnover"] < m["median_asset_turnover"]:
                drivers.append("周转更慢")
            else:
                drivers.append("周转不低于制造")
            if s["median_equity_multiplier"] > m["median_equity_multiplier"]:
                drivers.append("杠杆更高")
            else:
                drivers.append("杠杆不高")
            lines += [
                "",
                "制造中位数 ROE {m_roe}，软件 {s_roe}：{drivers}。".format(
                    m_roe=_pct(m["median_roe"]),
                    s_roe=_pct(s["median_roe"]),
                    drivers="，".join(drivers),
                ),
            ]
    if sensitivity is not None and not sensitivity.empty:
        lines += [
            "",
            "## 稳健",
            "",
            "主口径用有效 ROE 原始列。IQR 围栏在全样本 ROE 上算；缩尾用 `*_w`。",
            "",
            "| 口径 | 行业 | 家数 | 净利率 | 周转 | 乘数 | 中位数 ROE |",
            "|------|------|------|--------|------|------|------------|",
        ]
        for treatment in DUPONT_TREATMENTS:
            part = sensitivity[sensitivity["treatment"] == treatment]
            for _, row in part.iterrows():
                if int(row["n"]) == 0:
                    continue
                lines.append(
                    "| {lab} | {ind} | {n} | {nm} | {at} | {em} | {roe} |".format(
                        lab=row["treatment_label"],
                        ind=row["industry"],
                        n=int(row["n"]),
                        nm=_pct(row["median_net_margin"]),
                        at=_num(row["median_asset_turnover"], 3),
                        em=_num(row["median_equity_multiplier"], 3),
                        roe=_pct(row["median_roe"]),
                    )
                )
        q_mfg = sensitivity[(sensitivity["treatment"] == "quality") & (sensitivity["industry"] == GROUP_MANUFACTURING)]
        q_sw = sensitivity[(sensitivity["treatment"] == "quality") & (sensitivity["industry"] == GROUP_SOFTWARE)]
        holds = []
        if not q_mfg.empty and not q_sw.empty:
            for treatment in DUPONT_TREATMENTS:
                m_row = sensitivity[(sensitivity["treatment"] == treatment) & (sensitivity["industry"] == GROUP_MANUFACTURING)]
                s_row = sensitivity[(sensitivity["treatment"] == treatment) & (sensitivity["industry"] == GROUP_SOFTWARE)]
                if m_row.empty or s_row.empty:
                    continue
                if pd.notna(m_row.iloc[0]["median_roe"]) and pd.notna(s_row.iloc[0]["median_roe"]):
                    if m_row.iloc[0]["median_roe"] > s_row.iloc[0]["median_roe"]:
                        holds.append(DUPONT_TREAT_LABEL[treatment])
        if holds:
            lines += ["", "制造 ROE 高于软件的口径：" + "、".join(holds) + "。"]
    lines += [
        "",
        "## 恒等式（质量核对）",
        "",
        f"- 可核对家数：{identity['n']}",
        f"- |乘积 − ROE| 最大：{_num(identity['max_abs_gap'])}",
        f"- 中位差距：{_num(identity['median_abs_gap'])}",
        f"- 差距 ≤ {IDENTITY_ATOL:g} 的家数：{identity['n_match']}",
        "",
        "恒等式只说明科目口径一致，不是分析结果。",
        "",
        "## ROE 与三个因子",
        "",
        "| 因子 | Spearman(与 ROE) | 中位数 |",
        "|------|------------------|--------|",
    ]
    name_zh = {"net_margin": "净利率", "asset_turnover": "总资产周转", "equity_multiplier": "权益乘数"}
    for _, row in assoc.iterrows():
        lines.append(
            f"| {name_zh.get(row['factor'], row['factor'])} | {_num(row['spearman_with_roe'], 3)} | {_num(row['median'], 3)} |"
        )
    if not var_shares.empty:
        lines += [
            "",
            "利润、周转、杠杆均为正的子集上，用 `Var(log 因子) / Var(log ROE)` 看哪一项更散"
            f"（n={int(var_shares.iloc[0]['n'])}，协方差使三项份额不必加总为 1）：",
            "",
            "| 因子 | 对数方差份额 |",
            "|------|--------------|",
        ]
        for _, row in var_shares.iterrows():
            lines.append(f"| {name_zh.get(row['factor'], row['factor'])} | {_pct(row['var_share'])} |")
        if (var_shares["var_share"] > 1).any():
            lines.append("")
            lines.append("份额大于 100% 表示该因子比 ROE 更散：另外两项与它负相关，把 ROE 拉平滑了。")
    if corr is not None and not corr.empty:
        if "roe" in corr.columns:
            top = corr["roe"].drop(labels=["roe"], errors="ignore").abs().sort_values(ascending=False).head(5)
            lines += ["", "## 与 ROE 相关最强的指标（Spearman |ρ|）", ""]
            for name, value in top.items():
                signed = corr.loc[name, "roe"]
                lines.append(f"- `{name}`：{signed:.3f}")
    lines += [
        "",
        "## 附录：主成分（规模 / 杠杆 / 现金）",
        "",
        f"完整个案 {n_pca} 家（8 个指标同时非空，不是按营收样本删公司），用来看截面相关结构，不是 ROE 分解。"
        "规模：log营收、log资产；杠杆：资产负债率、权益乘数、流动比率；现金：应收/收入、存货/收入、OCF/收入（比率用 1%/99% 截尾列）。",
        "",
        "| 成分 | 解释比例 | 累计 | 按载荷归入 |",
        "|------|----------|------|------------|",
    ]
    for _, row in var_df.head(4).iterrows():
        lines.append(
            f"| {row['component']} | {_pct(row['explained_ratio'])} | {_pct(row['cumulative'])} | {row['theme'] or '—'} |"
        )
    if not load_df.empty:
        lines += ["", "前三个主成分上 |载荷| 最大的变量：", ""]
        for pc in ["PC1", "PC2", "PC3"]:
            if pc not in load_df.columns:
                continue
            top3 = load_df.assign(abs_load=load_df[pc].abs()).nlargest(3, "abs_load")
            bits = [f"{r['label']} ({r[pc]:+.2f})" for _, r in top3.iterrows()]
            theme = var_df.loc[var_df["component"] == pc, "theme"]
            theme_txt = theme.iloc[0] if len(theme) else ""
            lines.append(f"- **{pc}（{theme_txt}）**：{', '.join(bits)}")
    lines += [
        "",
        "## 局限",
        "",
        "- PCA 只用完整个案，比营收样本更少；缺净利润或资产负债表的公司不在里面。",
        "- 对数方差分解丢掉亏损和负权益，只描述仍能取对数的子集。",
        "- 主成分是相关结构，不是因果。",
        "- 软件有效 ROE 家数少，中位数对单家敏感；稳健表用来看方向会不会翻。",
        "",
    ]
    return "\n".join(lines)


def _save_fig(path, fig):
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def factor_bar_frame(industry_df):
    """NM / turnover / leverage medians; NaN when n=0, never filled with 0."""
    cols = [
        "industry",
        "n",
        "median_net_margin",
        "median_asset_turnover",
        "median_equity_multiplier",
        "median_roe",
    ]
    have = [c for c in cols if c in industry_df.columns]
    out = industry_df[have].copy()
    for col in have:
        if col != "industry":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    value_cols = [c for c in have if c not in ("industry", "n")]
    if "n" in out.columns and value_cols:
        out.loc[out["n"].fillna(0) <= 0, value_cols] = np.nan
    return out


def _factors_title(frame):
    mfg = frame[frame["industry"] == GROUP_MANUFACTURING]
    sw = frame[frame["industry"] == GROUP_SOFTWARE]
    if mfg.empty or sw.empty:
        return "杜邦三因子行业中位数（有效 ROE 完整个案）"
    mfg_roe = float(mfg.iloc[0]["median_roe"]) if "median_roe" in mfg.columns else np.nan
    sw_roe = float(sw.iloc[0]["median_roe"]) if "median_roe" in sw.columns else np.nan
    mfg_nm = float(mfg.iloc[0]["median_net_margin"]) if "median_net_margin" in mfg.columns else np.nan
    sw_nm = float(sw.iloc[0]["median_net_margin"]) if "median_net_margin" in sw.columns else np.nan
    if np.isfinite(mfg_roe) and np.isfinite(sw_roe) and mfg_roe > sw_roe:
        if np.isfinite(mfg_nm) and np.isfinite(sw_nm) and mfg_nm > sw_nm:
            return "制造 ROE 高于软件，差在净利率不是周转"
        return "制造 ROE 中位数高于软件"
    return "杜邦三因子行业中位数（有效 ROE 完整个案）"


def plot_dupont_factors(industry_df, output_dir):
    if industry_df is None or industry_df.empty:
        return None
    plt = _mpl()
    frame = factor_bar_frame(industry_df)
    industries = frame["industry"].tolist()
    x = np.arange(len(industries))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    nm = pd.to_numeric(frame.get("median_net_margin"), errors="coerce") * 100
    at = pd.to_numeric(frame.get("median_asset_turnover"), errors="coerce")
    em = pd.to_numeric(frame.get("median_equity_multiplier"), errors="coerce")
    ax.bar(x - width, nm.to_numpy(dtype=float), width, label="净利率 (%)", color="#4c78a8")
    ax.bar(x, at.to_numpy(dtype=float), width, label="总资产周转", color="#f58518")
    ax.bar(x + width, em.to_numpy(dtype=float), width, label="权益乘数", color="#54a24b")
    ax.set_xticks(x, industries)
    ax.set_title(_factors_title(frame), fontsize=11, fontweight="bold")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, FACTORS_CHART)
    _save_fig(path, fig)
    return path


def plot_spearman(corr, output_dir):
    if corr is None or corr.empty:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [CORR_LABELS.get(col, col) for col in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    roe_nm = np.nan
    if "roe" in corr.index and "net_margin" in corr.columns:
        roe_nm = float(corr.loc["roe", "net_margin"])
    if np.isfinite(roe_nm):
        ax.set_title(f"ROE 与净利率 Spearman {roe_nm:.2f}（完整个案相关阵）", fontweight="bold")
    else:
        ax.set_title("截面指标 Spearman 相关阵（完整个案）", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(output_dir, SPEARMAN_CHART)
    _save_fig(path, fig)
    return path


def plot_figures(corr, load_df, var_df, output_dir):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if var_df is not None and not var_df.empty:
        axes[0].bar(var_df["component"].head(6), var_df["explained_ratio"].head(6) * 100, color="#4c78a8")
        axes[0].plot(var_df["component"].head(6), var_df["cumulative"].head(6) * 100, color="#f58518", marker="o")
        axes[0].set_ylabel("%")
        pc1 = float(var_df.iloc[0]["explained_ratio"]) * 100 if len(var_df) else np.nan
        axes[0].set_title(f"PC1 解释 {pc1:.0f}% 方差（附录主成分）", fontweight="bold")
    else:
        axes[0].set_visible(False)

    pc_cols = [c for c in ["PC1", "PC2", "PC3"] if load_df is not None and c in load_df.columns]
    if pc_cols and load_df is not None and not load_df.empty:
        mat = load_df[pc_cols].to_numpy()
        vmax = np.nanmax(np.abs(mat)) or 1.0
        im2 = axes[1].imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[1].set_xticks(range(len(pc_cols)), pc_cols)
        axes[1].set_yticks(range(len(load_df)), list(load_df["label"]), fontsize=8)
        axes[1].set_title("PCA 载荷：规模 / 杠杆 / 现金", fontweight="bold")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, CHART_NAME)
    _save_fig(path, fig)
    return path


def load_metrics(metrics_path):
    default_metrics = os.path.join(OUTPUT_DIR_DEFAULT, "company_metrics.csv")
    path = metrics_path or default_metrics
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def main(metrics_path=None, output_dir=None, report_path=None, pdf_dir=None):
    output_dir = output_dir or OUTPUT_DIR_DEFAULT
    os.makedirs(output_dir, exist_ok=True)
    metrics = load_metrics(metrics_path)
    print("=" * 60)
    print("Industry DuPont + PCA appendix")
    print("=" * 60)
    if metrics.empty:
        print("No company_metrics.csv. Run company_metrics.py first.")
        return None

    labeled = attach_industry(metrics, pdf_root=pdf_dir or PDF_DIR_DEFAULT)
    frame = add_size_logs(labeled)
    identity = dupont_identity(frame)
    assoc = dupont_factor_assoc(frame)
    shares = log_variance_shares(frame)
    corr = spearman_matrix(frame, CORR_COLS)
    complete, load_df, var_df = pca_tables(frame)
    industry_df = industry_dupont_table(frame)
    coverage = dupont_coverage(frame)
    sensitivity = industry_dupont_sensitivity(frame)

    pd.DataFrame([identity]).to_csv(os.path.join(output_dir, IDENTITY_NAME), index=False, encoding="utf-8-sig")
    industry_df.to_csv(os.path.join(output_dir, INDUSTRY_DUPONT_NAME), index=False, encoding="utf-8-sig")
    coverage.to_csv(os.path.join(output_dir, COVERAGE_NAME), index=False, encoding="utf-8-sig")
    sensitivity.to_csv(os.path.join(output_dir, SENSITIVITY_NAME), index=False, encoding="utf-8-sig")
    corr.to_csv(os.path.join(output_dir, CORR_NAME), encoding="utf-8-sig")
    load_df.to_csv(os.path.join(output_dir, LOADINGS_NAME), index=False, encoding="utf-8-sig")
    var_df.to_csv(os.path.join(output_dir, VARIANCE_NAME), index=False, encoding="utf-8-sig")

    report = render_report(
        identity, assoc, shares, corr, load_df, var_df,
        n_metrics=len(frame), n_pca=len(complete), industry_df=industry_df,
        coverage=coverage, sensitivity=sensitivity,
    )
    md_path = os.path.join(output_dir, REPORT_NAME)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report)

    chart_path = None
    try:
        factor_path = plot_dupont_factors(industry_df, output_dir)
        spearman_path = plot_spearman(corr, output_dir)
        chart_path = plot_figures(corr, load_df, var_df, output_dir)
        if factor_path:
            print(f"wrote {factor_path}")
        if spearman_path:
            print(f"wrote {spearman_path}")
    except Exception as exc:
        print(f"dupont charts skipped: {exc}")
    print(f"dupont n={identity['n']} max|gap|={identity['max_abs_gap']:.3e} match={identity['n_match']}")
    print(industry_df.to_string(index=False))
    print(f"pca n={len(complete)}")
    if not var_df.empty:
        print(var_df.head(3).to_string(index=False))
    print(f"wrote {md_path}")
    if chart_path:
        print(f"wrote {chart_path}")
    if report_path:
        print(f"wrote {report_path}")
    return var_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industry DuPont decomposition, identity, SVD PCA")
    parser.add_argument("--metrics", default=None, help="company_metrics.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pdf-dir", default=None, help="PDF folder with industry subdirs")
    parser.add_argument("--report", default=None, help="Optional extra markdown path")
    args = parser.parse_args()
    main(
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        report_path=args.report,
        pdf_dir=args.pdf_dir,
    )
