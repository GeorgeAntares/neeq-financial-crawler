"""
Predict “profit > 0 but OCF < 0” from BS/IS ratios.

No OCF line items, no net-profit ratios (those leak the two halves of the label).
Random Forest vs L2 logistic regression, stratified 5-fold CV, optional SHAP.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from company_metrics import OUTPUT_DIR_DEFAULT
from industry_groups import PDF_DIR_DEFAULT, assign_industries, normalize_code

N_SPLITS = 5
RANDOM_STATE = 42
RF_TREES = 100
RF_DEPTH = 5

FEATURE_SPEC = [
    ("log_revenue", "log营收"),
    ("gross_margin_w", "毛利率"),
    ("asset_turnover_w", "总资产周转"),
    ("current_ratio_w", "流动比率"),
    ("debt_ratio_w", "资产负债率"),
    ("equity_multiplier_w", "权益乘数"),
    ("ar_to_revenue_w", "应收/收入"),
    ("inventory_to_revenue_w", "存货/收入"),
    ("revenue_yoy_w", "收入同比"),
]
FEATURE_COLS = [row[0] for row in FEATURE_SPEC]
FEATURE_LABELS = {row[0]: row[1] for row in FEATURE_SPEC}

FORBIDDEN_EXACT = {
    "net_profit",
    "net_margin",
    "roe",
    "dupont_product",
    "net_profit_yoy",
    "net_profit_prior",
    "ocf_minus_np",
}
INDUSTRY_DUMMY_PREFIX = "ind_"

REPORT_NAME = "cash_gap_model.md"
CV_NAME = "cash_gap_cv.csv"
COEF_NAME = "cash_gap_logit_coef.csv"
IMP_NAME = "cash_gap_rf_importance.csv"
ROC_CHART = "cash_gap_roc.png"
SHAP_CHART = "cash_gap_shap.png"


def assert_features_safe(columns):
    """Raise if a column would leak profit sign or OCF."""
    bad = []
    for col in columns:
        name = str(col)
        if name in FORBIDDEN_EXACT or "ocf" in name.lower():
            bad.append(name)
        if name.startswith(INDUSTRY_DUMMY_PREFIX):
            continue
    if bad:
        raise ValueError(f"leaky features: {bad}")
    return True


def add_log_revenue(metrics):
    frame = metrics.copy()
    revenue = pd.to_numeric(frame.get("revenue"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_revenue"] = np.where(revenue > 0, np.log10(revenue), np.nan)
    return frame


def make_labeled_frame(metrics):
    """Rows with both net profit and OCF; cash_gap = profit>0 and OCF<0."""
    frame = add_log_revenue(metrics)
    net_profit = pd.to_numeric(frame["net_profit"], errors="coerce")
    ocf = pd.to_numeric(frame["ocf"], errors="coerce")
    keep = net_profit.notna() & ocf.notna()
    labeled = frame.loc[keep].copy()
    labeled["cash_gap"] = ((net_profit.loc[keep] > 0) & (ocf.loc[keep] < 0)).astype(int)
    return labeled


def attach_industry_dummies(labeled, pdf_root=None):
    frame = labeled.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_code)
    assigned = pd.DataFrame(assign_industries(frame["stock_code"], pdf_root=pdf_root or PDF_DIR_DEFAULT))
    assigned = assigned.drop_duplicates(subset=["stock_code"])
    frame = frame.merge(assigned[["stock_code", "industry"]], on="stock_code", how="left")
    dummies = pd.get_dummies(frame["industry"], prefix="ind", dtype=float)
    drop_col = "ind_其他" if "ind_其他" in dummies.columns else None
    if drop_col:
        dummies = dummies.drop(columns=[drop_col])
    for col in dummies.columns:
        frame[col] = dummies[col].to_numpy()
    dummy_cols = list(dummies.columns)
    return frame, dummy_cols


def model_feature_columns(dummy_cols=None):
    cols = list(FEATURE_COLS)
    if dummy_cols:
        cols.extend(dummy_cols)
    assert_features_safe(cols)
    return cols


def _metrics_row(y_true, y_proba, model, fold):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_proba >= 0.5).astype(int)
    out = {
        "model": model,
        "fold": fold,
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return out


def _rf():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=RF_TREES,
                    max_depth=RF_DEPTH,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _logit():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_cv(X, y, pipe_factory, model_name):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    oof = np.full(len(y), np.nan)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        pipe = pipe_factory()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        rows.append(_metrics_row(y[test_idx], proba, model_name, fold))
    return pd.DataFrame(rows), oof


def summarize_cv(fold_df):
    num_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = []
    for model, part in fold_df.groupby("model"):
        row = {"model": model, "n_folds": int(len(part))}
        for col in num_cols:
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_std"] = float(part[col].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fit_logit_coefficients(X, y, feature_names):
    pipe = _logit()
    pipe.fit(X, y)
    coef = pipe.named_steps["model"].coef_.ravel()
    return pd.DataFrame(
        {
            "feature": feature_names,
            "label": [FEATURE_LABELS.get(name, name) for name in feature_names],
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)


def fit_rf_importance(X, y, feature_names):
    pipe = _rf()
    pipe.fit(X, y)
    imp = pipe.named_steps["model"].feature_importances_
    return pipe, pd.DataFrame(
        {
            "feature": feature_names,
            "label": [FEATURE_LABELS.get(name, name) for name in feature_names],
            "importance": imp,
        }
    ).sort_values("importance", ascending=False)


def shap_positive_class(explainer, X_imp):
    import shap

    values = explainer.shap_values(X_imp)
    if isinstance(values, list):
        return values[1]
    if hasattr(values, "ndim") and values.ndim == 3:
        return values[:, :, 1]
    return values


def maybe_shap(rf_pipe, X, feature_names, output_dir):
    try:
        import shap
    except ImportError:
        return None
    X_imp = rf_pipe.named_steps["impute"].transform(X)
    explainer = shap.TreeExplainer(rf_pipe.named_steps["model"])
    shap_pos = shap_positive_class(explainer, X_imp)
    mean_abs = np.abs(shap_pos).mean(axis=0)
    direction = []
    for i, name in enumerate(feature_names):
        if np.nanstd(X_imp[:, i]) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(X_imp[:, i], shap_pos[:, i])[0, 1]
        direction.append(
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "mean_abs_shap": float(mean_abs[i]),
                "corr_value_shap": float(corr) if corr == corr else np.nan,
            }
        )
    table = pd.DataFrame(direction).sort_values("mean_abs_shap", ascending=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(mean_abs)
    ax.barh(
        [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order],
        mean_abs[order],
        color="#4c78a8",
    )
    ax.set_xlabel("平均 |SHAP|")
    ax.set_title("随机森林：谁在推高「利润为正且 OCF 为负」", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, SHAP_CHART)
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    table.attrs["chart"] = path
    return table


def plot_roc_pr(y, oof, output_dir):
    from sklearn.metrics import precision_recall_curve, roc_curve

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, proba, color in oof:
        mask = np.isfinite(proba)
        fpr, tpr, _ = roc_curve(y[mask], proba[mask])
        prec, rec, _ = precision_recall_curve(y[mask], proba[mask])
        axes[0].plot(fpr, tpr, color=color, label=name)
        axes[1].plot(rec, prec, color=color, label=name)
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC（折外预测）", fontweight="bold")
    axes[0].legend(fontsize=8)
    prevalence = float(np.mean(y))
    axes[1].axhline(prevalence, color="#999999", linestyle="--", linewidth=1, label="基线=正例率")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR（折外预测）", fontweight="bold")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, ROC_CHART)
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _pct(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_report(n, n_pos, dummy_cols, fold_df, summary, coef_df, imp_df, shap_df, prevalence):
    lines = [
        "# 附录：现金缺口分类（利润为正且 OCF 为负）",
        "",
        "这是附录，不是封面结论。正例大约一成，分类几乎没有信号；主分析看应计利润和营运资金周期。",
        "",
        "标签：净利润 > 0 **且** 经营现金流净额 < 0。分母是两科目都有数字的公司。",
        "特征只用资产负债和利润表比率（截尾列）+ 行业哑变量；**不用 OCF 分项、OCF/收入、净利率、ROE**，避免把标签两半直接喂给模型。",
        "缺值在每一训练折内用中位数填补。随机森林 100 棵、`max_depth=5`、`class_weight=balanced`，与旧现金流模型同配置。",
        "对照：L2 逻辑回归（标准化后）。评价是分层 5 折的折外 ROC / PR。",
        "",
        f"- 样本：{n} 家，正例 {n_pos}（{_pct(prevalence)}）",
        f"- 行业哑变量：{', '.join(dummy_cols) if dummy_cols else '无'}",
        "",
        "## 交叉验证",
        "",
        "| 模型 | ROC-AUC | PR-AUC | 准确率 | 精确率 | 召回 | F1 |",
        "|------|---------|--------|--------|--------|------|----|",
    ]
    ordered = summary.copy()
    ordered["_ord"] = ordered["model"].map({"random_forest": 0, "logit": 1}).fillna(9)
    ordered = ordered.sort_values("_ord")
    for _, row in ordered.iterrows():
        lines.append(
            "| {model} | {auc} ± {aucs} | {pra} ± {pras} | {acc} | {prec} | {rec} | {f1} |".format(
                model=row["model"],
                auc=_num(row["roc_auc_mean"]),
                aucs=_num(row["roc_auc_std"]),
                pra=_num(row["pr_auc_mean"]),
                pras=_num(row["pr_auc_std"]),
                acc=_num(row["accuracy_mean"]),
                prec=_num(row["precision_mean"]),
                rec=_num(row["recall_mean"]),
                f1=_num(row["f1_mean"]),
            )
        )
    rf_row = summary.loc[summary["model"] == "random_forest"]
    logit_row = summary.loc[summary["model"] == "logit"]
    notes = [f"正例率 {_pct(prevalence)} 是 PR 曲线的无信息基线。正例少，折与折之间方差会大。"]
    if not rf_row.empty and float(rf_row.iloc[0]["recall_mean"]) == 0:
        notes.append("随机森林在 0.5 阈值下折外没有正预测，精确率/召回没有信息，只看 ROC/PR。")
    if not rf_row.empty and float(rf_row.iloc[0]["roc_auc_mean"]) < 0.45:
        notes.append("随机森林折外 ROC 低于 0.5，更像小样本噪声，不是可以反过来用的稳定信号。")
    if not logit_row.empty:
        notes.append(
            "逻辑回归 ROC {auc}，PR-AUC {pra}（基线 {base}）。不把 OCF 和净利率喂进去之后，资产负债/利润表比率几乎分不开这个缺口。".format(
                auc=_num(logit_row.iloc[0]["roc_auc_mean"]),
                pra=_num(logit_row.iloc[0]["pr_auc_mean"]),
                base=_pct(prevalence),
            )
        )
    lines += ["", " ".join(notes)]
    lines += [
        "",
        "## 逻辑回归系数（全样本，标准化后）",
        "",
        "| 特征 | 系数 | 方向 |",
        "|------|------|------|",
    ]
    for _, row in coef_df.head(12).iterrows():
        direction = "更像缺口" if row["coef"] > 0 else "更不像缺口"
        lines.append(f"| {row['label']} | {_num(row['coef'])} | {direction} |")
    lines += [
        "",
        "## 随机森林重要性（全样本）",
        "",
        "| 特征 | 重要性 |",
        "|------|--------|",
    ]
    for _, row in imp_df.iterrows():
        lines.append(f"| {row['label']} | {_num(row['importance'], 4)} |")
    if shap_df is not None and not shap_df.empty:
        lines += [
            "",
            "## SHAP（正类）",
            "",
            "全样本再拟合一棵同样配置的森林，看折外评价够不够之后的方向。",
            "",
            "| 特征 | 平均\\|SHAP\\| | corr(取值, SHAP) |",
            "|------|---------------|-------------------|",
        ]
        for _, row in shap_df.iterrows():
            lines.append(
                f"| {row['label']} | {_num(row['mean_abs_shap'], 4)} | {_num(row['corr_value_shap'])} |"
            )
    lines += [
        "",
        "## 局限",
        "",
        "- 正例大约一成，5 折每折只有两三个正例，AUC 标准差会偏大。",
        "- 不是违约或 ST 标签；只是利润和经营现金的符号组合。",
        "- 单期截面；行业「软件信息」正例可能为 0，哑变量接近完全分离，逻辑回归靠 L2 压住。",
        "",
    ]
    return "\n".join(lines)


def load_metrics(metrics_path):
    path = metrics_path or os.path.join(OUTPUT_DIR_DEFAULT, "company_metrics.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def main(metrics_path=None, pdf_dir=None, output_dir=None, report_path=None):
    output_dir = output_dir or OUTPUT_DIR_DEFAULT
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Cash-gap model / 利润为正且 OCF 为负")
    print("=" * 60)

    metrics = load_metrics(metrics_path)
    if metrics.empty:
        print("No company_metrics.csv. Run company_metrics.py first.")
        return None

    labeled = make_labeled_frame(metrics)
    labeled, dummy_cols = attach_industry_dummies(labeled, pdf_root=pdf_dir)
    feature_cols = model_feature_columns(dummy_cols)
    for col in FEATURE_COLS:
        if col not in labeled.columns:
            labeled[col] = np.nan
    X = labeled[feature_cols]
    y = labeled["cash_gap"].to_numpy(dtype=int)
    n_pos = int(y.sum())
    prevalence = float(y.mean()) if len(y) else np.nan
    print(f"n={len(y)}  cash_gap={n_pos} ({prevalence * 100:.1f}%)")
    print(f"features: {feature_cols}")

    rf_folds, rf_oof = run_cv(X, y, _rf, "random_forest")
    logit_folds, logit_oof = run_cv(X, y, _logit, "logit")
    fold_df = pd.concat([rf_folds, logit_folds], ignore_index=True)
    summary = summarize_cv(fold_df)
    print(summary.to_string(index=False))

    coef_df = fit_logit_coefficients(X, y, feature_cols)
    rf_pipe, imp_df = fit_rf_importance(X, y, feature_cols)
    shap_df = maybe_shap(rf_pipe, X, feature_cols, output_dir)

    fold_df.to_csv(os.path.join(output_dir, CV_NAME), index=False, encoding="utf-8-sig")
    coef_df.to_csv(os.path.join(output_dir, COEF_NAME), index=False, encoding="utf-8-sig")
    imp_df.to_csv(os.path.join(output_dir, IMP_NAME), index=False, encoding="utf-8-sig")
    plot_roc_pr(y, [("random_forest", rf_oof, "#4c78a8"), ("logit", logit_oof, "#f58518")], output_dir)

    report = render_report(len(y), n_pos, dummy_cols, fold_df, summary, coef_df, imp_df, shap_df, prevalence)
    md_path = os.path.join(output_dir, REPORT_NAME)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report)
    print(f"wrote {md_path}")
    print(f"wrote {os.path.join(output_dir, ROC_CHART)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF vs logit for profit>0 and OCF<0")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--pdf-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    main(
        metrics_path=args.metrics,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        report_path=args.report,
    )
