"""
Cross-section charts for firm report cards and industry questions.

One chart, one question. Titles are conclusions. Missing stays missing
(never drawn as 0). Sample is one year plus in-statement prior columns,
so there are no multi-year margin lines.

Blue = the firm / main series; gray = peers / industry; warm = deterioration.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

COLOR_MAIN = "#4c78a8"
COLOR_PEER = "#9e9e9e"
COLOR_WORSE = "#e45756"
COLOR_BETTER = "#54a24b"
COLOR_NA = "#d9d9d9"

STATE_ON = "on"
STATE_OFF = "off"
STATE_NA = "na"

RESIDUAL_TOL = 1.0
N_PEERS_DEFAULT = 7
PEER_METRIC_DEFAULT = "gross_margin"
PEER_METRIC_FALLBACKS = ("net_margin", "roe", "operating_margin")

DUPONT_FACTORS = (
    ("net_margin", "净利率", True),
    ("asset_turnover", "总资产周转", False),
    ("equity_multiplier", "权益乘数", False),
)

CASH_ITEMS = (
    ("ocf", "经营净额"),
    ("icf", "投资净额"),
    ("fcf", "筹资净额"),
)

KPI_SPECS = (
    ("revenue", "营收", "revenue_yoy", "money"),
    ("net_profit", "净利润", "net_profit_yoy", "money"),
    ("ocf", "经营净额", "ocf_yoy", "money"),
    ("gross_margin", "毛利率", None, "pct"),
    ("roe", "ROE", None, "pct"),
    ("cash_conversion", "收现率", None, "pct"),
)

EXPENSE_SPECS = (
    ("selling_expense", "销售费用"),
    ("admin_expense", "管理费用"),
    ("rd_expense", "研发费用"),
    ("finance_expense", "财务费用"),
)


def _get(row, key, default=np.nan):
    if row is None:
        return default
    if hasattr(row, "get"):
        value = row.get(key, default)
        return default if value is None else value
    try:
        return row[key]
    except Exception:
        return default


def _num(value):
    try:
        if value is None or pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(out):
        return np.nan
    return out


def _finite(value):
    return np.isfinite(_num(value))


def _flag_state(value):
    if value is None:
        return STATE_NA
    try:
        if pd.isna(value):
            return STATE_NA
    except (TypeError, ValueError):
        return STATE_NA
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "nan", "none", "na"):
            return STATE_NA
        if text in ("true", "1", "yes"):
            return STATE_ON
        if text in ("false", "0", "no"):
            return STATE_OFF
    if isinstance(value, (bool, np.bool_)):
        return STATE_ON if bool(value) else STATE_OFF
    try:
        return STATE_ON if bool(value) else STATE_OFF
    except (TypeError, ValueError):
        return STATE_NA


def firm_label(row):
    code = str(_get(row, "stock_code", "") or "")
    name = str(_get(row, "company_name", "") or "")
    return f"{code} {name}".strip()


def _money(value):
    amount = _num(value)
    if not np.isfinite(amount):
        return "缺"
    if abs(amount) >= 1e8:
        return f"{amount / 1e8:.2f} 亿元"
    if abs(amount) >= 1e4:
        return f"{amount / 1e4:.0f} 万元"
    return f"{amount:.0f} 元"


def _pct(value):
    amount = _num(value)
    if not np.isfinite(amount):
        return "缺"
    return f"{amount * 100:.1f}%"


def _num_text(value, digits=2):
    amount = _num(value)
    if not np.isfinite(amount):
        return "缺"
    return f"{amount:.{digits}f}"


def _yoy_text(value):
    amount = _num(value)
    if not np.isfinite(amount):
        return ""
    sign = "+" if amount >= 0 else ""
    return f"表内同比 {sign}{amount * 100:.1f}%"


def _median_lookup(medians, industry, field):
    if medians is None or getattr(medians, "empty", True):
        return np.nan, 0
    hit = medians[(medians["industry"] == industry) & (medians["field"] == field)]
    if hit.empty:
        return np.nan, 0
    return _num(hit.iloc[0]["median"]), int(hit.iloc[0]["n"])


def _add_residual(steps, running, target, label):
    gap = target - running
    if abs(gap) <= RESIDUAL_TOL:
        return target
    steps.append({"label": label, "kind": "delta", "value": gap})
    return target


def waterfall_steps(row):
    """P&L bridge. Missing line items are skipped, never filled with 0.

    Totals are checkpoints (revenue, gross profit, operating profit, net profit).
    The step from operating profit to net profit is a residual (tax not extracted).
    """
    revenue = _num(_get(row, "revenue"))
    if not np.isfinite(revenue):
        return []
    steps = [{"label": "营业收入", "kind": "total", "value": revenue}]
    running = revenue

    cogs = _num(_get(row, "cogs"))
    if np.isfinite(cogs):
        steps.append({"label": "营业成本", "kind": "delta", "value": -cogs})
        running = running - cogs
        steps.append({"label": "毛利", "kind": "total", "value": running})

    for field, label in EXPENSE_SPECS:
        amount = _num(_get(row, field))
        if np.isfinite(amount):
            steps.append({"label": label, "kind": "delta", "value": -amount})
            running = running - amount

    op = _num(_get(row, "operating_profit"))
    np_amt = _num(_get(row, "net_profit"))
    if np.isfinite(op):
        running = _add_residual(steps, running, op, "其他至营业利润")
        steps.append({"label": "营业利润", "kind": "total", "value": op})
        running = op
        if np.isfinite(np_amt):
            running = _add_residual(steps, running, np_amt, "税及其他")
            steps.append({"label": "净利润", "kind": "total", "value": np_amt})
    elif np.isfinite(np_amt):
        running = _add_residual(steps, running, np_amt, "其他至净利润")
        steps.append({"label": "净利润", "kind": "total", "value": np_amt})
    return steps


def waterfall_title(row, steps=None):
    steps = steps if steps is not None else waterfall_steps(row)
    name = firm_label(row)
    gm = next((s["value"] for s in steps if s["label"] == "毛利"), np.nan)
    op = _num(_get(row, "operating_profit"))
    np_amt = _num(_get(row, "net_profit"))
    bits = []
    if np.isfinite(gm):
        bits.append(f"毛利 {_money(gm)}")
    if np.isfinite(op):
        bits.append(f"营业利润 {_money(op)}")
    if np.isfinite(np_amt):
        bits.append(f"净利润 {_money(np_amt)}")
    if bits:
        return f"{name}：{'，'.join(bits)}"
    return f"{name}：利润结构（缺科目的阶梯已跳过）"


def cash_flow_items(row):
    """OCF / ICF / FCF. Omit missing; do not insert 0."""
    items = []
    for key, label in CASH_ITEMS:
        value = _num(_get(row, key))
        if np.isfinite(value):
            items.append({"key": key, "label": label, "value": value})
    return items


def cash_flow_title(row, items=None):
    items = items if items is not None else cash_flow_items(row)
    name = firm_label(row)
    if not items:
        return f"{name}：现金流三分类缺科目"
    parts = []
    for item in items:
        if item["value"] > 0:
            parts.append(f"{item['label']}为正")
        elif item["value"] < 0:
            parts.append(f"{item['label']}为负")
        else:
            parts.append(f"{item['label']}为 0")
    return f"{name}：{'、'.join(parts)}"


def dupont_compare(row, medians=None):
    industry = _get(row, "industry", "其他")
    items = []
    for field, label, as_pct in DUPONT_FACTORS:
        firm = _num(_get(row, field))
        med, n_med = _median_lookup(medians, industry, field)
        if not np.isfinite(firm) and not np.isfinite(med):
            continue
        items.append(
            {
                "field": field,
                "label": label,
                "firm": firm,
                "industry": med,
                "n": n_med,
                "as_pct": as_pct,
            }
        )
    return items


def dupont_title(row, items=None):
    name = firm_label(row)
    roe = _num(_get(row, "roe"))
    nm = _num(_get(row, "net_margin"))
    at = _num(_get(row, "asset_turnover"))
    em = _num(_get(row, "equity_multiplier"))
    if np.isfinite(roe):
        return (
            f"{name}：ROE {_pct(roe)} = 净利率 {_pct(nm)} × 周转 {_num_text(at, 3)} × 乘数 {_num_text(em, 2)}"
        )
    return f"{name}：杜邦三因子（缺 ROE 的因子已跳过）"


def kpi_items(row):
    items = []
    for field, label, yoy_field, kind in KPI_SPECS:
        value = _num(_get(row, field))
        yoy = _num(_get(row, yoy_field)) if yoy_field else np.nan
        if kind == "money":
            display = _money(value)
        else:
            display = _pct(value)
        items.append(
            {
                "field": field,
                "label": label,
                "value": value,
                "display": display,
                "yoy": yoy,
                "yoy_display": _yoy_text(yoy) if yoy_field else "",
            }
        )
    return items


def kpi_title(row):
    name = firm_label(row)
    yoy = _num(_get(row, "revenue_yoy"))
    if np.isfinite(yoy):
        sign = "+" if yoy >= 0 else ""
        return f"{name}：当期规模与表内同比（营收 {sign}{yoy * 100:.1f}%）"
    return f"{name}：当期规模（表内同比缺）"


def select_peers(labeled, row, n_peers=N_PEERS_DEFAULT, metric=PEER_METRIC_DEFAULT):
    """Same-industry names with the metric present, closest in log revenue."""
    if labeled is None or labeled.empty:
        return labeled.iloc[0:0].copy() if labeled is not None else pd.DataFrame()
    code = str(_get(row, "stock_code", "") or "")
    industry = _get(row, "industry", "其他")
    work = labeled.copy()
    work["stock_code"] = work["stock_code"].astype(str)
    if "industry" in work.columns:
        peers = work[(work["stock_code"] != code) & (work["industry"] == industry)].copy()
    else:
        peers = work[work["stock_code"] != code].copy()
    if metric not in peers.columns:
        return peers.iloc[0:0].copy()
    values = pd.to_numeric(peers[metric], errors="coerce")
    peers = peers.loc[values.notna()].copy()
    if peers.empty:
        return peers
    self_rev = _num(_get(row, "revenue"))
    peer_rev = pd.to_numeric(peers.get("revenue"), errors="coerce")
    if np.isfinite(self_rev) and self_rev > 0:
        dist = (np.log(peer_rev.clip(lower=1.0)) - np.log(self_rev)).abs()
    else:
        dist = pd.Series(0.0, index=peers.index)
    peers = peers.assign(_peer_dist=dist)
    peers = peers.sort_values(["_peer_dist", "revenue"], ascending=[True, False])
    return peers.head(int(n_peers)).drop(columns=["_peer_dist"], errors="ignore")


def pick_peer_metric(row, labeled):
    for metric in (PEER_METRIC_DEFAULT,) + PEER_METRIC_FALLBACKS:
        if not _finite(_get(row, metric)):
            continue
        peers = select_peers(labeled, row, metric=metric)
        if not peers.empty:
            return metric
    return PEER_METRIC_DEFAULT


def peer_bar_frame(labeled, row, metric=None, n_peers=N_PEERS_DEFAULT):
    metric = metric or pick_peer_metric(row, labeled)
    self_val = _num(_get(row, metric))
    if not np.isfinite(self_val):
        return pd.DataFrame(columns=["stock_code", "company_name", "value", "is_self", "metric"])
    peers = select_peers(labeled, row, n_peers=n_peers, metric=metric)
    records = [
        {
            "stock_code": str(_get(row, "stock_code", "") or ""),
            "company_name": str(_get(row, "company_name", "") or ""),
            "value": self_val,
            "is_self": True,
            "metric": metric,
        }
    ]
    for _, peer in peers.iterrows():
        value = _num(peer.get(metric))
        if not np.isfinite(value):
            continue
        records.append(
            {
                "stock_code": str(peer.get("stock_code", "")),
                "company_name": str(peer.get("company_name", "")),
                "value": value,
                "is_self": False,
                "metric": metric,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.sort_values(["value", "is_self"], ascending=[True, True]).reset_index(drop=True)


PEER_METRIC_LABELS = {
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    "roe": "ROE",
    "operating_margin": "营业利润率",
}


def peer_title(frame, row, medians=None):
    name = firm_label(row)
    if frame is None or frame.empty:
        return f"{name}：同业可比缺科目"
    metric = frame.iloc[0]["metric"]
    label = PEER_METRIC_LABELS.get(metric, metric)
    self_val = _num(frame.loc[frame["is_self"], "value"].iloc[0]) if frame["is_self"].any() else np.nan
    peers = frame.loc[~frame["is_self"], "value"]
    industry = _get(row, "industry", "其他")
    med, n_med = _median_lookup(medians, industry, metric)
    if not np.isfinite(med) and len(peers):
        med = float(peers.median())
        n_med = int(len(peers))
    if np.isfinite(self_val) and np.isfinite(med):
        side = "高于" if self_val >= med else "低于"
        return f"{name}：{label} {side}同业中位（{_pct(self_val)} vs {_pct(med)}，n={n_med}）"
    return f"{name}：{label}同业对照"


def flag_table_rows(row, flag_labels):
    rows = []
    index = getattr(row, "index", None)
    for col, label, rule in flag_labels:
        if index is not None and col not in index:
            state = STATE_NA
        else:
            state = _flag_state(_get(row, col))
        rows.append({"field": col, "label": label, "rule": rule, "state": state})
    return rows


def flag_matrix(frame, flag_labels):
    """n_firms × n_flags array: 1 triggered, 0 off, nan missing."""
    labels = [lab for _, lab, _ in flag_labels]
    fields = [col for col, _, _ in flag_labels]
    names = []
    matrix = []
    if frame is None or len(frame) == 0:
        return np.array([]).reshape(0, len(fields)), [], labels
    work = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    for _, row in work.iterrows():
        names.append(firm_label(row) or str(row.get("stock_code", "")))
        line = []
        for col, _, _ in flag_labels:
            if col not in row.index:
                line.append(np.nan)
                continue
            state = _flag_state(row[col])
            if state == STATE_ON:
                line.append(1.0)
            elif state == STATE_OFF:
                line.append(0.0)
            else:
                line.append(np.nan)
        matrix.append(line)
    return np.asarray(matrix, dtype=float), names, labels


def flag_title(rows, row=None):
    name = firm_label(row) if row is not None else ""
    prefix = f"{name}：" if name else ""
    triggered = [item["label"] for item in rows if item["state"] == STATE_ON]
    missing = [item["label"] for item in rows if item["state"] == STATE_NA]
    if triggered:
        return f"{prefix}红旗触发 { '、'.join(triggered) }"
    if missing and len(missing) == len(rows):
        return f"{prefix}红旗科目均缺，不记 0"
    if missing:
        return f"{prefix}可评价红旗未触发；缺科目保持灰色"
    return f"{prefix}可评价红旗均未触发"


def grouped_bar_values(frame, columns):
    """Keep NaN so callers do not 0-fill missing medians."""
    out = {}
    for col in columns:
        if frame is None or col not in frame.columns:
            out[col] = np.array([], dtype=float)
        else:
            out[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return out


def _use_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib, plt


def _save(fig, plt, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _amount_scale(values):
    finite = [abs(v) for v in values if np.isfinite(v)]
    if not finite:
        return 1.0, "元"
    peak = max(finite)
    if peak >= 1e8:
        return 1e8, "亿元"
    if peak >= 1e4:
        return 1e4, "万元"
    return 1.0, "元"


def plot_waterfall(steps, title, path):
    if not steps:
        return None
    _, plt = _use_mpl()
    scale, unit = _amount_scale([s["value"] for s in steps])
    fig, ax = plt.subplots(figsize=(10, 5))
    running = 0.0
    for i, step in enumerate(steps):
        val = step["value"] / scale
        if step["kind"] == "total":
            bottom = 0.0 if val >= 0 else val
            height = abs(val)
            color = COLOR_MAIN
            running = val
        else:
            if val >= 0:
                bottom = running
                height = val
                color = COLOR_BETTER
            else:
                bottom = running + val
                height = -val
                color = COLOR_WORSE
            running = running + val
        ax.bar(i, height, bottom=bottom, color=color, width=0.62, edgecolor="white")
    ax.set_xticks(range(len(steps)), [s["label"] for s in steps], rotation=22, ha="right")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_ylabel(unit)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, path)


def plot_cash_three_way(items, title, path):
    if not items:
        return None
    _, plt = _use_mpl()
    scale, unit = _amount_scale([it["value"] for it in items])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [it["label"] for it in items]
    values = [it["value"] / scale for it in items]
    colors = [COLOR_MAIN if v >= 0 else COLOR_WORSE for v in values]
    ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_ylabel(unit)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, path)


def plot_dupont(items, title, path):
    if not items:
        return None
    _, plt = _use_mpl()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(items))
    width = 0.36
    firm_vals = []
    ind_vals = []
    tick_labels = []
    for item in items:
        firm = item["firm"]
        ind = item["industry"]
        if item["as_pct"]:
            firm = firm * 100 if np.isfinite(firm) else np.nan
            ind = ind * 100 if np.isfinite(ind) else np.nan
            tick_labels.append(item["label"] + " (%)")
        else:
            tick_labels.append(item["label"])
        firm_vals.append(firm)
        ind_vals.append(ind)
    ax.bar(x - width / 2, firm_vals, width, color=COLOR_MAIN, label="本公司")
    ax.bar(x + width / 2, ind_vals, width, color=COLOR_PEER, label="行业中位")
    ax.set_xticks(x, tick_labels)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, plt, path)


def plot_peer_bars(frame, title, path):
    if frame is None or frame.empty:
        return None
    _, plt = _use_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    names = []
    for _, row in frame.iterrows():
        label = f"{row['stock_code']} {row['company_name']}".strip()
        names.append(label)
    values = frame["value"].to_numpy(dtype=float) * 100.0
    colors = [COLOR_MAIN if bool(flag) else COLOR_PEER for flag in frame["is_self"]]
    ax.barh(names, values, color=colors, edgecolor="white")
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("%")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, path)


def plot_flag_table(rows, title, path):
    if not rows:
        return None
    _, plt = _use_mpl()
    fig, ax = plt.subplots(figsize=(9, 2.8 + 0.28 * len(rows)))
    ax.axis("off")
    cell_text = []
    cell_colors = []
    state_text = {STATE_ON: "触发", STATE_OFF: "未触发", STATE_NA: "缺科目"}
    state_color = {STATE_ON: "#f4c7c3", STATE_OFF: "#cfe2f3", STATE_NA: COLOR_NA}
    for item in rows:
        cell_text.append([item["label"], state_text[item["state"]], item["rule"]])
        fill = state_color[item["state"]]
        cell_colors.append([fill, fill, "#ffffff"])
    table = ax.table(
        cellText=cell_text,
        colLabels=["红旗", "状态", "规则"],
        cellColours=cell_colors,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    return _save(fig, plt, path)


def plot_flag_grid(matrix, names, labels, title, path):
    if matrix is None or matrix.size == 0:
        return None
    _, plt = _use_mpl()
    from matplotlib.colors import ListedColormap

    fig_h = max(2.6, 0.42 * len(names) + 1.6)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    cmap = ListedColormap([COLOR_MAIN, COLOR_WORSE])
    if hasattr(cmap, "with_extremes"):
        cmap = cmap.with_extremes(bad=COLOR_NA)
    else:
        cmap.set_bad(COLOR_NA)
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, path)


def plot_kpi_cards(items, title, path):
    if not items:
        return None
    _, plt = _use_mpl()
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.4))
    for ax, item in zip(axes.ravel(), items):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#dddddd")
        ax.set_title(item["label"], fontsize=10, color="#333333")
        color = COLOR_MAIN if item["display"] != "缺" else COLOR_PEER
        ax.text(0.5, 0.58, item["display"], ha="center", va="center", fontsize=13, fontweight="bold", color=color, transform=ax.transAxes)
        if item["yoy_display"]:
            ax.text(0.5, 0.22, item["yoy_display"], ha="center", va="center", fontsize=8, color=COLOR_PEER, transform=ax.transAxes)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, plt, path)


def _safe_plot(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def plot_firm_set(row, labeled, medians, out_dir, flag_labels, prefix=None):
    """Write the single-firm chart set. Missing matplotlib or missing data skips a file."""
    os.makedirs(out_dir, exist_ok=True)
    code = str(_get(row, "stock_code", "unknown") or "unknown")
    name = str(_get(row, "company_name", "") or "").replace("/", "_").replace("\\", "_")
    year = str(_get(row, "year", "") or "")
    prefix = prefix or f"{code}_{name}_{year}"
    paths = {}

    kpis = kpi_items(row)
    path = os.path.join(out_dir, f"{prefix}_kpi.png")
    written = _safe_plot(plot_kpi_cards, kpis, kpi_title(row), path)
    if written:
        paths["kpi"] = written

    steps = waterfall_steps(row)
    path = os.path.join(out_dir, f"{prefix}_waterfall.png")
    written = _safe_plot(plot_waterfall, steps, waterfall_title(row, steps), path)
    if written:
        paths["waterfall"] = written

    cash = cash_flow_items(row)
    path = os.path.join(out_dir, f"{prefix}_cash.png")
    written = _safe_plot(plot_cash_three_way, cash, cash_flow_title(row, cash), path)
    if written:
        paths["cash"] = written

    factors = dupont_compare(row, medians)
    path = os.path.join(out_dir, f"{prefix}_dupont.png")
    written = _safe_plot(plot_dupont, factors, dupont_title(row, factors), path)
    if written:
        paths["dupont"] = written

    peers = peer_bar_frame(labeled, row)
    path = os.path.join(out_dir, f"{prefix}_peers.png")
    written = _safe_plot(plot_peer_bars, peers, peer_title(peers, row, medians), path)
    if written:
        paths["peers"] = written

    flags = flag_table_rows(row, flag_labels)
    path = os.path.join(out_dir, f"{prefix}_flags.png")
    written = _safe_plot(plot_flag_table, flags, flag_title(flags, row), path)
    if written:
        paths["flags"] = written
    return paths


def write_firm_charts(labeled, medians, cards_dir, flag_labels, codes=None):
    os.makedirs(cards_dir, exist_ok=True)
    work = labeled.copy()
    work["stock_code"] = work["stock_code"].astype(str)
    if codes is not None:
        wanted = {str(c) for c in codes}
        work = work[work["stock_code"].isin(wanted)]
    n_written = 0
    for _, row in work.iterrows():
        paths = plot_firm_set(row, labeled, medians, cards_dir, flag_labels)
        if paths:
            n_written += 1
    return n_written


def plot_example_flag_grid(examples, path, flag_labels):
    if not examples:
        return None
    rows = []
    for title, row in examples:
        series = row.copy() if hasattr(row, "copy") else pd.Series(row)
        if "company_name" in series.index:
            series["company_name"] = f"{title} {series.get('company_name', '')}".strip()
        rows.append(series)
    frame = pd.DataFrame(rows)
    matrix, names, labels = flag_matrix(frame, flag_labels)
    return plot_flag_grid(matrix, names, labels, "例卡红旗色块（红=触发，蓝=未触发，灰=缺科目）", path)
