"""
Cross-section charts for firm report cards and industry questions.

The firm deliverable is one tearsheet (`*_card.png`). Panels still map to
the four reading questions. Missing stays missing (never drawn as 0).
Sample is one year plus in-statement prior columns, so there are no
multi-year margin lines.

Ink blue = the firm; stone = peers / industry; rust = deterioration.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from chart_theme import (
    BG,
    DOWN,
    FIRM,
    FLAG,
    INK,
    MUTED,
    NA,
    PEER,
    RULE,
    SURFACE,
    UP,
    draw_empty,
    hide_axes,
    restyle,
    save_fig,
    set_panel,
    use_mpl,
)

COLOR_MAIN = FIRM
COLOR_PEER = PEER
COLOR_WORSE = DOWN
COLOR_BETTER = UP
COLOR_NA = NA
PART_KINDS = ("kpi", "waterfall", "cash", "dupont", "peers", "flags")

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


def _fmt_scaled(value, scale):
    if not np.isfinite(value):
        return ""
    scaled = value / scale
    if abs(scaled) >= 100:
        return f"{scaled:.0f}"
    if abs(scaled) >= 10:
        return f"{scaled:.1f}"
    return f"{scaled:.2f}"


def _short_yoy(value):
    amount = _num(value)
    if not np.isfinite(amount):
        return ""
    sign = "+" if amount >= 0 else ""
    return f"{sign}{amount * 100:.1f}%"


def draw_waterfall(ax, steps):
    if not steps:
        draw_empty(ax, "利润结构缺科目", kicker="1  怎么赚钱")
        return
    scale, unit = _amount_scale([s["value"] for s in steps])
    running = 0.0
    connectors = []
    for i, step in enumerate(steps):
        val = step["value"] / scale
        if step["kind"] == "total":
            bottom = 0.0 if val >= 0 else val
            height = abs(val)
            color = FIRM
            end = val
            running = val
        else:
            if val >= 0:
                bottom = running
                height = val
                color = UP
            else:
                bottom = running + val
                height = -val
                color = DOWN
            end = running + val
            running = end
        ax.bar(i, height, bottom=bottom, color=color, width=0.62, linewidth=0)
        label_y = bottom + height
        ax.text(i, label_y, _fmt_scaled(step["value"], scale), ha="center", va="bottom", fontsize=7, color=MUTED)
        connectors.append(end)
    for i in range(len(connectors) - 1):
        ax.plot([i + 0.31, i + 0.69], [connectors[i], connectors[i]], color=RULE, lw=0.7, zorder=0)
    ax.set_xticks(range(len(steps)), [s["label"] for s in steps], rotation=25, ha="right")
    ax.axhline(0, color=RULE, linewidth=0.8)
    set_panel(ax, ylabel=unit)


def draw_cash_three_way(ax, items):
    if not items:
        draw_empty(ax, "现金流缺科目", kicker="2  利润真不真")
        return
    scale, unit = _amount_scale([it["value"] for it in items])
    labels = [it["label"] for it in items]
    values = [it["value"] / scale for it in items]
    colors = [FIRM if v >= 0 else DOWN for v in values]
    bars = ax.bar(labels, values, color=colors, width=0.48, linewidth=0)
    for bar, raw in zip(bars, items):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, _fmt_scaled(raw["value"], scale), ha="center", va=va, fontsize=8, color=MUTED)
    ax.axhline(0, color=RULE, linewidth=0.8)
    set_panel(ax, ylabel=unit)


def wc_compare_items(row, medians=None):
    industry = _get(row, "industry", "其他")
    items = []
    for field, label in (("dso", "DSO"), ("dio", "DIO"), ("ccc", "现金周期")):
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
                "as_pct": False,
            }
        )
    return items


def wc_compare_title(row, items=None):
    name = firm_label(row)
    items = items if items is not None else wc_compare_items(row)
    dso = next((it["firm"] for it in items if it["field"] == "dso"), np.nan)
    ind = next((it["industry"] for it in items if it["field"] == "dso"), np.nan)
    if np.isfinite(dso) and np.isfinite(ind):
        side = "长于" if dso >= ind else "短于"
        return f"{name}：DSO {side}同业（{dso:.0f} 天 vs {ind:.0f} 天）"
    return f"{name}：营运天数对照"


def draw_dupont(ax, items, empty_message="杜邦因子缺科目", kicker="1  怎么赚钱"):
    """Three small multiples so % and multiples do not share an axis."""
    if not items:
        draw_empty(ax, empty_message, kicker=kicker)
        return
    hide_axes(ax)
    n = len(items)
    for i, item in enumerate(items):
        left = i / n + 0.04 / n
        width = 1 / n - 0.10 / n
        sub = ax.inset_axes([left, 0.0, width, 0.72])
        firm = item["firm"]
        ind = item["industry"]
        if item["as_pct"]:
            firm_v = firm * 100 if np.isfinite(firm) else np.nan
            ind_v = ind * 100 if np.isfinite(ind) else np.nan
            unit = "%"
        else:
            firm_v = firm
            ind_v = ind
            unit = "天" if item.get("field") in ("dso", "dio", "ccc") else "x"
        xs = [0, 1]
        vals = [firm_v, ind_v]
        colors = [FIRM, PEER]
        sub.bar(xs, vals, color=colors, width=0.55, linewidth=0)
        sub.set_xticks(xs, ["本公司", "行业"], fontsize=7)
        sub.axhline(0, color=RULE, linewidth=0.6)
        restyle(sub)
        shown = firm_v if np.isfinite(firm_v) else ind_v
        if np.isfinite(shown):
            if unit == "%":
                text = f"{shown:.1f}%"
            elif unit == "天":
                text = f"{shown:.0f}天"
            else:
                text = f"{shown:.2f}x"
        else:
            text = "缺"
        ax.text(left, 0.92, text, transform=ax.transAxes, fontsize=12, color=INK, fontweight="bold")
        ax.text(left, 0.82, item["label"], transform=ax.transAxes, fontsize=8, color=MUTED)


def draw_peer_bars(ax, frame):
    if frame is None or frame.empty:
        draw_empty(ax, "同业可比缺科目", kicker="1  怎么赚钱")
        return
    names = []
    for _, row in frame.iterrows():
        label = str(row["company_name"] or row["stock_code"]).strip()
        names.append(label)
    values = frame["value"].to_numpy(dtype=float) * 100.0
    colors = [FIRM if bool(flag) else PEER for flag in frame["is_self"]]
    ax.barh(names, values, color=colors, height=0.62, linewidth=0)
    ax.axvline(0, color=RULE, linewidth=0.8)
    set_panel(ax, ylabel=None)
    ax.set_xlabel("%", fontsize=8, color=MUTED)


def draw_flag_chips(ax, rows):
    hide_axes(ax)
    if not rows:
        ax.text(0.5, 0.5, "无红旗规则", ha="center", va="center", color=MUTED, fontsize=10, transform=ax.transAxes)
        return
    state_text = {STATE_ON: "触发", STATE_OFF: "未触发", STATE_NA: "缺科目"}
    n = len(rows)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    for i, item in enumerate(rows):
        r, c = divmod(i, cols)
        x = c / cols + 0.01
        y = 1 - (r + 1) / rows_n + 0.06 / rows_n
        w = 1 / cols - 0.03
        h = 1 / rows_n - 0.12 / rows_n
        if item["state"] == STATE_ON:
            face, tc = FLAG, "#FFFFFF"
        elif item["state"] == STATE_OFF:
            face, tc = SURFACE, INK
        else:
            face, tc = NA, MUTED
        from matplotlib.patches import FancyBboxPatch

        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                facecolor=face,
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(x + 0.018, y + h * 0.58, item["label"], transform=ax.transAxes, fontsize=9, color=tc, ha="left", va="center")
        ax.text(x + 0.018, y + h * 0.28, state_text[item["state"]], transform=ax.transAxes, fontsize=7.5, color=tc, ha="left", va="center", alpha=0.9)


def draw_kpi_strip(ax, items):
    hide_axes(ax)
    if not items:
        return
    from matplotlib.patches import FancyBboxPatch

    n = len(items)
    for i, item in enumerate(items):
        x0 = i / n + 0.006
        w = 1 / n - 0.012
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.06),
                w,
                0.88,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor=SURFACE,
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(x0 + w / 2, 0.78, item["label"], ha="center", va="center", fontsize=8, color=MUTED, transform=ax.transAxes)
        number_color = MUTED if item["display"] == "缺" else INK
        ax.text(x0 + w / 2, 0.46, item["display"], ha="center", va="center", fontsize=13, color=number_color, fontweight="bold", transform=ax.transAxes)
        yoy = item.get("yoy")
        yoy_text = _short_yoy(yoy) if item.get("yoy_display") else ""
        if yoy_text:
            yoy_color = DOWN if _num(yoy) < 0 else UP
            ax.text(x0 + w / 2, 0.2, yoy_text, ha="center", va="center", fontsize=8, color=yoy_color, transform=ax.transAxes)


def plot_waterfall(steps, title, path):
    if not steps:
        return None
    _, plt = use_mpl()
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_waterfall(ax, steps)
    set_panel(ax, title=title, kicker="1  怎么赚钱")
    return save_fig(fig, path)


def plot_cash_three_way(items, title, path):
    if not items:
        return None
    _, plt = use_mpl()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_cash_three_way(ax, items)
    set_panel(ax, title=title, kicker="2  利润真不真")
    return save_fig(fig, path)


def plot_dupont(items, title, path):
    if not items:
        return None
    _, plt = use_mpl()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    draw_dupont(ax, items)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    return save_fig(fig, path)


def plot_peer_bars(frame, title, path):
    if frame is None or frame.empty:
        return None
    _, plt = use_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_peer_bars(ax, frame)
    set_panel(ax, title=title, kicker="1  怎么赚钱")
    return save_fig(fig, path)


def plot_flag_table(rows, title, path):
    if not rows:
        return None
    _, plt = use_mpl()
    fig, ax = plt.subplots(figsize=(9, 2.4 + 0.38 * ((len(rows) + 2) // 3)))
    draw_flag_chips(ax, rows)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    return save_fig(fig, path)


def plot_flag_grid(matrix, names, labels, title, path):
    if matrix is None or matrix.size == 0:
        return None
    _, plt = use_mpl()
    from matplotlib.colors import ListedColormap

    fig_h = max(2.8, 0.42 * len(names) + 1.6)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    cmap = ListedColormap([FIRM, FLAG])
    if hasattr(cmap, "with_extremes"):
        cmap = cmap.with_extremes(bad=NA)
    else:
        cmap.set_bad(NA)
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_yticks(range(len(names)), names)
    restyle(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    return save_fig(fig, path)


def plot_kpi_cards(items, title, path):
    if not items:
        return None
    _, plt = use_mpl()
    fig, ax = plt.subplots(figsize=(11, 2.4))
    draw_kpi_strip(ax, items)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
    return save_fig(fig, path)


def _safe_plot(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _firm_header_bits(row, flag_rows):
    code = str(_get(row, "stock_code", "") or "")
    name = str(_get(row, "company_name", "") or "")
    year = str(_get(row, "year", "") or "")
    industry = str(_get(row, "industry", "") or "")
    sector = str(_get(row, "sector_label", "") or "")
    if sector in ("", industry, "制造业", "软件和信息技术"):
        sector = ""
    n_flags = _num(_get(row, "n_red_flags"))
    triggered = [item["label"] for item in (flag_rows or []) if item["state"] == STATE_ON]
    if np.isfinite(n_flags):
        flag_bit = f"红旗 {int(n_flags)} 项"
    else:
        flag_bit = "红旗未评价"
    bits = [b for b in (year, industry, sector, flag_bit) if b]
    return f"{code}  {name}".strip(), "  |  ".join(bits), triggered


def _strip_firm_prefix(title, row):
    prefix = firm_label(row) + "："
    if title.startswith(prefix):
        return title[len(prefix):]
    return title


def plot_firm_board(row, labeled, medians, flag_labels, path):
    """One-page tearsheet: KPI strip + four-question panels."""
    _, plt = use_mpl()
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    kpis = kpi_items(row)
    steps = waterfall_steps(row)
    cash = cash_flow_items(row)
    factors = dupont_compare(row, medians)
    wc_items = wc_compare_items(row, medians)
    flags = flag_table_rows(row, flag_labels)
    headline, subline, triggered = _firm_header_bits(row, flags)

    fig = plt.figure(figsize=(15.4, 10.6))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(
        4,
        12,
        figure=fig,
        height_ratios=[1.05, 2.55, 2.2, 1.2],
        hspace=0.55,
        wspace=0.55,
        left=0.055,
        right=0.97,
        top=0.88,
        bottom=0.05,
    )
    fig.text(0.055, 0.955, headline, fontsize=18, color=INK, fontweight="bold", ha="left")
    fig.text(0.055, 0.925, subline, fontsize=10, color=MUTED, ha="left")
    if triggered:
        fig.text(0.97, 0.955, "、".join(triggered), fontsize=9, color=FLAG, ha="right")
    fig.add_artist(Line2D([0.055, 0.97], [0.912, 0.912], transform=fig.transFigure, color=RULE, lw=0.8))

    ax_kpi = fig.add_subplot(gs[0, :])
    draw_kpi_strip(ax_kpi, kpis)

    ax_wf = fig.add_subplot(gs[1, 0:8])
    draw_waterfall(ax_wf, steps)
    if steps:
        set_panel(ax_wf, title=_strip_firm_prefix(waterfall_title(row, steps), row), kicker="1  怎么赚钱")

    ax_cash = fig.add_subplot(gs[1, 8:12])
    draw_cash_three_way(ax_cash, cash)
    if cash:
        set_panel(ax_cash, title=_strip_firm_prefix(cash_flow_title(row, cash), row), kicker="2  利润真不真")

    ax_dupont = fig.add_subplot(gs[2, 0:6])
    draw_dupont(ax_dupont, factors)
    if factors:
        ax_dupont.set_title(_strip_firm_prefix(dupont_title(row, factors), row), loc="left", fontsize=10, color=INK, pad=8)
        ax_dupont.text(0.0, 1.16, "1  怎么赚钱", transform=ax_dupont.transAxes, fontsize=7.5, color=MUTED)

    ax_wc = fig.add_subplot(gs[2, 6:12])
    draw_dupont(ax_wc, wc_items, empty_message="营运天数缺科目", kicker="4  营运与现金")
    if wc_items:
        ax_wc.set_title(_strip_firm_prefix(wc_compare_title(row, wc_items), row), loc="left", fontsize=10, color=INK, pad=8)
        ax_wc.text(0.0, 1.16, "4  营运与现金", transform=ax_wc.transAxes, fontsize=7.5, color=MUTED)

    ax_flags = fig.add_subplot(gs[3, :])
    draw_flag_chips(ax_flags, flags)
    ax_flags.text(0.0, 1.18, "3  会不会被困住", transform=ax_flags.transAxes, fontsize=7.5, color=MUTED)
    ax_flags.set_title(_strip_firm_prefix(flag_title(flags, row), row), loc="left", fontsize=10, color=INK, pad=8)
    return save_fig(fig, path, dpi=170)


def _clear_part_files(out_dir, prefix):
    for kind in PART_KINDS:
        path = os.path.join(out_dir, f"{prefix}_{kind}.png")
        if os.path.exists(path):
            os.remove(path)


def clean_part_charts(cards_dir):
    """Remove the old six-file firm set. The deliverable is *_card.png."""
    if not os.path.isdir(cards_dir):
        return 0
    n = 0
    suffixes = tuple(f"_{kind}.png" for kind in PART_KINDS)
    for name in os.listdir(cards_dir):
        lower = name.lower()
        if not lower.endswith(suffixes):
            continue
        os.remove(os.path.join(cards_dir, name))
        n += 1
    return n


def plot_firm_set(row, labeled, medians, out_dir, flag_labels, prefix=None, write_parts=False):
    """Write the firm tearsheet. Optional per-question pngs via write_parts."""
    os.makedirs(out_dir, exist_ok=True)
    code = str(_get(row, "stock_code", "unknown") or "unknown")
    name = str(_get(row, "company_name", "") or "").replace("/", "_").replace("\\", "_")
    year = str(_get(row, "year", "") or "")
    prefix = prefix or f"{code}_{name}_{year}"
    paths = {}
    card_path = os.path.join(out_dir, f"{prefix}_card.png")
    written = _safe_plot(plot_firm_board, row, labeled, medians, flag_labels, card_path)
    if written:
        paths["card"] = written
    if write_parts:
        kpis = kpi_items(row)
        path = os.path.join(out_dir, f"{prefix}_kpi.png")
        part = _safe_plot(plot_kpi_cards, kpis, kpi_title(row), path)
        if part:
            paths["kpi"] = part

        steps = waterfall_steps(row)
        path = os.path.join(out_dir, f"{prefix}_waterfall.png")
        part = _safe_plot(plot_waterfall, steps, waterfall_title(row, steps), path)
        if part:
            paths["waterfall"] = part

        cash = cash_flow_items(row)
        path = os.path.join(out_dir, f"{prefix}_cash.png")
        part = _safe_plot(plot_cash_three_way, cash, cash_flow_title(row, cash), path)
        if part:
            paths["cash"] = part

        factors = dupont_compare(row, medians)
        path = os.path.join(out_dir, f"{prefix}_dupont.png")
        part = _safe_plot(plot_dupont, factors, dupont_title(row, factors), path)
        if part:
            paths["dupont"] = part

        peers = peer_bar_frame(labeled, row)
        path = os.path.join(out_dir, f"{prefix}_peers.png")
        part = _safe_plot(plot_peer_bars, peers, peer_title(peers, row, medians), path)
        if part:
            paths["peers"] = part

        flags = flag_table_rows(row, flag_labels)
        path = os.path.join(out_dir, f"{prefix}_flags.png")
        part = _safe_plot(plot_flag_table, flags, flag_title(flags, row), path)
        if part:
            paths["flags"] = part
    else:
        _clear_part_files(out_dir, prefix)
    return paths


def write_firm_charts(labeled, medians, cards_dir, flag_labels, codes=None, write_parts=False):
    os.makedirs(cards_dir, exist_ok=True)
    if not write_parts:
        clean_part_charts(cards_dir)
    work = labeled.copy()
    work["stock_code"] = work["stock_code"].astype(str)
    if codes is not None:
        wanted = {str(c) for c in codes}
        work = work[work["stock_code"].isin(wanted)]
    n_written = 0
    for _, row in work.iterrows():
        paths = plot_firm_set(row, labeled, medians, cards_dir, flag_labels, write_parts=write_parts)
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
    return plot_flag_grid(matrix, names, labels, "例卡红旗色块（锈红=触发，墨蓝=未触发，浅灰=缺科目）", path)
