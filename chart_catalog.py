"""Classify analysis charts by the four reading questions.

Industry medians stay 制造 / 软件信息 / 其他. This module:
  1. unpacks CSRC sectors so 「其他」 is not a junk drawer
  2. copies existing pngs into output/analysis/charts/{四问}/

Default copy is industry figures plus a few example firms. Pass
--all-firms to copy every report-card png.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd

from company_metrics import OUTPUT_DIR_DEFAULT
from industry_groups import (
    GROUP_ORDER,
    GROUP_OTHER,
    PDF_DIR_DEFAULT,
    SECTOR_UNFILED,
    assign_industries,
    normalize_code,
    sector_summary,
)

CHARTS_DIRNAME = "charts"
CATALOG_CSV = "chart_catalog.csv"
SECTORS_CSV = "industry_sectors.csv"
ASSIGN_NAME = "industry_assignments.csv"
INDEX_NAME = "index.md"

DEFAULT_EXAMPLE_CODES = ("839944", "873567", "873762", "873893")

QUESTION_SPECS = (
    ("0", "一页纸", "board"),
    ("1", "怎么赚钱", "profit"),
    ("2", "利润真不真", "earnings"),
    ("3", "会不会被困住", "solvency"),
    ("4", "营运与现金", "operating"),
    ("9", "附录", "appendix"),
)

QUESTION_LABEL = {qid: label for qid, label, _key in QUESTION_SPECS}
QUESTION_FOLDER = {qid: f"{qid}_{label}" for qid, label, _key in QUESTION_SPECS}

KIND_QUESTION = {
    "card": "0",
    "kpi": "1",
    "waterfall": "1",
    "peers": "1",
    "dupont": "1",
    "gm": "1",
    "margins": "1",
    "dupont_factors": "1",
    "dupont_spearman": "1",
    "cash": "2",
    "accruals": "2",
    "cash_gap": "2",
    "flags": "3",
    "flag_grid": "3",
    "dso": "4",
    "dio": "4",
    "wc_cycle": "4",
    "sensitivity": "4",
}

INDUSTRY_STEMS = {
    "industry_board": ("0", "board"),
    "industry_gm": ("1", "gm"),
    "industry_margins": ("1", "margins"),
    "dupont_factors": ("1", "dupont_factors"),
    "dupont_spearman": ("1", "dupont_spearman"),
    "industry_accruals": ("2", "accruals"),
    "industry_cash_gap": ("2", "cash_gap"),
    "company_report_card_flag_grid": ("3", "flag_grid"),
    "company_report_card_flags": ("3", "flags"),
    "industry_dso": ("4", "dso"),
    "industry_dio": ("4", "dio"),
    "industry_wc_cycle": ("4", "wc_cycle"),
    "industry_sensitivity": ("4", "sensitivity"),
}

FIRM_RE = re.compile(
    r"^(?P<code>\d{6})_(?P<name>.+)_(?P<year>\d{4})_(?P<kind>[A-Za-z0-9_]+)\.(?P<ext>png|md)$"
)


def question_folder(qid):
    return QUESTION_FOLDER.get(str(qid), QUESTION_FOLDER["9"])


def classify_kind(kind):
    return KIND_QUESTION.get(str(kind or "").strip().lower(), "9")


def parse_firm_chart(filename):
    match = FIRM_RE.match(os.path.basename(str(filename)))
    if not match:
        return None
    return {
        "stock_code": match.group("code"),
        "company_name": match.group("name"),
        "year": match.group("year"),
        "kind": match.group("kind").lower(),
        "ext": match.group("ext").lower(),
    }


def classify_path(path):
    path = Path(path)
    firm = parse_firm_chart(path.name)
    if firm:
        qid = classify_kind(firm["kind"])
        return {
            "scope": "firm",
            "stock_code": firm["stock_code"],
            "company_name": firm["company_name"],
            "year": firm["year"],
            "kind": firm["kind"],
            "question": qid,
            "question_label": QUESTION_LABEL.get(qid, "附录"),
            "folder": question_folder(qid),
            "name": path.name,
            "source": str(path),
        }
    stem = path.stem
    if stem in INDUSTRY_STEMS:
        qid, kind = INDUSTRY_STEMS[stem]
    else:
        qid, kind = "9", stem
    return {
        "scope": "industry",
        "stock_code": "",
        "company_name": "",
        "year": "",
        "kind": kind,
        "question": qid,
        "question_label": QUESTION_LABEL.get(qid, "附录"),
        "folder": question_folder(qid),
        "name": path.name,
        "source": str(path),
    }


def list_analysis_pngs(output_dir):
    root = Path(output_dir)
    dest = root / CHARTS_DIRNAME
    paths = []
    if not root.is_dir():
        return paths
    for png in sorted(root.glob("*.png")):
        paths.append(png)
    cards = root / "report_cards"
    if cards.is_dir():
        paths.extend(sorted(cards.glob("*.png")))
    return [p for p in paths if dest not in p.parents and p != dest]


def _wanted_codes(firm_codes):
    if firm_codes is None:
        return None
    return {normalize_code(code) for code in firm_codes if str(code).strip()}


def should_copy(record, firm_codes=None, all_firms=False):
    if record.get("scope") != "firm":
        return True
    if all_firms:
        return True
    wanted = _wanted_codes(firm_codes)
    if not wanted:
        wanted = set(DEFAULT_EXAMPLE_CODES)
    return record.get("stock_code") in wanted


def dest_path(charts_root, record):
    bucket = "firms" if record.get("scope") == "firm" else "industry"
    return Path(charts_root) / record["folder"] / bucket / record["name"]


def refresh_assignments(output_dir, pdf_root=None):
    """Rewrite industry_assignments.csv with sector columns. Medians unchanged."""
    output_dir = Path(output_dir)
    assign_path = output_dir / ASSIGN_NAME
    metrics_path = output_dir / "company_metrics.csv"
    if assign_path.is_file():
        frame = pd.read_csv(assign_path)
    elif metrics_path.is_file():
        frame = pd.read_csv(metrics_path)
    else:
        return None, []
    keep = [col for col in ("company_name", "year") if col in frame.columns]
    base = frame[["stock_code"] + keep].copy()
    base["stock_code"] = base["stock_code"].map(normalize_code)
    assigned = pd.DataFrame(assign_industries(base["stock_code"], pdf_root=pdf_root or PDF_DIR_DEFAULT))
    assigned["stock_code"] = assigned["stock_code"].map(normalize_code)
    extra = [col for col in ("industry", "industry_raw", "sector", "sector_label") if col in base.columns]
    if extra:
        base = base.drop(columns=extra)
    merged = base.drop_duplicates("stock_code").merge(assigned, on="stock_code", how="left")
    ordered = [
        col
        for col in (
            "stock_code",
            "company_name",
            "year",
            "industry",
            "industry_raw",
            "sector",
            "sector_label",
        )
        if col in merged.columns
    ]
    extra_cols = [col for col in merged.columns if col not in ordered]
    merged = merged[ordered + extra_cols]
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(assign_path, index=False, encoding="utf-8-sig")
    summary = sector_summary(merged.to_dict(orient="records"))
    sectors_path = output_dir / SECTORS_CSV
    pd.DataFrame(summary).to_csv(sectors_path, index=False, encoding="utf-8-sig")
    return merged, summary


def _pct(share):
    try:
        return f"{float(share) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_index(copied, summary, n_total, dest):
    dest = Path(dest)
    lines = [
        "# 分析图目录",
        "",
        "按四问归类。对照中位数仍是 **制造 / 软件信息 / 其他** 三组；门类只用来拆开「其他」。",
        "原图仍在 `output/analysis/` 与 `report_cards/`，这里是拷贝。",
        "",
        f"样本 {n_total} 家。" if n_total else "样本家数见下表。",
        "",
        "## 门类构成",
        "",
    ]
    if summary:
        other_n = sum(int(row["n"]) for row in summary if row.get("industry") == GROUP_OTHER)
        pending = sum(int(row["n"]) for row in summary if row.get("sector") == "00")
        unfiled = sum(int(row["n"]) for row in summary if row.get("sector") == SECTOR_UNFILED)
        small = other_n - pending - unfiled
        lines.append(
            f"「其他」{other_n} 家 = 待分类 {pending} + 未归档 {unfiled} + 小品类 {small}。"
            "小品类样本不够单独做中位数。"
        )
        lines.append("")
        lines.append(
            _md_table(
                ["门类", "标签", "对照分组", "n", "占样本", "说明"],
                [
                    [
                        row["sector"],
                        row["sector_label"],
                        row["industry"],
                        row["n"],
                        _pct(row.get("share")),
                        row.get("note") or "",
                    ]
                    for row in summary
                ],
            )
        )
        lines.append("")
        by_group = []
        for group in GROUP_ORDER:
            n = sum(int(row["n"]) for row in summary if row.get("industry") == group)
            by_group.append(f"- {group}：{n} 家")
        lines.extend(by_group)
        lines.append("")
    lines.extend(["## 四问", ""])
    by_q = {}
    for rec in copied:
        by_q.setdefault(rec["question"], []).append(rec)
    for qid, label, _key in QUESTION_SPECS:
        folder = question_folder(qid)
        recs = by_q.get(qid, [])
        industry_recs = [r for r in recs if r["scope"] == "industry"]
        firm_recs = [r for r in recs if r["scope"] == "firm"]
        lines.append(f"### {qid}. {label}")
        lines.append("")
        lines.append(f"目录：`{folder}/`")
        lines.append("")
        if industry_recs:
            lines.append("行业截面：")
            for rec in industry_recs:
                rel = f"{folder}/industry/{rec['name']}"
                lines.append(f"- [{rec['name']}]({rel})")
            lines.append("")
        if firm_recs:
            firms = []
            seen = set()
            for rec in firm_recs:
                key = rec["stock_code"]
                if key in seen:
                    continue
                seen.add(key)
                firms.append(f"{rec['stock_code']} {rec['company_name']}")
            lines.append("例卡：" + "、".join(firms))
            lines.append("")
            for rec in firm_recs:
                rel = f"{folder}/firms/{rec['name']}"
                lines.append(f"- [{rec['name']}]({rel})")
            lines.append("")
        if not recs:
            lines.append("（本目录暂无拷贝）")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def plot_sector_mix(summary, path):
    if not summary:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    labels = [f"{row['sector']} {row['sector_label']}" for row in summary]
    values = [int(row["n"]) for row in summary]
    colors = []
    for row in summary:
        if row["industry"] == "制造":
            colors.append("#4c78a8")
        elif row["industry"] == "软件信息":
            colors.append("#54a24b")
        elif row.get("sector") == "00":
            colors.append("#e45756")
        elif row.get("sector") == SECTOR_UNFILED:
            colors.append("#f2cf5b")
        else:
            colors.append("#9e9e9e")
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(labels) + 1.2)))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("家数")
    other_n = sum(int(row["n"]) for row in summary if row.get("industry") == GROUP_OTHER)
    pending = sum(int(row["n"]) for row in summary if row.get("sector") == "00")
    ax.set_title(
        f"「其他」{other_n} 家里 {pending} 家仍待分类；蓝=制造，绿=软件，灰=小品类",
        fontsize=11,
        fontweight="bold",
    )
    for i, n in enumerate(values):
        ax.text(n + 0.4, i, str(n), va="center", fontsize=8, color="#333333")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def organize_charts(
    output_dir,
    pdf_root=None,
    dest_dir=None,
    firm_codes=None,
    all_firms=False,
    refresh=True,
):
    """Copy classified pngs into charts/{四问}/ and write index.md."""
    output_dir = Path(output_dir)
    dest = Path(dest_dir) if dest_dir else output_dir / CHARTS_DIRNAME
    if dest.resolve() == output_dir.resolve():
        raise ValueError("chart dest cannot be the analysis root")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    assigned = None
    summary = []
    if refresh:
        assigned, summary = refresh_assignments(output_dir, pdf_root=pdf_root)
        if assigned is None:
            assigned = pd.DataFrame()
            summary = []

    records = [classify_path(path) for path in list_analysis_pngs(output_dir)]
    copied = []
    for rec in records:
        if not should_copy(rec, firm_codes=firm_codes, all_firms=all_firms):
            continue
        target = dest_path(dest, rec)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec["source"], target)
        rec = dict(rec)
        rec["dest"] = str(target)
        copied.append(rec)

    mix_path = dest / "sector_mix.png"
    try:
        if summary:
            plot_sector_mix(summary, str(mix_path))
    except Exception as exc:
        print(f"sector mix chart skipped: {exc}")

    n_total = int(len(assigned)) if assigned is not None and not getattr(assigned, "empty", True) else 0
    index_text = render_index(copied, summary, n_total, dest)
    if mix_path.is_file():
        index_text = index_text.replace(
            "## 门类构成\n",
            "## 门类构成\n\n![门类构成](sector_mix.png)\n",
            1,
        )
    (dest / INDEX_NAME).write_text(index_text, encoding="utf-8")

    catalog = pd.DataFrame(copied)
    if not catalog.empty:
        catalog.to_csv(dest / CATALOG_CSV, index=False, encoding="utf-8-sig")
        catalog.to_csv(output_dir / CATALOG_CSV, index=False, encoding="utf-8-sig")
    if summary:
        pd.DataFrame(summary).to_csv(dest / SECTORS_CSV, index=False, encoding="utf-8-sig")
    return str(dest)


def main(
    output_dir=None,
    pdf_root=None,
    firm_codes=None,
    all_firms=False,
):
    output_dir = output_dir or OUTPUT_DIR_DEFAULT
    dest = organize_charts(
        output_dir,
        pdf_root=pdf_root,
        firm_codes=firm_codes,
        all_firms=all_firms,
    )
    print(f"wrote chart catalog under {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify analysis charts by four questions")
    parser.add_argument("--output-dir", default=None, help="analysis output folder")
    parser.add_argument("--pdf-dir", default=None, help="PDF folder with CSRC subdirs")
    parser.add_argument(
        "--firms",
        default=None,
        help="Comma-separated stock codes to copy (default: four example cards)",
    )
    parser.add_argument("--all-firms", action="store_true", help="Copy every firm png")
    args = parser.parse_args()
    codes = None
    if args.firms:
        codes = [part.strip() for part in args.firms.split(",") if part.strip()]
    main(
        output_dir=args.output_dir,
        pdf_root=args.pdf_dir,
        firm_codes=codes,
        all_firms=args.all_firms,
    )
