"""Map NEEQ stock codes to three industry groups using local PDF folders.

Peer medians stay 制造 / 软件信息 / 其他. CSRC folder codes are kept as
sector labels so 「其他」 can be split into 待分类 / 未归档 / small sectors.
"""
from __future__ import annotations

import os
from pathlib import Path

GROUP_MANUFACTURING = "制造"
GROUP_SOFTWARE = "软件信息"
GROUP_OTHER = "其他"
GROUP_ORDER = [GROUP_MANUFACTURING, GROUP_SOFTWARE, GROUP_OTHER]

SECTOR_UNFILED = "unfiled"
SECTOR_UNFILED_LABEL = "未归档"
SECTOR_UNKNOWN = "99"
SECTOR_UNKNOWN_LABEL = "其他门类"

SECTOR_LABELS = {
    "00": "待分类",
    "01": "制造业",
    "02": "软件和信息技术",
    "03": "批发零售",
    "04": "科研技术",
    "05": "租赁商务",
    "06": "水利环境",
    "07": "教育",
    "08": "农林牧渔",
    "09": "文化体育",
    "10": "房地产",
    "11": "住宿餐饮",
    "12": "建筑业",
    "13": "交运仓储",
    "14": "电力燃气水",
    "15": "金融业",
    "16": "采矿业",
    "17": "卫生社工",
    "18": "居民服务",
}

PDF_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "pdf")


def normalize_code(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def _group_from_name(name):
    if name.startswith("01_") or "制造业" in name:
        return GROUP_MANUFACTURING
    if name.startswith("02_") or "软件和信息技术" in name:
        return GROUP_SOFTWARE
    return GROUP_OTHER


def _folder_code(name):
    if len(name) >= 2 and name[:2].isdigit():
        return name[:2]
    return None


def _label_from_folder(name, code):
    if code in SECTOR_LABELS:
        return SECTOR_LABELS[code]
    if code and len(name) > 3 and name[2] in "_-":
        rest = name[3:].replace("_", "")
        return rest or SECTOR_UNKNOWN_LABEL
    if name:
        return name.replace("_", "")
    return SECTOR_UNKNOWN_LABEL


def classify_folder(folder_name):
    """Return group plus CSRC sector for a PDF folder name.

    Empty root → 未归档. 00_待分类 stays its own sector inside 其他.
    """
    name = folder_name or ""
    group = _group_from_name(name)
    if not name:
        return {
            "folder": "",
            "group": GROUP_OTHER,
            "sector": SECTOR_UNFILED,
            "sector_label": SECTOR_UNFILED_LABEL,
        }
    code = _folder_code(name)
    if code is None:
        return {
            "folder": name,
            "group": group,
            "sector": SECTOR_UNKNOWN,
            "sector_label": _label_from_folder(name, None),
        }
    return {
        "folder": name,
        "group": group,
        "sector": code,
        "sector_label": _label_from_folder(name, code),
    }


def group_from_folder(folder_name):
    """Collapse CSRC-style folder names into 制造 / 软件信息 / 其他."""
    return classify_folder(folder_name)["group"]


def folder_rank(folder_name):
    """Prefer an explicit CSRC folder over 待分类 or the PDF root."""
    name = folder_name or ""
    group = group_from_folder(name)
    if group != GROUP_OTHER:
        return 40
    if len(name) >= 2 and name[:2].isdigit() and not name.startswith("00"):
        return 30
    if name.startswith("00"):
        return 10
    return 0


def _code_from_pdf_name(filename):
    stem = os.path.basename(filename)
    if not stem.lower().endswith(".pdf"):
        return None
    code = stem.split("_")[0]
    if not code or not any(ch.isdigit() for ch in code):
        return None
    return normalize_code(code)


def scan_pdf_folders(pdf_root):
    """Return stock_code -> folder name, keeping the highest-rank folder."""
    root = Path(pdf_root)
    mapping = {}
    if not root.is_dir():
        return mapping

    def consider(code, folder):
        current = mapping.get(code)
        if current is None or folder_rank(folder) > folder_rank(current):
            mapping[code] = folder

    for pdf in root.glob("*.pdf"):
        code = _code_from_pdf_name(pdf.name)
        if code:
            consider(code, "")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        for pdf in child.glob("*.pdf"):
            code = _code_from_pdf_name(pdf.name)
            if code:
                consider(code, child.name)
    return mapping


def assign_industries(stock_codes, pdf_root=None):
    """List of dicts: stock_code, industry_raw, industry, sector, sector_label."""
    pdf_root = pdf_root or PDF_DIR_DEFAULT
    mapping = scan_pdf_folders(pdf_root)
    rows = []
    for raw in stock_codes:
        code = normalize_code(raw)
        folder = mapping.get(code, "")
        info = classify_folder(folder)
        rows.append(
            {
                "stock_code": code,
                "industry_raw": folder,
                "industry": info["group"],
                "sector": info["sector"],
                "sector_label": info["sector_label"],
            }
        )
    return rows


def sector_summary(rows):
    """Count CSRC sectors; used to unpack 其他 without changing GROUP_ORDER."""
    counts = {}
    for row in rows:
        key = (row.get("sector") or SECTOR_UNKNOWN, row.get("sector_label") or SECTOR_UNKNOWN_LABEL, row.get("industry") or GROUP_OTHER)
        counts[key] = counts.get(key, 0) + 1
    n = len(rows)
    out = []
    for (sector, label, group), k in sorted(counts.items(), key=lambda item: (-item[1], item[0][0])):
        note = ""
        if sector == "00":
            note = "未分到门类，不是一个行业"
        elif sector == SECTOR_UNFILED:
            note = "PDF 在根目录，尚未归档"
        elif group == GROUP_OTHER:
            note = "样本小，中位数并入其他"
        elif group in (GROUP_MANUFACTURING, GROUP_SOFTWARE):
            note = "对照分组"
        out.append(
            {
                "sector": sector,
                "sector_label": label,
                "industry": group,
                "n": k,
                "share": (k / n) if n else 0.0,
                "note": note,
            }
        )
    return out
