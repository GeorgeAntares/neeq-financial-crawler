import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from industry_groups import GROUP_MANUFACTURING, GROUP_SOFTWARE
from report_charts import (
    STATE_NA,
    STATE_OFF,
    STATE_ON,
    cash_flow_items,
    clean_part_charts,
    dupont_compare,
    flag_matrix,
    flag_table_rows,
    grouped_bar_values,
    peer_bar_frame,
    plot_firm_set,
    select_peers,
    waterfall_steps,
)


FLAG_LABELS = [
    ("flag_cash_debt_high", "存贷双高", "rule-a"),
    ("flag_other_receivables", "其他应收偏高", "rule-b"),
    ("flag_cash_conversion_low", "收现率偏低", "rule-c"),
]


def _row(**kwargs):
    base = {
        "stock_code": "430001",
        "company_name": "测试",
        "year": "2025",
        "industry": GROUP_MANUFACTURING,
        "revenue": 1_000_000.0,
        "cogs": 400_000.0,
        "selling_expense": 10_000.0,
        "admin_expense": 20_000.0,
        "rd_expense": 5_000.0,
        "finance_expense": 3_000.0,
        "operating_profit": 120_000.0,
        "net_profit": 100_000.0,
        "ocf": 40_000.0,
        "icf": -10_000.0,
        "fcf": 15_000.0,
        "gross_margin": 0.60,
        "net_margin": 0.10,
        "roe": 0.12,
        "asset_turnover": 1.05,
        "equity_multiplier": 1.65,
        "cash_conversion": 0.90,
        "revenue_yoy": 0.25,
        "flag_cash_debt_high": False,
        "flag_other_receivables": True,
        "flag_cash_conversion_low": pd.NA,
    }
    base.update(kwargs)
    return pd.Series(base)


class ReportChartsTest(unittest.TestCase):
    def test_waterfall_residual_and_skips_missing_expense(self):
        row = _row()
        del row["rd_expense"]
        steps = waterfall_steps(row)
        labels = [s["label"] for s in steps]
        self.assertEqual(labels[0], "营业收入")
        self.assertIn("毛利", labels)
        self.assertIn("销售费用", labels)
        self.assertIn("管理费用", labels)
        self.assertNotIn("研发费用", labels)
        self.assertIn("营业利润", labels)
        self.assertIn("税及其他", labels)
        self.assertEqual(labels[-1], "净利润")
        residual = next(s for s in steps if s["label"] == "税及其他")
        self.assertAlmostEqual(residual["value"], 100_000.0 - 120_000.0)
        gm = next(s for s in steps if s["label"] == "毛利")
        self.assertAlmostEqual(gm["value"], 600_000.0)

    def test_waterfall_missing_not_zero(self):
        row = _row(cogs=np.nan, selling_expense=np.nan, admin_expense=np.nan, finance_expense=np.nan)
        row = row.drop(labels=["rd_expense"])
        steps = waterfall_steps(row)
        labels = [s["label"] for s in steps]
        self.assertNotIn("营业成本", labels)
        self.assertNotIn("毛利", labels)
        self.assertNotIn("销售费用", labels)
        self.assertIn("其他至营业利润", labels)
        zeros = [s for s in steps if s["kind"] == "delta" and s["value"] == 0]
        self.assertEqual(zeros, [])

    def test_waterfall_empty_without_revenue(self):
        self.assertEqual(waterfall_steps(_row(revenue=np.nan)), [])

    def test_cash_omits_missing_does_not_insert_zero(self):
        items = cash_flow_items(_row(icf=np.nan))
        keys = [it["key"] for it in items]
        self.assertEqual(keys, ["ocf", "fcf"])
        self.assertTrue(all(it["value"] != 0 or it["key"] == "never" for it in items))
        empty = cash_flow_items(_row(ocf=np.nan, icf=np.nan, fcf=np.nan))
        self.assertEqual(empty, [])

    def test_dupont_keeps_industry_when_firm_missing(self):
        medians = pd.DataFrame(
            [
                {"industry": GROUP_MANUFACTURING, "field": "net_margin", "median": 0.04, "n": 10},
                {"industry": GROUP_MANUFACTURING, "field": "asset_turnover", "median": 0.7, "n": 10},
                {"industry": GROUP_MANUFACTURING, "field": "equity_multiplier", "median": 2.0, "n": 10},
            ]
        )
        items = dupont_compare(_row(asset_turnover=np.nan), medians)
        by_field = {it["field"]: it for it in items}
        self.assertTrue(np.isnan(by_field["asset_turnover"]["firm"]))
        self.assertAlmostEqual(by_field["asset_turnover"]["industry"], 0.7)
        self.assertAlmostEqual(by_field["net_margin"]["firm"], 0.10)

    def test_peers_same_industry_named_and_not_filled(self):
        labeled = pd.DataFrame(
            [
                _row(stock_code="430001", company_name="本公司", revenue=1_000_000, gross_margin=0.40).to_dict(),
                _row(stock_code="430002", company_name="近邻甲", revenue=1_100_000, gross_margin=0.30).to_dict(),
                _row(stock_code="430003", company_name="近邻乙", revenue=900_000, gross_margin=0.35).to_dict(),
                _row(stock_code="430004", company_name="缺毛利", revenue=950_000, gross_margin=np.nan).to_dict(),
                _row(
                    stock_code="430010",
                    company_name="软件同行",
                    industry=GROUP_SOFTWARE,
                    revenue=1_000_000,
                    gross_margin=0.55,
                ).to_dict(),
            ]
        )
        row = labeled.iloc[0]
        peers = select_peers(labeled, row, n_peers=7, metric="gross_margin")
        codes = set(peers["stock_code"].astype(str))
        self.assertIn("430002", codes)
        self.assertIn("430003", codes)
        self.assertNotIn("430001", codes)
        self.assertNotIn("430010", codes)
        self.assertNotIn("430004", codes)
        bars = peer_bar_frame(labeled, row, metric="gross_margin", n_peers=7)
        self.assertTrue(bars["is_self"].any())
        self.assertFalse(bars["value"].isna().any())
        self.assertLessEqual(len(bars), 8)

    def test_flag_states_on_off_missing(self):
        rows = flag_table_rows(_row(), FLAG_LABELS)
        by_field = {item["field"]: item["state"] for item in rows}
        self.assertEqual(by_field["flag_cash_debt_high"], STATE_OFF)
        self.assertEqual(by_field["flag_other_receivables"], STATE_ON)
        self.assertEqual(by_field["flag_cash_conversion_low"], STATE_NA)
        matrix, names, labels = flag_matrix(pd.DataFrame([_row()]), FLAG_LABELS)
        self.assertEqual(matrix.shape, (1, 3))
        self.assertEqual(matrix[0, 0], 0.0)
        self.assertEqual(matrix[0, 1], 1.0)
        self.assertTrue(np.isnan(matrix[0, 2]))
        self.assertEqual(len(names), 1)
        self.assertEqual(len(labels), 3)

    def test_grouped_bar_values_keep_nan(self):
        frame = pd.DataFrame({"median_dso": [105.0, np.nan], "median_dio": [113.0, 100.0]})
        values = grouped_bar_values(frame, ["median_dso", "median_dio"])
        self.assertTrue(np.isnan(values["median_dso"][1]))
        self.assertAlmostEqual(values["median_dso"][0], 105.0)

    def test_plot_firm_set_writes_or_skips(self):
        labeled = pd.DataFrame([_row().to_dict(), _row(stock_code="430002", company_name="乙", gross_margin=0.22).to_dict()])
        medians = pd.DataFrame(
            [
                {"industry": GROUP_MANUFACTURING, "field": "net_margin", "median": 0.04, "n": 2},
                {"industry": GROUP_MANUFACTURING, "field": "asset_turnover", "median": 0.7, "n": 2},
                {"industry": GROUP_MANUFACTURING, "field": "equity_multiplier", "median": 2.0, "n": 2},
                {"industry": GROUP_MANUFACTURING, "field": "gross_margin", "median": 0.30, "n": 2},
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = plot_firm_set(labeled.iloc[0], labeled, medians, temp_dir, FLAG_LABELS)
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                self.assertEqual(paths, {})
                return
            self.assertIn("card", paths)
            self.assertTrue(Path(paths["card"]).is_file())
            for key in ("waterfall", "cash", "dupont", "peers", "flags", "kpi"):
                self.assertNotIn(key, paths)
            parts = plot_firm_set(labeled.iloc[0], labeled, medians, temp_dir, FLAG_LABELS, write_parts=True)
            for key in ("card", "waterfall", "cash", "dupont", "peers", "flags", "kpi"):
                self.assertIn(key, parts)
                self.assertTrue(Path(parts[key]).is_file())

    def test_clean_part_charts_keeps_card(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "430001_甲_2025_waterfall.png").write_bytes(b"x")
            (root / "430001_甲_2025_kpi.png").write_bytes(b"x")
            (root / "430001_甲_2025_card.png").write_bytes(b"x")
            (root / "430001_甲_2025.md").write_text("ok", encoding="utf-8")
            n = clean_part_charts(root)
            self.assertEqual(n, 2)
            self.assertTrue((root / "430001_甲_2025_card.png").is_file())
            self.assertTrue((root / "430001_甲_2025.md").is_file())
            self.assertFalse((root / "430001_甲_2025_waterfall.png").is_file())


if __name__ == "__main__":
    unittest.main()
