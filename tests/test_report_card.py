import tempfile
import unittest
from pathlib import Path

import pandas as pd

from industry_groups import GROUP_MANUFACTURING, GROUP_SOFTWARE
from report_card import (
    coverage_table,
    flag_summary,
    industry_medians,
    main as report_card_main,
    render_card,
    render_index,
)


class ReportCardTest(unittest.TestCase):
    def _row(self, **kwargs):
        base = {
            "stock_code": "430001",
            "company_name": "测试",
            "year": "2025",
            "industry": GROUP_MANUFACTURING,
            "revenue": 1_000_000.0,
            "net_profit": 80_000.0,
            "ocf": 60_000.0,
            "gross_margin": 0.40,
            "operating_margin": 0.12,
            "net_margin": 0.08,
            "roe": 0.10,
            "core_profit_ratio": 0.95,
            "sga_to_revenue": 0.18,
            "rd_to_revenue": 0.05,
            "accruals_to_revenue": 0.02,
            "ocf_to_revenue": 0.06,
            "cash_conversion": 0.92,
            "customer_advances_to_revenue": 0.03,
            "current_ratio": 1.5,
            "quick_ratio": 1.1,
            "cash_ratio": 0.4,
            "debt_ratio": 0.45,
            "cash_to_assets": 0.12,
            "st_debt_to_assets": 0.08,
            "other_receivables_to_assets": 0.02,
            "goodwill_to_assets": 0.0,
            "interest_coverage": 6.0,
            "dso": 90.0,
            "dio": 70.0,
            "ccc": 120.0,
            "revenue_yoy": 0.10,
            "net_profit_yoy": 0.20,
            "ocf_yoy": 0.05,
            "bs_articulation_ok": True,
            "equity_negative": False,
            "np_truncated": False,
            "n_red_flags": 0,
            "flag_cash_debt_high": False,
            "flag_other_receivables": False,
            "flag_goodwill": False,
            "flag_core_profit_off": False,
            "flag_cash_conversion_low": False,
            "flag_interest_cover_weak": False,
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_card_has_four_sections_and_keeps_missing(self):
        row = self._row(gross_margin=pd.NA, cash_conversion=pd.NA, n_red_flags=1, flag_other_receivables=True)
        text = render_card(row)
        for heading in ("怎么赚钱", "利润真不真", "会不会被困住", "营运与现金", "红旗"):
            self.assertIn(heading, text)
        self.assertIn("毛利率：缺", text)
        self.assertIn("收现率", text)
        self.assertIn("缺", text)
        self.assertIn("其他应收偏高", text)

    def test_index_and_coverage(self):
        frame = pd.DataFrame([
            self._row(),
            self._row(
                stock_code="430002",
                company_name="软件",
                industry=GROUP_SOFTWARE,
                cash=pd.NA,
                operating_profit=pd.NA,
                flag_cash_conversion_low=True,
                n_red_flags=1,
                cash_conversion=0.5,
            ),
        ])
        frame["cash"] = [200_000.0, pd.NA]
        frame["operating_profit"] = [120_000.0, pd.NA]
        coverage = coverage_table(frame)
        by_field = coverage.set_index("field")
        self.assertEqual(int(by_field.loc["cash", "n"]), 1)
        flags = flag_summary(frame)
        total = flags[(flags["industry"] == "合计") & (flags["flag"] == "flag_cash_conversion_low")].iloc[0]
        self.assertEqual(int(total["n_flag"]), 1)
        medians = industry_medians(frame)
        examples = [("制造", frame.iloc[0])]
        index = render_index(frame, coverage, flags, medians, examples)
        self.assertIn("科目覆盖", index)
        self.assertIn("怎么赚钱", index)
        self.assertIn("缺科目写「缺」", index)
        self.assertIn("一张图一个问题", index)

    def test_main_writes_markdown(self):
        metrics = pd.DataFrame([self._row().to_dict(), self._row(stock_code="430002", company_name="乙").to_dict()])
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "company_metrics.csv"
            metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
            pdf_dir = Path(temp_dir) / "pdf"
            pdf_dir.mkdir()
            extra = Path(temp_dir) / "cards.md"
            out = Path(temp_dir) / "out"
            labeled = report_card_main(
                metrics_path=str(metrics_path),
                pdf_dir=str(pdf_dir),
                output_dir=str(out),
                report_path=str(extra),
                write_firms=True,
            )
            self.assertEqual(len(labeled), 2)
            self.assertTrue((out / "company_report_cards.md").is_file())
            self.assertTrue(extra.is_file())
            cards = list((out / "report_cards").glob("*.md"))
            self.assertEqual(len(cards), 2)
            text = extra.read_text(encoding="utf-8")
            self.assertIn("公司报告卡", text)

    def test_script_has_main_guard(self):
        text = Path(__file__).resolve().parents[1].joinpath("report_card.py").read_text(encoding="utf-8")
        self.assertIn("def main(", text)
        self.assertIn('if __name__ == "__main__":', text)


if __name__ == "__main__":
    unittest.main()
