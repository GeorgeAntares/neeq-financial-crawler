import unittest
from pathlib import Path


class AnalysisReportTest(unittest.TestCase):
    def test_report_has_outline_sections(self):
        text = Path(__file__).resolve().parents[1].joinpath("ANALYSIS_REPORT.md").read_text(encoding="utf-8")
        for heading in (
            "数据来源与预处理",
            "字段体检",
            "缺失",
            "异常",
            "报表质量",
            "营运资金",
            "盈余质量",
            "行业杜邦",
            "附录",
            "局限",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, text)
        self.assertIn("195", text)
        self.assertIn("company_metrics.py", text)
        self.assertIn("应计", text)
        self.assertIn("完整个案", text)
        self.assertIn("稳健", text)


if __name__ == "__main__":
    unittest.main()
