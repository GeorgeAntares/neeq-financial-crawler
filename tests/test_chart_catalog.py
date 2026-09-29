import tempfile
import unittest
from pathlib import Path

from chart_catalog import (
    classify_kind,
    classify_path,
    organize_charts,
    parse_firm_chart,
    should_copy,
)
from industry_groups import GROUP_MANUFACTURING, GROUP_OTHER


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n")


class ChartCatalogTest(unittest.TestCase):
    def test_parse_and_classify_firm(self):
        parsed = parse_firm_chart("839944_神州精工_2025_waterfall.png")
        self.assertEqual(parsed["stock_code"], "839944")
        self.assertEqual(parsed["kind"], "waterfall")
        rec = classify_path("output/analysis/report_cards/839944_神州精工_2025_waterfall.png")
        self.assertEqual(rec["scope"], "firm")
        self.assertEqual(rec["question"], "1")
        self.assertEqual(rec["folder"], "1_怎么赚钱")
        board = classify_path("839944_神州精工_2025_card.png")
        self.assertEqual(board["question"], "0")
        self.assertEqual(board["folder"], "0_一页纸")
        cash = classify_path("839944_神州精工_2025_cash.png")
        self.assertEqual(cash["question"], "2")
        flags = classify_path("839944_神州精工_2025_flags.png")
        self.assertEqual(flags["question"], "3")

    def test_classify_industry_stems(self):
        self.assertEqual(classify_path("industry_board.png")["question"], "0")
        self.assertEqual(classify_path("industry_gm.png")["question"], "1")
        self.assertEqual(classify_path("industry_accruals.png")["question"], "2")
        self.assertEqual(classify_path("company_report_card_flag_grid.png")["question"], "3")
        self.assertEqual(classify_path("industry_dso.png")["question"], "4")
        self.assertEqual(classify_path("industry_wc_cycle.png")["kind"], "wc_cycle")
        pca = classify_path("dupont_pca.png")
        self.assertEqual(pca["question"], "9")
        self.assertEqual(pca["scope"], "industry")
        self.assertEqual(classify_kind("unknown_kind"), "9")

    def test_should_copy_defaults_to_examples(self):
        rec = classify_path("839944_神州精工_2025_kpi.png")
        other = classify_path("430001_甲_2025_kpi.png")
        industry = classify_path("industry_gm.png")
        self.assertTrue(should_copy(rec))
        self.assertFalse(should_copy(other))
        self.assertTrue(should_copy(other, all_firms=True))
        self.assertTrue(should_copy(other, firm_codes=["430001"]))
        self.assertTrue(should_copy(industry))

    def test_organize_copies_into_question_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analysis = root / "analysis"
            pdf = root / "pdf"
            cards = analysis / "report_cards"
            _touch(analysis / "industry_gm.png")
            _touch(analysis / "industry_dso.png")
            _touch(analysis / "dupont_pca.png")
            _touch(cards / "839944_神州精工_2025_waterfall.png")
            _touch(cards / "839944_神州精工_2025_cash.png")
            _touch(cards / "430001_甲_2025_waterfall.png")
            (pdf / "01_制造业").mkdir(parents=True)
            (pdf / "01_制造业" / "839944_神州精工_2025_x.pdf").write_bytes(b"%PDF-1.4\n")
            (pdf / "00_待分类").mkdir(parents=True)
            (pdf / "00_待分类" / "430001_甲_2025_x.pdf").write_bytes(b"%PDF-1.4\n")
            (pdf / "03_批发和零售业").mkdir(parents=True)
            (pdf / "03_批发和零售业" / "430002_乙_2025_x.pdf").write_bytes(b"%PDF-1.4\n")
            pd_csv = (
                "stock_code,company_name,year,industry,industry_raw\n"
                "839944,神州精工,2025,制造,01_制造业\n"
                "430001,甲,2025,其他,00_待分类\n"
                "430002,乙,2025,其他,03_批发和零售业\n"
            )
            (analysis / "industry_assignments.csv").write_text(pd_csv, encoding="utf-8-sig")
            dest = Path(
                organize_charts(
                    analysis,
                    pdf_root=pdf,
                    firm_codes=["839944"],
                )
            )
            self.assertTrue((dest / "1_怎么赚钱" / "industry" / "industry_gm.png").is_file())
            self.assertTrue((dest / "4_营运与现金" / "industry" / "industry_dso.png").is_file())
            self.assertTrue((dest / "9_附录" / "industry" / "dupont_pca.png").is_file())
            self.assertTrue((dest / "1_怎么赚钱" / "firms" / "839944_神州精工_2025_waterfall.png").is_file())
            self.assertTrue((dest / "2_利润真不真" / "firms" / "839944_神州精工_2025_cash.png").is_file())
            self.assertFalse((dest / "1_怎么赚钱" / "firms" / "430001_甲_2025_waterfall.png").is_file())
            index = (dest / "index.md").read_text(encoding="utf-8")
            self.assertIn("怎么赚钱", index)
            self.assertIn("待分类", index)
            self.assertIn("批发零售", index)
            assigned = (analysis / "industry_assignments.csv").read_text(encoding="utf-8-sig")
            self.assertIn("sector", assigned)
            self.assertIn("待分类", assigned)
            sectors = (analysis / "industry_sectors.csv").read_text(encoding="utf-8-sig")
            self.assertIn(GROUP_MANUFACTURING, sectors)
            self.assertIn(GROUP_OTHER, sectors)
            self.assertIn("00", sectors)
