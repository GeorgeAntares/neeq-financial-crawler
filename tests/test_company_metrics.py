import tempfile
import unittest
from pathlib import Path

import pandas as pd

from company_metrics import (
    DAYS_ANOMALY,
    add_ratios_and_flags,
    build_company_metrics,
    quality_summary,
    score_cash,
    score_equity,
    score_interest_expense,
    score_net_profit,
    score_operating_profit,
    score_other_receivables,
    score_revenue,
    score_sales_cash,
    winsorize_series,
    yoy,
)


def write_csv(folder, name, rows, columns):
    path = Path(folder) / name
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    return path


class CompanyMetricsTest(unittest.TestCase):
    def test_uses_first_block_not_parent_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_csv(
                temp_dir,
                "430001_测试_2025_合并利润表.csv",
                [
                    ["项目", "", ""],
                    ["一、营业总收入", 1_000_000, 800_000],
                    ["其中：营业收入", 1_000_000, 800_000],
                    ["其中：营业成本", 400_000, 350_000],
                    ["二、营业总成本", 700_000, 600_000],
                    ["销售费用", 10_000, 8_000],
                    ["管理费用", 20_000, 18_000],
                    ["研发费用", 5_000, 4_000],
                    ["财务费用", 3_000, 2_500],
                    ["其中：利息费用", 2_500, 2_000],
                    ["三、营业利润（亏损以“-”号填列）", 120_000, 60_000],
                    ["加：营业外收入", 5_000, 1_000],
                    ["五、净利润（净亏损以“－”号填列）", 100_000, 50_000],
                    ["1.持续经营净利润（净亏损以“-”号填列）", 100_000, 50_000],
                    ["2.归属于母公司所有者的净利润", 80_000, 40_000],
                    ["项目", "", ""],
                    ["一、营业收入", 10_000, 9_000],
                    ["减：营业成本", 1_000, 900],
                    ["三、营业利润（亏损以“-”号填列）", 1, 1],
                    ["四、净利润（净亏损以“-”号填列）", 1, 1],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430001_测试_2025_合并资产负债表.csv",
                [
                    ["项目", "", ""],
                    ["货币资金", 200_000, 180_000],
                    ["流动资产合计", 500_000, 450_000],
                    ["应收账款", 120_000, 100_000],
                    ["预付款项", 10_000, 8_000],
                    ["其他应收款", 5_000, 4_000],
                    ["存货", 80_000, 70_000],
                    ["商誉", "", ""],
                    ["资产总计", 1_000_000, 900_000],
                    ["短期借款", 80_000, 70_000],
                    ["流动负债合计", 250_000, 200_000],
                    ["应付账款", 50_000, 40_000],
                    ["预收款项", 1_000, 800],
                    ["合同负债", 4_000, 3_000],
                    ["一年内到期的非流动负债", 20_000, 15_000],
                    ["负债合计", 400_000, 350_000],
                    ["所有者权益（或股东权益）合计", 600_000, 550_000],
                    ["负债和所有者权益（或股东权益）总计", 1_000_000, 900_000],
                    ["项目", "", ""],
                    ["货币资金", 1, 1],
                    ["资产总计", 50_000, 40_000],
                    ["所有者权益合计", 10_000, 8_000],
                ],
                ["项目", "期末余额", "期初余额"],
            )
            write_csv(
                temp_dir,
                "430001_测试_2025_合并现金流量表.csv",
                [
                    ["项目", "", ""],
                    ["销售商品、提供劳务收到的现金", 900_000, 700_000],
                    ["经营活动产生的现金流量净额", 40_000, 20_000],
                    ["投资活动产生的现金流量净额", -10_000, -5_000],
                    ["项目", "", ""],
                    ["销售商品、提供劳务收到的现金", 1, 1],
                    ["经营活动产生的现金流量净额", 999, 1],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            metrics = build_company_metrics(temp_dir)

        self.assertEqual(len(metrics), 1)
        row = metrics.iloc[0]
        self.assertEqual(row["revenue"], 1_000_000)
        self.assertEqual(row["cogs"], 400_000)
        self.assertEqual(row["cost_source"], "营业成本")
        self.assertEqual(row["net_profit"], 100_000)
        self.assertEqual(row["ocf"], 40_000)
        self.assertEqual(row["equity"], 600_000)
        self.assertEqual(row["equity_source"], "total")
        self.assertAlmostEqual(row["gross_margin"], 0.6)
        self.assertAlmostEqual(row["net_margin"], 0.1)
        self.assertAlmostEqual(row["avg_assets"], 950_000)
        self.assertAlmostEqual(row["avg_equity"], 575_000)
        self.assertAlmostEqual(row["asset_turnover"], 1_000_000 / 950_000)
        self.assertAlmostEqual(row["equity_multiplier"], 950_000 / 575_000)
        self.assertAlmostEqual(row["roe"], 100_000 / 575_000)
        self.assertAlmostEqual(row["dupont_product"], row["roe"], places=10)
        self.assertAlmostEqual(row["current_ratio"], 2.0)
        self.assertAlmostEqual(row["debt_ratio"], 0.4)
        self.assertAlmostEqual(row["ar_to_revenue"], 0.12)
        self.assertAlmostEqual(row["inventory_to_revenue"], 0.08)
        self.assertAlmostEqual(row["ocf_to_revenue"], 0.04)
        self.assertEqual(row["ocf_minus_np"], -60_000)
        self.assertTrue(row["gm_valid"])
        self.assertFalse(row["equity_negative"])
        self.assertTrue(row["roe_valid"])
        self.assertTrue(row["bs_articulation_ok"])
        self.assertAlmostEqual(row["bs_rel_gap"], 0.0)
        self.assertFalse(row["np_truncated"])
        self.assertEqual(row["ocf_missing_kind"], "present")
        self.assertEqual(row["inventory_status"], "positive")
        self.assertAlmostEqual(row["dso"], 120_000 / 1_000_000 * 365)
        self.assertAlmostEqual(row["dio"], 80_000 / 400_000 * 365)
        self.assertAlmostEqual(row["dpo"], 50_000 / 400_000 * 365)
        self.assertAlmostEqual(row["accruals_to_revenue"], 0.06)
        self.assertFalse(row["dso_anomalous"])
        self.assertAlmostEqual(row["revenue_yoy"], 0.25)
        self.assertAlmostEqual(row["net_profit_yoy"], 1.0)
        self.assertAlmostEqual(row["ocf_yoy"], 1.0)
        self.assertEqual(row["cash"], 200_000)
        self.assertEqual(row["operating_profit"], 120_000)
        self.assertEqual(row["sales_cash"], 900_000)
        self.assertEqual(row["interest_expense"], 2_500)
        self.assertEqual(row["st_interest_bearing"], 100_000)
        self.assertEqual(row["customer_advances"], 5_000)
        self.assertTrue(pd.isna(row["goodwill"]))
        self.assertEqual(row["goodwill_status"], "missing")
        self.assertAlmostEqual(row["core_profit_ratio"], 120_000 / 125_000)
        self.assertAlmostEqual(row["cash_conversion"], 0.9)
        self.assertAlmostEqual(row["sga_to_revenue"], 0.03)
        self.assertAlmostEqual(row["interest_coverage"], 120_000 / 2_500)
        self.assertAlmostEqual(row["quick_ratio"], (500_000 - 80_000) / 250_000)
        self.assertFalse(bool(row["flag_cash_debt_high"]))
        self.assertFalse(bool(row["flag_cash_conversion_low"]))

    def test_drops_tiny_revenue_and_does_not_use_total_cost_for_gm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_csv(
                temp_dir,
                "430002_小额_2025_合并利润表.csv",
                [["营业收入", 17.4, 10.0], ["营业总成本", 8.0, 7.0]],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430003_总成本_2025_合并利润表.csv",
                [
                    ["一、营业总收入", 500_000, 400_000],
                    ["二、营业总成本", 200_000, 180_000],
                    ["五、净利润（净亏损以“－”号填列）", 50_000, 40_000],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            metrics = build_company_metrics(temp_dir)

        self.assertEqual(list(metrics["stock_code"]), ["430003"])
        self.assertTrue(pd.isna(metrics.iloc[0]["cogs"]))
        self.assertEqual(metrics.iloc[0]["total_operating_cost"], 200_000)
        self.assertEqual(metrics.iloc[0]["cost_source"], "营业总成本")
        self.assertFalse(bool(metrics.iloc[0]["gm_valid"]))
        self.assertTrue(pd.isna(metrics.iloc[0]["gross_margin"]))
        self.assertTrue(pd.isna(metrics.iloc[0]["dio"]))

    def test_yoy_uses_abs_prior(self):
        series = yoy(pd.Series([10.0, -20.0]), pd.Series([-10.0, -10.0]))
        self.assertAlmostEqual(series.iloc[0], 2.0)
        self.assertAlmostEqual(series.iloc[1], -1.0)

    def test_winsorize_clips_tails(self):
        values = pd.Series([0.2] * 40 + [0.21] * 40 + [50.0, -40.0])
        clipped = winsorize_series(values)
        self.assertLess(clipped.max(), 50.0)
        self.assertGreater(clipped.min(), -40.0)

    def test_score_helpers_reject_footnotes(self):
        self.assertEqual(score_revenue("其中：营业收入"), 0)
        self.assertGreater(score_revenue("一、营业总收入"), score_revenue("一、营业收入"))
        self.assertEqual(score_net_profit("1.持续经营净利润（净亏损以“-”号填列）"), 0)
        self.assertEqual(score_net_profit("2.归属于母公司所有者的净利润"), 0)
        self.assertGreater(score_net_profit("五、净利润（净亏损以“－”号填列）"), 0)
        self.assertEqual(score_equity("负债和所有者权益（或股东权益）总计"), 0)
        self.assertEqual(score_equity("所有者权益（或股东权益）："), 0)
        self.assertGreater(score_equity("所有者权益（或股东权益）合计"), score_equity("归属于母公司所有者权益合计"))
        self.assertEqual(score_cash("货币资金"), 1)
        self.assertEqual(score_cash("其中：货币资金"), 0)
        self.assertEqual(score_other_receivables("其他应收款"), 1)
        self.assertEqual(score_other_receivables("其中：应收利息"), 0)
        self.assertGreater(score_operating_profit("三、营业利润（亏损以“-”号填列）"), 0)
        self.assertEqual(score_operating_profit("营业利润率"), 0)
        self.assertGreater(score_interest_expense("其中：利息费用"), score_interest_expense("利息费用"))
        self.assertEqual(score_interest_expense("利息收入"), 0)
        self.assertGreater(score_sales_cash("销售商品、提供劳务收到的现金"), 0)

    def test_quality_flags_equity_inventory_dso_articulation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_csv(
                temp_dir,
                "430010_负权益_2025_合并利润表.csv",
                [
                    ["一、营业总收入", 500_000, 400_000],
                    ["其中：营业成本", 200_000, 180_000],
                    ["五、净利润（净亏损以“－”号填列）", 20_000, 10_000],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430010_负权益_2025_合并资产负债表.csv",
                [
                    ["资产总计", 100_000, 120_000],
                    ["负债合计", 180_000, 150_000],
                    ["所有者权益（或股东权益）合计", -80_000, -30_000],
                    ["应收账款", 10_000, 8_000],
                    ["存货", 0, 0],
                ],
                ["项目", "期末余额", "期初余额"],
            )
            write_csv(
                temp_dir,
                "430011_长账期_2025_合并利润表.csv",
                [
                    ["一、营业总收入", 500_000, 400_000],
                    ["其中：营业成本", 300_000, 280_000],
                    ["五、净利润（净亏损以“－”号填列）", 10_000, 8_000],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430011_长账期_2025_合并资产负债表.csv",
                [
                    ["资产总计", 2_000_000, 1_800_000],
                    ["负债合计", 800_000, 700_000],
                    ["所有者权益（或股东权益）合计", 1_200_000, 1_100_000],
                    ["应收账款", 1_200_000, 1_000_000],
                ],
                ["项目", "期末余额", "期初余额"],
            )
            write_csv(
                temp_dir,
                "430011_长账期_2025_合并现金流量表.csv",
                [["经营活动产生的现金流量净额", -5_000, 2_000]],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430012_勾稽_2025_合并利润表.csv",
                [["一、营业总收入", 500_000, 400_000], ["其中：营业成本", 200_000, 180_000]],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430012_勾稽_2025_合并资产负债表.csv",
                [
                    ["资产总计", 1_000_000, 900_000],
                    ["负债合计", 100_000, 90_000],
                    ["所有者权益（或股东权益）合计", 100_000, 90_000],
                ],
                ["项目", "期末余额", "期初余额"],
            )
            metrics = build_company_metrics(temp_dir)

        by_code = metrics.set_index("stock_code")
        neg = by_code.loc["430010"]
        self.assertTrue(bool(neg["equity_negative"]))
        self.assertFalse(bool(neg["roe_valid"]))
        self.assertTrue(pd.isna(neg["roe"]))
        self.assertTrue(pd.isna(neg["equity_multiplier"]))
        self.assertEqual(neg["inventory_status"], "zero")
        self.assertAlmostEqual(neg["dio"], 0.0)
        self.assertFalse(bool(neg["np_truncated"]))
        self.assertEqual(neg["ocf_missing_kind"], "no_statement")

        long_ar = by_code.loc["430011"]
        self.assertGreater(long_ar["dso"], DAYS_ANOMALY)
        self.assertTrue(bool(long_ar["dso_anomalous"]))
        self.assertEqual(long_ar["inventory_status"], "missing")
        self.assertTrue(pd.isna(long_ar["dio"]))
        self.assertAlmostEqual(long_ar["accruals_to_revenue"], (10_000 - (-5_000)) / 500_000)

        broken = by_code.loc["430012"]
        self.assertFalse(bool(broken["bs_articulation_ok"]))
        self.assertGreater(broken["bs_rel_gap"], 0.5)
        self.assertTrue(bool(broken["np_truncated"]))

        summary = quality_summary(metrics).set_index("flag")
        self.assertEqual(int(summary.loc["equity_negative", "n"]), 1)
        self.assertEqual(int(summary.loc["dso_anomalous", "n"]), 1)
        self.assertEqual(int(summary.loc["inventory_zero", "n"]), 1)
        self.assertEqual(int(summary.loc["inventory_missing", "n"]), 2)
        self.assertEqual(int(summary.loc["bs_articulation_fail", "n"]), 1)

    def test_add_ratios_and_flags_on_wide_frame(self):
        frame = pd.DataFrame({
            "stock_code": ["1"],
            "year": ["2025"],
            "revenue": [1_000_000.0],
            "cogs": [400_000.0],
            "total_operating_cost": [700_000.0],
            "net_profit": [100_000.0],
            "total_assets": [1_000_000.0],
            "total_assets_begin": [1_000_000.0],
            "total_liabilities": [400_000.0],
            "equity": [600_000.0],
            "equity_begin": [600_000.0],
            "equity_item": ["所有者权益（或股东权益）合计"],
            "current_assets": [500_000.0],
            "current_liabilities": [250_000.0],
            "accounts_receivable": [100_000.0],
            "inventory": [50_000.0],
            "accounts_payable": [40_000.0],
            "ocf": [80_000.0],
            "cashflow_n_rows": [3],
        })
        out = add_ratios_and_flags(frame).iloc[0]
        self.assertTrue(out["gm_valid"])
        self.assertTrue(out["bs_articulation_ok"])
        self.assertAlmostEqual(out["ccc"], out["dso"] + out["dio"] - out["dpo"])
        self.assertAlmostEqual(out["accruals_to_revenue"], 0.02)

    def test_red_flags_keep_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_csv(
                temp_dir,
                "430020_红旗_2025_合并利润表.csv",
                [
                    ["一、营业总收入", 1_000_000, 800_000],
                    ["其中：营业成本", 400_000, 350_000],
                    ["销售费用", 50_000, 40_000],
                    ["管理费用", 40_000, 30_000],
                    ["三、营业利润（亏损以“-”号填列）", 10_000, 8_000],
                    ["加：营业外收入", 20_000, 5_000],
                    ["其中：利息费用", 8_000, 7_000],
                    ["五、净利润（净亏损以“－”号填列）", 8_000, 6_000],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430020_红旗_2025_合并资产负债表.csv",
                [
                    ["货币资金", 400_000, 350_000],
                    ["应收账款", 50_000, 40_000],
                    ["其他应收款", 150_000, 120_000],
                    ["存货", 20_000, 18_000],
                    ["商誉", 200_000, 200_000],
                    ["资产总计", 1_000_000, 900_000],
                    ["短期借款", 250_000, 200_000],
                    ["一年内到期的非流动负债", 50_000, 40_000],
                    ["流动负债合计", 400_000, 350_000],
                    ["负债合计", 400_000, 350_000],
                    ["所有者权益（或股东权益）合计", 600_000, 550_000],
                ],
                ["项目", "期末余额", "期初余额"],
            )
            write_csv(
                temp_dir,
                "430020_红旗_2025_合并现金流量表.csv",
                [
                    ["销售商品、提供劳务收到的现金", 500_000, 400_000],
                    ["经营活动产生的现金流量净额", 9_000, 7_000],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430021_缺存货_2025_合并利润表.csv",
                [
                    ["一、营业总收入", 500_000, 400_000],
                    ["其中：营业成本", 200_000, 180_000],
                    ["五、净利润（净亏损以“－”号填列）", 20_000, 10_000],
                ],
                ["项目", "本期金额", "上期金额"],
            )
            write_csv(
                temp_dir,
                "430021_缺存货_2025_合并资产负债表.csv",
                [
                    ["货币资金", 80_000, 70_000],
                    ["流动资产合计", 200_000, 180_000],
                    ["资产总计", 400_000, 380_000],
                    ["流动负债合计", 100_000, 90_000],
                    ["负债合计", 150_000, 140_000],
                    ["所有者权益（或股东权益）合计", 250_000, 240_000],
                ],
                ["项目", "期末余额", "期初余额"],
            )
            metrics = build_company_metrics(temp_dir)

        by_code = metrics.set_index("stock_code")
        flagged = by_code.loc["430020"]
        self.assertTrue(bool(flagged["flag_cash_debt_high"]))
        self.assertTrue(bool(flagged["flag_other_receivables"]))
        self.assertTrue(bool(flagged["flag_goodwill"]))
        self.assertTrue(bool(flagged["flag_core_profit_off"]))
        self.assertTrue(bool(flagged["flag_cash_conversion_low"]))
        self.assertTrue(bool(flagged["flag_interest_cover_weak"]))
        self.assertGreaterEqual(int(flagged["n_red_flags"]), 6)
        self.assertAlmostEqual(flagged["core_profit_ratio"], 10_000 / 30_000)
        self.assertAlmostEqual(flagged["cash_conversion"], 0.5)
        self.assertAlmostEqual(flagged["interest_coverage"], 10_000 / 8_000)

        missing_inv = by_code.loc["430021"]
        self.assertEqual(missing_inv["inventory_status"], "missing")
        self.assertTrue(pd.isna(missing_inv["quick_ratio"]))
        self.assertTrue(pd.isna(missing_inv["st_interest_bearing"]))
        self.assertTrue(pd.isna(missing_inv["goodwill"]))
        self.assertTrue(pd.isna(missing_inv["flag_cash_debt_high"]))
        self.assertTrue(pd.isna(missing_inv["interest_coverage"]))

    def test_main_guard_present(self):
        text = Path(__file__).resolve().parents[1].joinpath("company_metrics.py").read_text(encoding="utf-8")
        self.assertIn("def main(", text)
        self.assertIn('if __name__ == "__main__":', text)


if __name__ == "__main__":
    unittest.main()
