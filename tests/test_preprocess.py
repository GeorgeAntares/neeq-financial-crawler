import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (
    apply_pipeline,
    constant_feature_columns,
    drop_duplicate_keys,
    field_exam,
    iqr_bounds,
    missing_policy_table,
    outlier_table,
    unify_missing_frame,
    winsorize_iqr,
    zscore_minmax,
)


class PreprocessTest(unittest.TestCase):
    def test_unify_missing_keeps_negative_numbers(self):
        frame = pd.DataFrame({
            "revenue": ["1,000", "--", "未知", "-50"],
            "net_profit": [10.0, np.nan, 3.0, 4.0],
        })
        out = unify_missing_frame(frame, columns=["revenue", "net_profit"])
        self.assertAlmostEqual(out.loc[0, "revenue"], 1000.0)
        self.assertTrue(pd.isna(out.loc[1, "revenue"]))
        self.assertTrue(pd.isna(out.loc[2, "revenue"]))
        self.assertAlmostEqual(out.loc[3, "revenue"], -50.0)

    def test_truncated_profit_policy_is_keep_nan(self):
        frame = pd.DataFrame({
            "stock_code": ["1", "2"],
            "year": ["2025", "2025"],
            "revenue": [1e6, 2e6],
            "net_profit": [1e5, np.nan],
            "np_truncated": [False, True],
            "ocf": [1e4, np.nan],
            "ocf_missing_kind": ["present", "truncated"],
            "inventory": [1.0, np.nan],
            "inventory_status": ["positive", "missing"],
            "cogs": [5e5, np.nan],
            "gross_margin": [0.5, np.nan],
            "roe": [0.1, np.nan],
            "dso": [80.0, 900.0],
        })
        policy = missing_policy_table(frame).set_index("field")
        self.assertEqual(policy.loc["net_profit", "action"], "keep_nan")
        self.assertIn("截断", policy.loc["net_profit", "reason"])
        self.assertEqual(policy.loc["inventory", "action"], "keep_nan_distinct_from_zero")
        self.assertNotIn("mean", " ".join(policy["action"]).lower())

    def test_iqr_flags_and_does_not_drop_revenue(self):
        values = pd.Series([10.0] * 40 + [11.0] * 40 + [1000.0])
        low, high = iqr_bounds(values)
        self.assertLess(high, 1000.0)
        clipped = winsorize_iqr(values)
        self.assertLess(clipped.max(), 1000.0)
        frame = pd.DataFrame({
            "gross_margin": list(values),
            "revenue": list(values * 1e5),
        })
        table = outlier_table(frame, columns=["gross_margin"]).set_index("field")
        self.assertGreater(int(table.loc["gross_margin", "n_iqr_out"]), 0)
        self.assertEqual(table.loc["revenue", "treatment"], "keep_real_extreme")

    def test_dedup_and_keep_id_even_if_constant_year(self):
        frame = pd.DataFrame({
            "stock_code": ["1", "1", "2"],
            "company_name": ["甲", "甲", "乙"],
            "year": ["2025", "2025", "2025"],
            "revenue": [1.0, 9.0, 2.0],
            "const_feat": [0, 0, 0],
        })
        deduped, n_dup = drop_duplicate_keys(frame)
        self.assertEqual(n_dup, 1)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(float(deduped.iloc[0]["revenue"]), 1.0)
        const = constant_feature_columns(deduped)
        self.assertIn("const_feat", const)
        self.assertNotIn("year", const)

    def test_zscore_mean_near_zero(self):
        frame = pd.DataFrame({
            "stock_code": ["1", "2", "3", "4"],
            "company_name": ["a", "b", "c", "d"],
            "year": ["2025"] * 4,
            "gross_margin": [0.1, 0.2, 0.3, 0.4],
            "debt_ratio": [0.2, 0.4, 0.6, 0.8],
        })
        scaled, stats = zscore_minmax(frame, columns=["gross_margin", "debt_ratio"])
        self.assertAlmostEqual(scaled["gross_margin_z"].mean(), 0.0, places=10)
        self.assertAlmostEqual(scaled["gross_margin_z"].std(ddof=0), 1.0, places=10)
        self.assertAlmostEqual(scaled["gross_margin_minmax"].min(), 0.0)
        self.assertAlmostEqual(scaled["gross_margin_minmax"].max(), 1.0)
        self.assertEqual(stats.set_index("field").loc["debt_ratio", "direction"], "cost")

    def test_pipeline_does_not_interpolate_across_firms(self):
        frame = pd.DataFrame({
            "stock_code": ["1", "2", "3"],
            "company_name": ["甲", "乙", "丙"],
            "year": ["2025"] * 3,
            "revenue": [1e6, 2e6, 3e6],
            "net_profit": [1e5, np.nan, 2e5],
            "np_truncated": [False, True, False],
            "ocf": [8e4, np.nan, 9e4],
            "gross_margin": [0.2, 0.25, 0.3],
            "debt_ratio": [0.4, 0.5, 0.6],
            "dso": [80.0, 90.0, 100.0],
            "accruals_to_revenue": [0.02, np.nan, 0.01],
            "current_ratio": [1.2, 1.3, 1.4],
            "ar_to_revenue": [0.2, 0.3, 0.25],
            "ocf_to_revenue": [0.08, np.nan, 0.03],
            "net_margin": [0.1, np.nan, 0.07],
        })
        prep, exam, missing, outliers, scaled, stats, extras = apply_pipeline(frame)
        self.assertTrue(pd.isna(prep.loc[prep["stock_code"] == "2", "net_profit"]).all())
        self.assertEqual(int(exam.set_index("field").loc["net_profit", "n_nonnull"]), 2)
        self.assertEqual(missing.set_index("field").loc["net_profit", "action"], "keep_nan")
        self.assertEqual(extras["n_dup"], 0)

    def test_field_exam_reports_missing_rate(self):
        frame = pd.DataFrame({"revenue": [1.0, 2.0, np.nan], "net_profit": [1.0, np.nan, np.nan]})
        exam = field_exam(frame, columns=["revenue", "net_profit"]).set_index("field")
        self.assertAlmostEqual(exam.loc["revenue", "missing_rate"], 1 / 3)
        self.assertAlmostEqual(exam.loc["net_profit", "missing_rate"], 2 / 3)

    def test_main_guard_present(self):
        text = Path(__file__).resolve().parents[1].joinpath("preprocess.py").read_text(encoding="utf-8")
        self.assertIn("def main(", text)
        self.assertIn('if __name__ == "__main__":', text)


if __name__ == "__main__":
    unittest.main()
