import unittest

import pandas as pd

from financial_analysis import (
    MIN_REVENUE_CNY,
    clip_gross_margin,
    filter_plausible_revenue,
    is_missing_token,
    pick_operating_cost,
    pick_primary_row,
    to_numeric_safe,
)


class FinancialAnalysisCleanTest(unittest.TestCase):
    def test_drops_footnote_sized_revenue(self):
        df = pd.DataFrame({
            'stock_code': ['a', 'b'],
            'company_name': ['甲', '乙'],
            'revenue': [1.0, 5_000_000],
        })
        kept, dropped = filter_plausible_revenue(df)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept.iloc[0]['stock_code'], 'b')
        self.assertGreaterEqual(kept.iloc[0]['revenue'], MIN_REVENUE_CNY)

    def test_prefers_cogs_over_total_operating_cost(self):
        income = pd.DataFrame({
            'stock_code': ['a', 'a', 'b'],
            'company_name': ['甲', '甲', '乙'],
            'item': ['营业成本', '营业总成本', '营业总成本'],
            'current': [100, 999, 50],
        })
        cost = pick_operating_cost(income)
        by_code = cost.set_index('stock_code')['value']
        self.assertEqual(by_code['a'], 100)
        self.assertEqual(by_code['b'], 50)

    def test_missing_tokens_become_nan_negatives_stay(self):
        self.assertTrue(is_missing_token("--"))
        self.assertTrue(is_missing_token("未知"))
        self.assertTrue(is_missing_token("  "))
        self.assertTrue(pd.isna(to_numeric_safe("--")))
        self.assertTrue(pd.isna(to_numeric_safe("—")))
        self.assertAlmostEqual(to_numeric_safe("-123.5"), -123.5)
        self.assertAlmostEqual(to_numeric_safe("1,000"), 1000.0)

    def test_clips_gross_margin_extremes(self):
        s = pd.Series([-4000.0, 21.0, 200.0])
        clipped = clip_gross_margin(s)
        self.assertEqual(clipped.min(), -50.0)
        self.assertEqual(clipped.max(), 80.0)


if __name__ == '__main__':
    unittest.main()
