import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dupont_pca import (
    add_size_logs,
    dupont_coverage,
    dupont_factor_assoc,
    dupont_identity,
    factor_bar_frame,
    fit_pca,
    industry_dupont_sensitivity,
    industry_dupont_table,
    log_variance_shares,
    spearman_matrix,
    standardize,
    theme_for_loadings,
)


class DupontPcaTest(unittest.TestCase):
    def test_identity_holds_on_product(self):
        nm, at, em = 0.1, 2.0, 1.5
        df = pd.DataFrame({
            "net_margin": [nm, 0.05],
            "asset_turnover": [at, 0.8],
            "equity_multiplier": [em, 3.0],
        })
        df["dupont_product"] = df["net_margin"] * df["asset_turnover"] * df["equity_multiplier"]
        df["roe"] = df["dupont_product"]
        result = dupont_identity(df)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_match"], 2)
        self.assertLess(result["max_abs_gap"], 1e-12)

    def test_spearman_diagonal_is_one(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 1, 4, 2, 3]})
        corr = spearman_matrix(df, ["a", "b", "c"])
        self.assertAlmostEqual(corr.loc["a", "a"], 1.0)
        self.assertAlmostEqual(corr.loc["a", "b"], 1.0)

    def test_pca_recovers_dominant_direction(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=120)
        matrix = np.column_stack([z, 1.5 * z + 0.02 * rng.normal(size=120)])
        scaled, _, _ = standardize(matrix)
        loadings, ratio, _ = fit_pca(scaled)
        self.assertGreater(ratio[0], 0.95)
        self.assertGreater(abs(loadings[0, 0]), 0.6)
        self.assertGreater(abs(loadings[1, 0]), 0.6)

    def test_theme_follows_largest_abs_group(self):
        names = [
            "log_revenue", "log_assets", "debt_ratio_w", "equity_multiplier_w",
            "current_ratio_w", "ar_to_revenue_w", "inventory_to_revenue_w", "ocf_to_revenue_w",
        ]
        loadings = np.array([0.7, 0.6, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01])
        self.assertEqual(theme_for_loadings(loadings, names), "规模")
        mixed = np.array([0.50, 0.45, 0.40, 0.27, 0.26, 0.17, 0.33, 0.34])
        self.assertEqual(theme_for_loadings(mixed, names), "规模")

    def test_log_variance_shares_sum_near_one_if_uncorrelated(self):
        rng = np.random.default_rng(1)
        n = 200
        log_nm = rng.normal(0, 1.0, size=n)
        log_at = rng.normal(0, 1.0, size=n)
        log_em = rng.normal(0, 1.0, size=n)
        df = pd.DataFrame({
            "net_margin": np.exp(log_nm),
            "asset_turnover": np.exp(log_at),
            "equity_multiplier": np.exp(log_em),
        })
        df["roe"] = df["net_margin"] * df["asset_turnover"] * df["equity_multiplier"]
        shares = log_variance_shares(df)
        total = shares["var_share"].sum()
        self.assertTrue((shares["var_share"] > 0.15).all())
        self.assertGreater(total, 0.7)
        self.assertLess(total, 1.3)

    def test_factor_assoc_tracks_net_margin(self):
        df = pd.DataFrame({
            "roe": [0.2, 0.1, 0.0, -0.1, -0.2],
            "net_margin": [0.2, 0.1, 0.0, -0.1, -0.2],
            "asset_turnover": [1.0, 1.0, 1.0, 1.0, 1.0],
            "equity_multiplier": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        assoc = dupont_factor_assoc(df).set_index("factor")
        self.assertAlmostEqual(assoc.loc["net_margin", "spearman_with_roe"], 1.0)

    def test_industry_dupont_uses_valid_roe_only(self):
        df = pd.DataFrame({
            "industry": ["制造", "制造", "软件信息", "软件信息"],
            "net_margin": [0.10, 0.08, 0.02, 0.01],
            "asset_turnover": [1.0, 1.0, 0.5, 0.5],
            "equity_multiplier": [2.0, 2.0, 2.0, 2.0],
            "roe": [0.20, 0.16, 0.02, 0.01],
            "dupont_product": [0.20, 0.16, 0.02, 0.01],
            "roe_valid": [True, True, True, False],
        })
        table = industry_dupont_table(df).set_index("industry")
        self.assertEqual(int(table.loc["制造", "n"]), 2)
        self.assertEqual(int(table.loc["软件信息", "n"]), 1)
        self.assertAlmostEqual(table.loc["制造", "median_roe"], 0.18)
        self.assertAlmostEqual(table.loc["制造", "product_of_medians"], 0.09 * 1.0 * 2.0)

    def test_dupont_coverage_counts_frozen_roe(self):
        df = pd.DataFrame({
            "revenue": [1e8, 2e8, 3e8],
            "net_profit": [10.0, np.nan, 20.0],
            "np_truncated": [False, True, False],
            "equity_negative": [False, False, True],
            "roe": [0.10, np.nan, np.nan],
            "roe_valid": [True, False, False],
            "net_margin": [0.05, np.nan, 0.02],
            "asset_turnover": [1.0, 1.0, 1.0],
            "equity_multiplier": [2.0, 2.0, np.nan],
        })
        cov = dupont_coverage(df).set_index("item")
        self.assertEqual(int(cov.loc["有效营收", "n"]), 3)
        self.assertEqual(int(cov.loc["有净利润", "n"]), 2)
        self.assertEqual(int(cov.loc["权益非正冻结", "n"]), 1)
        self.assertEqual(int(cov.loc["有效 ROE", "n"]), 1)
        self.assertEqual(int(cov.loc["四项齐全", "n"]), 1)

    def test_dupont_sensitivity_drops_roe_iqr(self):
        roe = [0.04] * 8 + [0.05] * 7 + [0.40]
        n = len(roe)
        df = pd.DataFrame({
            "industry": ["制造"] * n,
            "net_margin": [0.02] * n,
            "asset_turnover": [1.0] * n,
            "equity_multiplier": [2.0] * n,
            "roe": roe,
            "dupont_product": roe,
            "roe_valid": [True] * n,
            "roe_w": roe,
            "net_margin_w": [0.02] * n,
            "asset_turnover_w": [1.0] * n,
            "equity_multiplier_w": [2.0] * n,
        })
        sens = industry_dupont_sensitivity(df)
        quality = sens[(sens["treatment"] == "quality") & (sens["industry"] == "制造")].iloc[0]
        iqr = sens[(sens["treatment"] == "iqr") & (sens["industry"] == "制造")].iloc[0]
        self.assertEqual(int(quality["n"]), 16)
        self.assertEqual(int(iqr["n"]), 15)
        self.assertLess(float(iqr["median_roe"]), float(quality["median_roe"]))

    def test_add_size_logs_skips_nonpositive(self):
        df = add_size_logs(pd.DataFrame({"revenue": [100.0, 0.0], "avg_assets": [50.0, -1.0]}))
        self.assertTrue(np.isfinite(df.loc[0, "log_revenue"]))
        self.assertTrue(np.isnan(df.loc[1, "log_revenue"]))
        self.assertTrue(np.isnan(df.loc[1, "log_assets"]))

    def test_factor_bar_frame_keeps_nan_for_empty_industry(self):
        table = pd.DataFrame([
            {
                "industry": "制造",
                "n": 45,
                "median_net_margin": 0.032,
                "median_asset_turnover": 0.694,
                "median_equity_multiplier": 2.054,
                "median_roe": 0.044,
            },
            {
                "industry": "软件信息",
                "n": 0,
                "median_net_margin": np.nan,
                "median_asset_turnover": np.nan,
                "median_equity_multiplier": np.nan,
                "median_roe": np.nan,
            },
        ])
        frame = factor_bar_frame(table)
        sw = frame[frame["industry"] == "软件信息"].iloc[0]
        self.assertTrue(np.isnan(sw["median_net_margin"]))
        self.assertTrue(np.isnan(sw["median_roe"]))
        self.assertAlmostEqual(float(frame[frame["industry"] == "制造"].iloc[0]["median_roe"]), 0.044)

    def test_main_guard_present(self):
        text = Path(__file__).resolve().parents[1].joinpath("dupont_pca.py").read_text(encoding="utf-8")
        self.assertIn("def main(", text)
        self.assertIn('if __name__ == "__main__":', text)


if __name__ == "__main__":
    unittest.main()
