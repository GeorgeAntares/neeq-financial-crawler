import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cashflow_features import extract_features, labeled_cashflow_frame, split_and_scale


class CashflowFeaturesTest(unittest.TestCase):
    def test_extracts_ocf_and_health_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / '430001_测试_2025_合并现金流量表.csv'
            pd.DataFrame({
                '项目': [
                    '销售商品、提供劳务收到的现金',
                    '经营活动产生的现金流量净额',
                    '现金及现金等价物净增加额',
                ],
                '本期金额': [100, 40, 10],
            }).to_csv(path, index=False, encoding='utf-8-sig')
            raw = extract_features(temp_dir)
            labeled = labeled_cashflow_frame(temp_dir)
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw.iloc[0]['ocf'], 40)
        self.assertEqual(int(labeled.iloc[0]['healthy']), 1)

    def test_analysis_scripts_have_main_guards(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            'financial_analysis.py',
            'company_metrics.py',
            'industry_portrait.py',
            'dupont_pca.py',
            'cash_gap_model.py',
            'preprocess.py',
            'ml_financial_health.py',
            'ml_evaluation.py',
            'shap_analysis.py',
            'dl_financial_health.py',
        ):
            text = (root / name).read_text(encoding='utf-8')
            with self.subTest(name=name):
                self.assertIn('def main(', text)
                self.assertIn('if __name__', text)


class DlSplitScaleTest(unittest.TestCase):
    def test_scaler_matches_train_not_full_matrix(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest('sklearn not installed in CI')
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 7))
        y = np.array([0, 1] * 20)
        X_tr, X_val, X_test, y_tr, y_val, y_test, scaler = split_and_scale(X, y, seed=0)
        self.assertEqual(len(X_tr) + len(X_val) + len(X_test), 40)
        np.testing.assert_allclose(X_tr.mean(axis=0), 0, atol=0.35)
        full_mean = scaler.mean_
        self.assertFalse(np.allclose(full_mean, X.mean(axis=0)))


if __name__ == '__main__':
    unittest.main()
