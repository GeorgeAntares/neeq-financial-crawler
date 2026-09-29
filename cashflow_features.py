"""Shared cash-flow feature extraction for analysis / ML / DL scripts."""
import glob
import os

import numpy as np
import pandas as pd

from financial_analysis import to_numeric_safe

FEATURE_ITEMS = {
    'sales_cash': ['销售商品、提供劳务收到的现金'],
    'tax_refund': ['收到的税费返还'],
    'purchase_cash': ['购买商品、接受劳务支付的现金'],
    'employee_cost': ['支付给职工以及为职工支付的现金'],
    'tax_paid': ['支付的各项税费'],
    'ocf': ['经营活动产生的现金流量净额'],
    'icf': ['投资活动产生的现金流量净额'],
    'fcf': ['筹资活动产生的现金流量净额'],
    'net_cash_increase': ['现金及现金等价物净增加额'],
}

FEATURE_COLS_MODEL = [
    'sales_cash',
    'tax_refund',
    'purchase_cash',
    'employee_cost',
    'tax_paid',
    'icf',
    'fcf',
]


def extract_features(csv_dir):
    """One row per cash-flow CSV: company + line-item amounts."""
    files = glob.glob(os.path.join(csv_dir, '*现金流量表*.csv'))
    records = []
    for path in files:
        parts = os.path.basename(path).replace('.csv', '').split('_')
        if len(parts) < 4:
            continue
        try:
            frame = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            continue
        cols = frame.columns.tolist()
        if len(cols) < 2:
            continue
        item_col, val_col = cols[0], cols[1]
        features = {'stock_code': parts[0], 'company_name': parts[1]}
        for _, row in frame.iterrows():
            item = str(row[item_col]).strip()
            val = to_numeric_safe(row[val_col])
            if pd.isna(val):
                continue
            for feat_name, keywords in FEATURE_ITEMS.items():
                if any(keyword in item for keyword in keywords):
                    if feat_name not in features or pd.isna(features.get(feat_name)):
                        features[feat_name] = val
        records.append(features)
    return pd.DataFrame(records)


def split_and_scale(X, y, seed=42, test_size=0.2, val_size=0.2):
    """Fit StandardScaler on train only; carve val from train for early stopping."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test, scaler


def labeled_cashflow_frame(csv_dir):
    """Drop rows missing OCF / net cash; fill model features; add healthy label."""
    raw = extract_features(csv_dir)
    if raw.empty:
        return raw
    df = raw.dropna(subset=['ocf', 'net_cash_increase']).copy()
    for col in FEATURE_COLS_MODEL:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    df['healthy'] = ((df['ocf'] > 0) & (df['net_cash_increase'] > 0)).astype(int)
    return df
