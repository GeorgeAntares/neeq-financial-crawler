"""
NEEQ 财报数据分析 / NEEQ Financial Data Analysis
=================================================
新三板企业财务数据采集与分析项目 — 数据分析模块
NEEQ Financial Data Collection & Analysis Project — Data Analysis Module

描述统计入口（营收 / 毛利率直方图 / OCF 正负）。行业营运资金与盈余质量
在 industry_portrait.py；报表质量标记在 company_metrics.py。
营收低于 10 万元是样本筛选（附注编号），不是报表勾稽清洗。
"""

import pandas as pd
import numpy as np
import glob
import os
import re
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 配置 / Configuration
# ============================================================

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CSV_255 = os.path.join(_ROOT, 'output', 'analysis', '_csv_255')
_CSV_MIXED = os.path.join(_ROOT, 'output', 'csv')
CSV_DIR = _CSV_255 if os.path.isdir(_CSV_255) else _CSV_MIXED
OUTPUT_DIR = os.path.join(_ROOT, 'output', 'analysis')

# 低于该金额的「营收」多半是附注编号/错列（1.0、17.4），不进入描述统计
MIN_REVENUE_CNY = 100_000
# 毛利率均值和直方图用截尾，避免 -4000% 拉垮均值；中位数仍用原始值
GROSS_MARGIN_CLIP = (-50.0, 80.0)
CHART_NAME = 'financial_analysis_clean.png'
SUMMARY_NAME = 'summary_statistics_clean.csv'

# ============================================================
# 1. 数据加载 / Data Loading
# ============================================================

def load_csvs(statement_type):
    """
    加载指定类型的所有 CSV 文件 / Load all CSVs of a given statement type

    Parameters:
        statement_type: '利润表' or '现金流量表' or '资产负债表'
    Returns:
        DataFrame with columns: [stock_code, company_name, year, item, current, prior]
    """
    pattern = os.path.join(CSV_DIR, f'*{statement_type}*.csv')
    files = glob.glob(pattern)

    records = []
    for f in files:
        filename = os.path.basename(f)
        # 解析文件名: {code}_{name}_{year}_合并{type}.csv
        parts = filename.replace('.csv', '').split('_')
        if len(parts) < 4:
            continue
        stock_code = parts[0]
        company_name = parts[1]
        year = parts[2]

        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
        except Exception:
            continue

        # 获取列名
        cols = df.columns.tolist()
        if len(cols) < 2:
            continue

        item_col = cols[0]
        val_col = cols[1] if len(cols) >= 2 else None
        prior_col = cols[2] if len(cols) >= 3 else None

        for _, row in df.iterrows():
            item = str(row[item_col]).strip()
            if not item or item == 'nan' or '年' in item and len(item) < 10:
                continue

            current = row[val_col] if val_col else None
            prior = row[prior_col] if prior_col else None

            records.append({
                'stock_code': stock_code,
                'company_name': company_name,
                'year': year,
                'item': item,
                'current': current,
                'prior': prior
            })

    return pd.DataFrame(records)


MISSING_TOKENS = {
    "",
    "-",
    "--",
    "---",
    "—",
    "–",
    "－",
    "/",
    "\\",
    "*",
    "无",
    "未知",
    "不适用",
    "n/a",
    "na",
    "none",
    "null",
    "nan",
    "nat",
}


def is_missing_token(value):
    """True for blank / dash / 未知-style placeholders, not for numeric '-123'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if isinstance(value, (int, float)):
        return False
    text = str(value).replace("\u3000", " ").strip()
    if text == "":
        return True
    return text.lower() in MISSING_TOKENS


def to_numeric_safe(value):
    """Unify missing tokens then parse a number. '-123' stays numeric; '--' becomes NaN."""
    if pd.isna(value) or value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if is_missing_token(value):
        return np.nan
    s = str(value).replace(",", "").replace("%", "").replace('"', "").replace("\u3000", " ").strip()
    s = s.replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def pick_primary_row(df, keywords, value_col='current'):
    """
    从明细行中挑出"主表行"，并保证每家公司只保留一条。

    直接用 str.contains 做子串匹配会把附注明细行一并命中（例如
    "营业收入" 会同时命中 "其中：营业收入"，现金流量表子项也会命中
    "经营活动产生的现金流量净额"），导致同一公司被重复计数、
    统计口径虚高。这里做两步收敛：
      1. 排除以"其中"开头的附注细分行；
      2. 同等条件下优先取与关键词完全相等（去空白后）的行，
         其次取项目名最短的行（主表行通常最简短）。
    :return: 每家公司一行的 DataFrame，附带 value 列
    """
    matched = df[df['item'].str.contains('|'.join(keywords), na=False)].copy()
    if matched.empty:
        matched['value'] = np.nan
        return matched

    # 排除附注细分行（"其中：xxx"）
    primary = matched[~matched['item'].str.strip().str.startswith('其中')].copy()
    if primary.empty:
        primary = matched.copy()

    primary['value'] = primary[value_col].apply(to_numeric_safe)
    primary = primary.dropna(subset=['value'])

    if primary.empty:
        return primary

    # 排序优先级：完全匹配关键词优先，其次项目名更短（更像主表行）
    stripped = primary['item'].str.strip()
    primary['_exact'] = stripped.isin(keywords).astype(int)
    primary['_len'] = stripped.str.len()
    primary = primary.sort_values(['_exact', '_len'], ascending=[False, True])
    primary = primary.drop_duplicates(subset=['stock_code'], keep='first')
    return primary.drop(columns=['_exact', '_len'])


def pick_operating_cost(income_df):
    """Prefer 营业成本 (COGS); fill with 营业总成本 only when COGS is missing."""
    cogs = pick_primary_row(income_df, ['营业成本'])
    fallback = pick_primary_row(income_df, ['营业总成本'])
    if cogs.empty:
        return fallback
    if fallback.empty:
        return cogs
    missing = set(fallback['stock_code']) - set(cogs['stock_code'])
    extra = fallback[fallback['stock_code'].isin(missing)]
    return pd.concat([cogs, extra], ignore_index=True)


def filter_plausible_revenue(revenue_df, min_revenue=MIN_REVENUE_CNY):
    """Drop tiny amounts that are usually footnote ids, not revenue."""
    if revenue_df.empty:
        return revenue_df, 0
    kept = revenue_df[revenue_df['revenue'] >= min_revenue].copy()
    return kept, len(revenue_df) - len(kept)


def clip_gross_margin(series, bounds=GROSS_MARGIN_CLIP):
    return series.clip(lower=bounds[0], upper=bounds[1])


def main(csv_dir=None, output_dir=None):
    global CSV_DIR, OUTPUT_DIR
    if csv_dir:
        CSV_DIR = csv_dir
    if output_dir:
        OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("NEEQ 财报数据分析 / NEEQ Financial Data Analysis")
    print("=" * 60)

    # 加载利润表 / Load income statements
    print("\n[1/6] 加载利润表数据 / Loading income statement data...")
    income_df = load_csvs('利润表')
    print(f"  利润表记录数 Records: {len(income_df)}")
    print(f"  企业数 Companies: {income_df['stock_code'].nunique()}")

    # 加载现金流量表 / Load cash flow statements
    print("\n[2/6] 加载现金流量表数据 / Loading cash flow data...")
    cashflow_df = load_csvs('现金流量表')
    print(f"  现金流量表记录数 Records: {len(cashflow_df)}")
    print(f"  企业数 Companies: {cashflow_df['stock_code'].nunique()}")

    # ============================================================
    # 2. 营收分析 / Revenue Analysis
    # ============================================================

    print("\n[3/6] 营收分析 / Revenue Analysis...")

    revenue_df = pick_primary_row(income_df, ['营业总收入', '营业收入'])
    revenue_df = revenue_df.rename(columns={'value': 'revenue'})
    revenue_df = revenue_df[revenue_df['revenue'] > 0]
    revenue_df, n_dropped_rev = filter_plausible_revenue(revenue_df)
    if n_dropped_rev:
        print(f"  清洗：去掉营收 < {MIN_REVENUE_CNY:,.0f} 元的 {n_dropped_rev} 家（多为附注编号误入金额列）")

    if revenue_df.empty:
        print('  无有效营收样本，结束。')
        return

    print(f"\n  有效营收数据企业 Companies with valid revenue: {len(revenue_df)}")
    print(f"  （利润表覆盖企业 Income statement coverage: {income_df['stock_code'].nunique()} 家，"
          f"有效率 {len(revenue_df) / max(income_df['stock_code'].nunique(), 1) * 100:.1f}%）")
    print(f"  营收均值 Mean revenue: {revenue_df['revenue'].mean() / 1e8:.2f} 亿元 / 100M CNY")
    print(f"  营收中位数 Median revenue: {revenue_df['revenue'].median() / 1e8:.2f} 亿元 / 100M CNY")
    print(f"  营收标准差 Std: {revenue_df['revenue'].std() / 1e8:.2f} 亿元 / 100M CNY")
    print(f"  营收最大值 Max: {revenue_df['revenue'].max() / 1e8:.2f} 亿元 / 100M CNY")
    print(f"  营收最小值 Min: {revenue_df['revenue'].min() / 1e4:.2f} 万元 / 10K CNY")

    # 营收分布区间 / Revenue distribution
    bins = [0, 1e7, 5e7, 1e8, 5e8, 1e9, float('inf')]
    labels = ['<1000万', '1000万-5000万', '5000万-1亿', '1亿-5亿', '5亿-10亿', '>10亿']
    revenue_df['revenue_range'] = pd.cut(revenue_df['revenue'], bins=bins, labels=labels)
    print("\n  营收分布 / Revenue Distribution:")
    dist = revenue_df['revenue_range'].value_counts().sort_index()
    for label, count in dist.items():
        pct = count / len(revenue_df) * 100
        print(f"    {label}: {count} 家 ({pct:.1f}%)")

    # 营收 Top 20 / Top 20 by revenue
    print("\n  营收 Top 20 企业 / Top 20 Companies by Revenue:")
    top20 = revenue_df.nlargest(20, 'revenue')[['company_name', 'stock_code', 'revenue']]
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"    {i:2d}. {row['company_name']} ({row['stock_code']}): {row['revenue'] / 1e8:.2f} 亿元")

    # ============================================================
    # 3. 盈利能力分析 / Profitability Analysis
    # ============================================================

    print("\n[4/6] 盈利能力分析 / Profitability Analysis...")

    cost_df = pick_operating_cost(income_df).rename(columns={'value': 'cost'})

    # 合并营收和成本 / Merge revenue and cost
    profit_df = revenue_df[['stock_code', 'company_name', 'revenue']].merge(
        cost_df[['stock_code', 'cost']], on='stock_code', how='inner'
    )
    profit_df['gross_profit'] = profit_df['revenue'] - profit_df['cost']
    profit_df['gross_margin'] = (profit_df['gross_profit'] / profit_df['revenue']) * 100
    profit_df['gross_margin_clipped'] = clip_gross_margin(profit_df['gross_margin'])

    print(f"  可计算毛利率的企业 Companies with gross margin: {len(profit_df)}")
    if profit_df.empty:
        print('  无毛利率样本。')
        return
    print(f"  毛利率中位数 Median gross margin: {profit_df['gross_margin'].median():.2f}%")
    print(
        f"  毛利率均值（截尾 {GROSS_MARGIN_CLIP[0]:.0f}%～{GROSS_MARGIN_CLIP[1]:.0f}%）"
        f" Clipped mean: {profit_df['gross_margin_clipped'].mean():.2f}%"
    )
    print(f"  毛利率原始均值 Raw mean (含极端值): {profit_df['gross_margin'].mean():.2f}%")

    # 盈亏分析 / Profit/Loss analysis
    profitable = profit_df[profit_df['gross_profit'] > 0]
    loss_making = profit_df[profit_df['gross_profit'] <= 0]
    n_profit = max(len(profit_df), 1)
    print(f"\n  盈利企业 Profitable: {len(profitable)} ({len(profitable)/n_profit*100:.1f}%)")
    print(f"  亏损企业 Loss-making: {len(loss_making)} ({len(loss_making)/n_profit*100:.1f}%)")

    # 毛利率分布 / Gross margin distribution
    margin_bins = [-float('inf'), 0, 10, 20, 30, 50, float('inf')]
    margin_labels = ['<0%', '0-10%', '10-20%', '20-30%', '30-50%', '>50%']
    profit_df['margin_range'] = pd.cut(profit_df['gross_margin'], bins=margin_bins, labels=margin_labels)
    print("\n  毛利率分布 / Gross Margin Distribution:")
    margin_dist = profit_df['margin_range'].value_counts().sort_index()
    for label, count in margin_dist.items():
        pct = count / len(profit_df) * 100
        print(f"    {label}: {count} 家 ({pct:.1f}%)")

    # ============================================================
    # 4. 现金流分析 / Cash Flow Analysis
    # ============================================================

    print("\n[5/6] 现金流分析 / Cash Flow Analysis...")

    operating_cf = pick_primary_row(cashflow_df, ['经营活动产生的现金流量净额'])
    operating_cf = operating_cf.rename(columns={'value': 'ocf'})

    print(f"  有效经营现金流企业 Companies with OCF data: {len(operating_cf)}")
    print(f"  （现金流量表覆盖企业 Cash flow statement coverage: "
          f"{cashflow_df['stock_code'].nunique()} 家）")
    print(f"  经营现金流均值 Mean OCF: {operating_cf['ocf'].mean() / 1e8:.2f} 亿元 / 100M CNY")
    print(f"  经营现金流中位数 Median OCF: {operating_cf['ocf'].median() / 1e4:.2f} 万元 / 10K CNY")

    positive_ocf = operating_cf[operating_cf['ocf'] > 0]
    negative_ocf = operating_cf[operating_cf['ocf'] <= 0]
    n_ocf = max(len(operating_cf), 1)
    print(f"\n  正现金流 Positive OCF: {len(positive_ocf)} ({len(positive_ocf)/n_ocf*100:.1f}%)")
    print(f"  负现金流 Negative OCF: {len(negative_ocf)} ({len(negative_ocf)/n_ocf*100:.1f}%)")

    # ============================================================
    # 5. 可视化 / Visualization
    # ============================================================

    print("\n[6/6] 生成图表 / Generating charts...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 图1: 营收分布直方图 / Revenue distribution histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    log_revenue = np.log10(revenue_df['revenue'][revenue_df['revenue'] > 0])
    ax1.hist(log_revenue, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('营收（log10 元）/ Revenue (log10 CNY)', fontsize=10)
    ax1.set_ylabel('企业数 / Number of Companies', fontsize=10)
    med_rev = revenue_df['revenue'].median()
    ax1.set_title(f'有效营收右偏，中位数 {med_rev/1e8:.2f} 亿元', fontsize=12, fontweight='bold')
    ax1.axvline(log_revenue.mean(), color='red', linestyle='--', label=f'均值 Mean: {10**log_revenue.mean()/1e8:.1f}亿')
    ax1.legend(fontsize=9)

    # 图2: 毛利率分布 / Gross margin distribution
    ax2 = axes[0, 1]
    margins = profit_df['gross_margin_clipped']
    ax2.hist(margins, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1, label='盈亏平衡 Break-even')
    ax2.set_xlabel('毛利率 (%) / Gross Margin (%)', fontsize=10)
    ax2.set_ylabel('企业数 / Number of Companies', fontsize=10)
    med_gm = float(np.nanmedian(profit_df['gross_margin'])) if 'gross_margin' in profit_df.columns else float(np.nanmedian(margins))
    ax2.set_title(f'毛利率中位数 {med_gm:.1f}%（直方图已截尾）', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)

    # 图3: 营收 Top 15 / Top 15 by revenue
    ax3 = axes[1, 0]
    top15 = revenue_df.nlargest(15, 'revenue').sort_values('revenue')
    bars = ax3.barh(top15['company_name'], top15['revenue'] / 1e8, color='teal', alpha=0.8)
    ax3.set_xlabel('营收（亿元）/ Revenue (100M CNY)', fontsize=10)
    ax3.set_title('营收前 15 家拉开了样本右尾', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelsize=8)

    # 图4: 经营现金流正负对比 / OCF positive vs negative
    ax4 = axes[1, 1]
    categories = ['正现金流\nPositive OCF', '负现金流\nNegative OCF']
    values = [len(positive_ocf), len(negative_ocf)]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax4.bar(categories, values, color=colors, edgecolor='white', alpha=0.85)
    ax4.set_ylabel('企业数 / Number of Companies', fontsize=10)
    pos_share = len(positive_ocf) / n_ocf * 100
    ax4.set_title(f'有 OCF 的企业中 {pos_share:.0f}% 经营现金为正', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, CHART_NAME)
    if os.path.exists(chart_path):
        os.remove(chart_path)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存 Chart saved: output/analysis/{CHART_NAME}")

    # ============================================================
    # 6. 汇总统计 / Summary Statistics
    # ============================================================

    summary = pd.DataFrame({
        '指标 Metric': [
            '利润表覆盖企业 Income Statement Coverage',
            '营收有效样本 Companies with Valid Revenue',
            '营收有效率 Revenue Valid Rate (%)',
            '现金流量表覆盖企业 Cash Flow Coverage',
            'OCF有效样本 Companies with Valid OCF',
            '营收均值 Mean Revenue (CNY)',
            '营收中位数 Median Revenue (CNY)',
            '毛利率均值（截尾）Clipped Mean Gross Margin (%)',
            '正现金流企业 Positive OCF (%)',
            '营收 Top 1 公司 Top 1 Revenue Company',
        ],
        '值 Value': [
            income_df['stock_code'].nunique(),
            len(revenue_df),
            f"{len(revenue_df) / max(income_df['stock_code'].nunique(), 1) * 100:.1f}%",
            cashflow_df['stock_code'].nunique(),
            len(operating_cf),
            f"{revenue_df['revenue'].mean():,.0f}",
            f"{revenue_df['revenue'].median():,.0f}",
            f"{profit_df['gross_margin_clipped'].mean():.2f}%",
            f"{len(positive_ocf) / max(len(operating_cf), 1) * 100:.1f}%",
            (
                f"{top20.iloc[0]['company_name']} ({top20.iloc[0]['revenue'] / 1e8:.2f}亿)"
                if len(top20) else ''
            ),
        ]
    })

    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_NAME)
    if os.path.exists(summary_path):
        os.remove(summary_path)
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  汇总统计已保存 Summary saved: output/analysis/{SUMMARY_NAME}")

    print("\n" + "=" * 60)
    print("分析完成 / Analysis Complete")
    print(f"输出目录 Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NEEQ financial descriptive stats')
    parser.add_argument('--csv-dir', default=None, help='CSV folder (default output/analysis/_csv_255 when present)')
    parser.add_argument('--output-dir', default=None, help='Chart/summary folder')
    args = parser.parse_args()
    main(csv_dir=args.csv_dir, output_dir=args.output_dir)
