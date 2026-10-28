import pandas as pd
import numpy as np
import pickle

# 加载信号
sig = pd.read_csv(r'results/signals.csv')
sig['date'] = pd.to_datetime(sig['date'])

# 加载阈值
with open(r'checkpoints/stats_2026.pkl', 'rb') as f:
    stats = pickle.load(f)
t_low, t_high = stats['threshold']

print(f'=== 阈值 ===')
print(f'  t_low={t_low:.4f}, t_high={t_high:.4f}')
print(f'  score 区间宽度: 空仓[{sig["score"].min():.4f}, {t_low:.4f}] = {t_low - sig["score"].min():.4f}')
print(f'  score 区间宽度: 半仓[{t_low:.4f}, {t_high:.4f}] = {t_high - t_low:.4f}')
print(f'  score 区间宽度: 满仓[{t_high:.4f}, {sig["score"].max():.4f}] = {sig["score"].max() - t_high:.4f}')

print(f'\n=== 信号分布 ===')
for s in [0, 1, 2]:
    sub = sig[sig['signal'] == s]
    print(f'  信号{s}: {len(sub):4d}天 ({len(sub)/len(sig)*100:.1f}%)  score范围[{sub["score"].min():.4f}, {sub["score"].max():.4f}]  均值{sub["score"].mean():.4f}')

# score分布
print(f'\n=== score 分布 ===')
print(f'  均值: {sig["score"].mean():.4f}')
print(f'  中位数: {sig["score"].median():.4f}')
print(f'  标准差: {sig["score"].std():.4f}')
pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
for p in pcts:
    print(f'  {p}%分位: {sig["score"].quantile(p/100):.4f}')

# 逐年信号分布
print(f'\n=== 逐年信号分布 ===')
sig['year'] = sig['date'].dt.year
for yr in sorted(sig['year'].unique()):
    sub = sig[sig['year'] == yr]
    dist = sub['signal'].value_counts().sort_index()
    s0 = dist.get(0, 0)
    s1 = dist.get(1, 0)
    s2 = dist.get(2, 0)
    n = len(sub)
    print(f'  {yr}: 0={s0:3d}({s0/n*100:.0f}%)  1={s1:3d}({s1/n*100:.0f}%)  2={s2:3d}({s2/n*100:.0f}%)  score均值={sub["score"].mean():.4f}')

# 半仓信号内部的区分度
print(f'\n=== 半仓信号(1)内部细分 ===')
half = sig[sig['signal'] == 1].copy()
half['sub_group'] = pd.cut(half['score'], bins=5, labels=['深半仓(-)', '偏空半仓', '中性半仓', '偏多半仓', '浅半仓(+)'])
print(half.groupby('sub_group')['score'].agg(['count', 'mean']))

# 对比: 不同信号的次日收益
idx = pd.read_parquet(r'data/index_daily.parquet')
idx['date'] = pd.to_datetime(idx['date'])
idx = idx.sort_values('date').reset_index(drop=True)
idx['future_ret'] = idx['close'].shift(-1) / idx['close'] - 1
merged = pd.merge(sig[['date', 'signal', 'score']], idx[['date', 'future_ret']], on='date', how='inner').dropna()

print(f'\n=== 各信号次日收益 ===')
for s in [0, 1, 2]:
    sub = merged[merged['signal'] == s]
    print(f'  信号{s}: 均值={sub["future_ret"].mean()*100:+.4f}%  胜率={((sub["future_ret"]>0).mean())*100:.1f}%  n={len(sub)}')

# 半仓信号按score五等分的次日收益
print(f'\n=== 半仓信号按score五等分的次日收益 ===')
half_merged = merged[merged['signal'] == 1].copy()
half_merged['quintile'] = pd.qcut(half_merged['score'], 5, labels=['Q1最低', 'Q2', 'Q3', 'Q4', 'Q5最高'])
for q in ['Q1最低', 'Q2', 'Q3', 'Q4', 'Q5最高']:
    sub = half_merged[half_merged['quintile'] == q]
    print(f'  {q}: score范围[{sub["score"].min():.4f}, {sub["score"].max():.4f}]  次日均值={sub["future_ret"].mean()*100:+.4f}%  胜率={(sub["future_ret"]>0).mean()*100:.1f}%  n={len(sub)}')
