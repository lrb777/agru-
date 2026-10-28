import pandas as pd, numpy as np
from scipy.stats import spearmanr
from config import LABEL

idx = pd.read_parquet(r'data/index_daily.parquet')
idx['date'] = pd.to_datetime(idx['date'])
idx = idx.sort_values('date').reset_index(drop=True)

df = idx[['date','close']].sort_values('date').copy()
df['future_ret'] = df['close'].shift(-1) / df['close'] - 1
df = df.dropna(subset=['future_ret'])
df['year'] = df['date'].dt.year

# 2026折训练集 2016-2023
fit = df[df['year'].between(2016, 2023)]['future_ret']
t1 = fit.quantile(LABEL['class_ratios'][0])
t2 = fit.quantile(LABEL['class_ratios'][0] + LABEL['class_ratios'][1])
print(f'2026折 训练集分位数阈值: t1={t1:.6f}, t2={t2:.6f}')

def classify(r):
    if r < t1: return 0
    elif r < t2: return 1
    else: return 2
df['label'] = df['future_ret'].apply(classify)

for name, yrs in [('训练集2016-2023', [2016,2023]), ('验证集2024-2025', [2024,2025]), ('测试集2026', [2026,2026])]:
    sub = df[df['year'].between(yrs[0], yrs[1])]
    dist = sub['label'].value_counts().sort_index().to_dict()
    rho, _ = spearmanr(sub['label'], sub['future_ret'])
    means = sub.groupby('label')['future_ret'].mean().to_dict()
    print(f'\n{name}:')
    print(f'  标签分布: {dist}')
    print(f'  label vs future_ret spearman: {rho:.4f}')
    print(f'  各label未来收益均值: {means}')

# 验证集 IC 口径对比
# train.py 用的是 spearman(score, label)
# 研报用的是 spearman(score, future_ret)
# label 是 0/1/2 离散值，future_ret 是连续值
# 如果模型预测score和label相关，但因为label离散化损失信息，
# 可能在验证集上 IC(基于label) 虚高，但实际 IC(基于ret) 很低

print('\n--- 关键对比：两种IC口径 ---')
sig = pd.read_csv(r'results/signals.csv')
sig['date'] = pd.to_datetime(sig['date'])
merged = pd.merge(sig[['date','score']], df[['date','future_ret','label']], on='date', how='inner')

for yr in [2024, 2025, 2026]:
    sub = merged[merged['date'].dt.year == yr]
    if len(sub) < 2: continue
    ic_label, _ = spearmanr(sub['score'], sub['label'])
    ic_ret, _ = spearmanr(sub['score'], sub['future_ret'])
    print(f'  {yr}: IC(score,label)={ic_label:.4f}  IC(score,ret)={ic_ret:.4f}  差异={ic_label-ic_ret:.4f}')
