import pandas as pd
import numpy as np

# 1. 训练时生成的2026信号 (Jan-Aug)
sig_2026 = pd.read_csv(r'checkpoints/signals_2026.csv')
sig_2026['date'] = pd.to_datetime(sig_2026['date'])
print('=== 训练时2026信号 (Jan-Aug) ===')
print(f'  天数: {len(sig_2026)}')
print(f'  分布: {sig_2026["signal"].value_counts().sort_index().to_dict()}')
print(f'  score 范围: [{sig_2026["score"].min():.4f}, {sig_2026["score"].max():.4f}]')
print(f'  score 均值: {sig_2026["score"].mean():.4f}')
print(f'  score > 0.0959 (满仓阈值): {(sig_2026["score"] > 0.0959).sum()} 天')

# 2. 批量预测的信号 (Sep-Oct)
dp = pd.read_csv(r'results/daily_predict.csv')
dp['date'] = pd.to_datetime(dp['date'])
batch = dp[(dp['date'] >= '2026-09-02') & (dp['date'] <= '2026-10-16')]
print(f'\n=== 批量预测 (Sep-Oct) ===')
print(f'  天数: {len(batch)}')
print(f'  分布: {batch["signal"].value_counts().sort_index().to_dict()}')
print(f'  score 范围: [{batch["score"].min():.4f}, {batch["score"].max():.4f}]')
print(f'  score 均值: {batch["score"].mean():.4f}')
print(f'  score > 0.0959 (满仓阈值): {(batch["score"] > 0.0959).sum()} 天')

# 3. 逐月统计
sig_all = pd.read_csv(r'results/signals.csv')
sig_all['date'] = pd.to_datetime(sig_all['date'])
sig_all['year_month'] = sig_all['date'].dt.to_period('M')
monthly = sig_all.groupby('year_month').agg(
    n_days=('signal', 'count'),
    n_full=('signal', lambda x: (x == 2).sum()),
    n_empty=('signal', lambda x: (x == 0).sum()),
    mean_score=('score', 'mean'),
    max_score=('score', 'max'),
)
monthly['full_ratio'] = (monthly['n_full'] / monthly['n_days'] * 100).round(1)
print(f'\n=== 逐月统计 ===')
print(monthly.tail(20).to_string())

# 4. 验证集score分布 vs 近期score分布
import pickle
with open(r'checkpoints/stats_2026.pkl', 'rb') as f:
    stats = pickle.load(f)
val_scores = [s for _, s in stats['val_scores']]
print(f'\n=== 验证集(2024-2025) score分布 ===')
print(f'  均值: {np.mean(val_scores):.4f}')
print(f'  中位数: {np.median(val_scores):.4f}')
print(f'  > 0的天数: {sum(1 for s in val_scores if s > 0)}/{len(val_scores)} ({sum(1 for s in val_scores if s > 0)/len(val_scores)*100:.1f}%)')
print(f'  > 0.0959的天数: {sum(1 for s in val_scores if s > 0.0959)}/{len(val_scores)} ({sum(1 for s in val_scores if s > 0.0959)/len(val_scores)*100:.1f}%)')
print(f'  t_low={stats["threshold"][0]:.4f}, t_high={stats["threshold"][1]:.4f}')

print(f'\n=== 近期(9-10月) score分布 ===')
print(f'  均值: {batch["score"].mean():.4f}')
print(f'  > 0的天数: {(batch["score"] > 0).sum()}/{len(batch)} ({(batch["score"] > 0).sum()/len(batch)*100:.1f}%)')
print(f'  > 0.0959的天数: {(batch["score"] > 0.0959).sum()}/{len(batch)} ({(batch["score"] > 0.0959).sum()/len(batch)*100:.1f}%)')
