import pandas as pd
import numpy as np
import dolphindb as ddb

# 合并 signals.csv + daily_predict.csv
sig = pd.read_csv(r'results/signals.csv')
sig['date'] = pd.to_datetime(sig['date'])

dp = pd.read_csv(r'results/daily_predict.csv')
dp['date'] = pd.to_datetime(dp['date'])
dp = dp[['date', 'signal', 'score']]
dp = dp[~dp['date'].isin(sig['date'])]

combined = pd.concat([sig[['date','signal','score']], dp], ignore_index=True)
combined = combined.sort_values('date').reset_index(drop=True)

# 拉取指数收益
session = ddb.session()
session.connect('10.65.0.75', 8902, 'admin', '123456')
script = (
    't = loadTable("dfs://d0_day", "jy_qt_indexquote")\n'
    's = loadTable("dfs://d0_day", "jy_secumain")\n'
    'select s.SecuCode as code, t.TradingDay as date, t.ClosePrice as close\n'
    'from t join s on t.InnerCode = s.InnerCode\n'
    'where s.SecuCode = "000985"\n'
    '  and t.TradingDay between 2026.08.25 and 2026.10.30\n'
    'order by t.TradingDay'
)
idx = session.run(script)
session.close()
idx['date'] = pd.to_datetime(idx['date'])
idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
idx = idx.sort_values('date').reset_index(drop=True)
idx['future_ret'] = idx['close'].shift(-1) / idx['close'] - 1

merged = pd.merge(combined, idx[['date', 'future_ret']], on='date', how='left')
merged = merged.dropna(subset=['future_ret'])

recent = merged[merged['date'] >= '2026-09-01'].reset_index(drop=True)

sig_labels = {0: '空仓', 1: '半仓', 2: '满仓'}
print(f'=== 2026年9-10月信号准确率 ({len(recent)} 天) ===')
print()

for s in [0, 1, 2]:
    sub = recent[recent['signal'] == s]
    if len(sub) == 0:
        continue
    if s == 0:
        correct = (sub['future_ret'] < 0).sum()
    elif s == 2:
        correct = (sub['future_ret'] > 0).sum()
    else:
        correct = (sub['future_ret'] >= 0).sum()
    print(f'  {s} {sig_labels[s]}: {correct}/{len(sub)} = {correct/len(sub)*100:.1f}%  次日均值={sub["future_ret"].mean()*100:+.4f}%')

correct_total = 0
for _, r in recent.iterrows():
    s = r['signal']
    ret = r['future_ret']
    if (s == 0 and ret < 0) or (s == 1 and ret >= 0) or (s == 2 and ret > 0):
        correct_total += 1
print(f'\n  整体准确率: {correct_total}/{len(recent)} = {correct_total/len(recent)*100:.1f}%')

print(f'\n  日期          信号      score      次日涨跌   验证')
print(f'  ------------ -------- ---------- ---------- ------')
for _, r in recent.iterrows():
    s = int(r['signal'])
    ret = r['future_ret']
    if s == 0:
        v = '正确' if ret < 0 else '错误'
    elif s == 2:
        v = '正确' if ret > 0 else '错误'
    else:
        v = '中性'
    print(f'  {r["date"].strftime("%Y-%m-%d")}  {s} {sig_labels[s]:<4} {r["score"]:>10.6f} {ret:>+9.2%} {v}')
