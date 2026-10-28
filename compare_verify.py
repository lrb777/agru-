import pandas as pd
import numpy as np
import dolphindb as ddb

dp = pd.read_csv(r'results/daily_predict.csv')
dp['date'] = pd.to_datetime(dp['date'])
dp = dp.sort_values('date').reset_index(drop=True)

session = ddb.session()
session.connect('10.65.0.75', 8902, 'admin', '123456')
script = (
    't = loadTable("dfs://d0_day", "jy_qt_indexquote")\n'
    's = loadTable("dfs://d0_day", "jy_secumain")\n'
    'select s.SecuCode as code, t.TradingDay as date, t.ClosePrice as close\n'
    'from t join s on t.InnerCode = s.InnerCode\n'
    'where s.SecuCode = "000985"\n'
    '  and t.TradingDay between 2026.08.28 and 2026.10.30\n'
    'order by t.TradingDay'
)
idx = session.run(script)
session.close()
idx['date'] = pd.to_datetime(idx['date'])
idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
idx = idx.sort_values('date').reset_index(drop=True)

idx['close_ret'] = idx['close'].pct_change()
idx['future_ret'] = idx['close'].shift(-1) / idx['close'] - 1

merged = pd.merge(dp, idx[['date', 'close_ret', 'future_ret']], on='date', how='left')
recent = merged[merged['date'] >= '2026-09-01'].dropna(subset=['future_ret']).reset_index(drop=True)

sig_labels = {0: '空仓', 1: '半仓', 2: '满仓'}

print('日期          信号      score      当日涨跌   验证(当日)  次日涨跌   验证(次日)')
print('------------ -------- ---------- ---------- ---------- ---------- ----------')
for _, r in recent.iterrows():
    s = int(r['signal'])
    cr = r['close_ret']
    fr = r['future_ret']
    if s == 0:
        v1 = '正确' if cr < 0 else '错误'
        v2 = '正确' if fr < 0 else '错误'
    elif s == 2:
        v1 = '正确' if cr > 0 else '错误'
        v2 = '正确' if fr > 0 else '错误'
    else:
        v1 = '中性'
        v2 = '中性'
    print(f'{r["date"].strftime("%Y-%m-%d")}  {s} {sig_labels[s]:<4} {r["score"]:>10.6f} {cr:>+9.2%} {v1:>6}  {fr:>+9.2%} {v2:>6}')

print()
for label, col in [('当日涨跌(close_ret)', 'close_ret'), ('次日涨跌(future_ret)', 'future_ret')]:
    correct_total = 0
    n_valid = 0
    for _, r in recent.iterrows():
        s = r['signal']
        ret = r[col]
        if s in [0, 2]:
            n_valid += 1
            if (s == 0 and ret < 0) or (s == 2 and ret > 0):
                correct_total += 1
    acc = correct_total / n_valid * 100 if n_valid > 0 else 0
    print(f'{label}: 方向准确率 {correct_total}/{n_valid} = {acc:.1f}%')
