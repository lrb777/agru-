import os, time, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"D:\agru择时")

import torch
import numpy as np
import pandas as pd

import config
config.TRAIN["max_epochs"] = 3
config.TRAIN["batch_size"] = 2048
config.TRAIN["patience"] = 3

from src.features import FeatureManager
from src.dataset import make_market_labels, StreamingTimingDataset
from src.model import AGRU
from src.metrics import tsrankic
from src.backtest import Backtester

t0 = time.time()

print("=== 加载数据 (仅2022-2025) ===", flush=True)
data_dir = r"D:\agru择时\data"
idx = pd.read_parquet(os.path.join(data_dir, "index_daily.parquet"))
idx["date"] = pd.to_datetime(idx["date"])
print(f"  指数: {len(idx)} 行 ({time.time()-t0:.1f}s)", flush=True)

sel = ["date", "code"] + [c for c in config.SELECTED_FEATURES if c != "date" and c != "code"]
parts = []
for yr in [2022, 2023, 2024, 2025]:
    f = os.path.join(data_dir, f"factors_{yr}.parquet")
    df = pd.read_parquet(f)
    df = df[[c for c in sel if c in df.columns]]
    parts.append(df)
    print(f"  {yr}: {len(df):,} 行 ({time.time()-t0:.1f}s)", flush=True)

factors = pd.concat(parts, ignore_index=True)
factors["date"] = pd.to_datetime(factors["date"])
print(f"  合计: {len(factors):,} 行, {len(factors.columns)-2} 特征 ({time.time()-t0:.1f}s)", flush=True)

fm = FeatureManager()
fm.feature_cols = [c for c in sel if c not in ("date", "code")]
for col in fm.feature_cols:
    factors[col] = pd.to_numeric(factors[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

all_codes = factors["code"].unique()
np.random.seed(42)
sampled = set(np.random.choice(all_codes, size=min(500, len(all_codes)), replace=False))
factors = factors[factors["code"].isin(sampled)].copy()
print(f"  采样: {len(factors):,} 行, {len(sampled)} 股票 ({time.time()-t0:.1f}s)", flush=True)

print("\n=== 构建数据集 ===", flush=True)
feat = factors.copy()
feat["year"] = feat["date"].dt.year

train_feat = feat[feat["year"].between(2022, 2023)]
val_feat = feat[feat["year"] == 2024]
test_feat = feat[feat["year"] == 2025]

labels = make_market_labels(idx, fit_dates=idx[idx["date"].dt.year == 2022]["date"])
labels["date"] = pd.to_datetime(labels["date"])

T = config.FEATURES["lookback_window"]
fc = fm.feature_cols

train_ds = StreamingTimingDataset(train_feat, labels, T, fc)
val_ds = StreamingTimingDataset(val_feat, labels, T, fc)
test_ds = StreamingTimingDataset(test_feat, labels, T, fc)
print(f"  train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,} ({time.time()-t0:.1f}s)", flush=True)

print("\n=== 训练 ===", flush=True)
device = torch.device("cpu")
torch.manual_seed(42)
model = AGRU(
    input_dim=len(fc),
    hidden_dim=config.MODEL["hidden_dim"],
    num_gru_layers=config.MODEL["num_gru_layers"],
    num_classes=config.MODEL["num_classes"],
    dropout=config.MODEL["dropout"],
).to(device)

from torch.utils.data import DataLoader
import torch.nn as nn

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=True, num_workers=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def predict_daily(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    scores = []
    with torch.no_grad():
        for xb, _ in loader:
            p = model.predict_proba(xb).cpu().numpy()
            scores.append(p[:, 2] - p[:, 0])
    scores = np.concatenate(scores)
    d2s = {}
    for d, s in zip(ds.sample_dates, scores):
        d2s.setdefault(d, []).append(float(s))
    return [(d, np.mean(v)) for d, v in sorted(d2s.items())]

best_ic = -999
best_state = None
for epoch in range(3):
    model.train()
    total_loss = 0
    nb = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nb += 1
    print(f"  Epoch {epoch+1} | loss={total_loss/nb:.4f} ({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    vs = predict_daily(model, val_ds)
    vl = val_ds.label_map
    common = sorted(set(dict(vs).keys()) & set(vl.keys()))
    if len(common) >= 2:
        ic = tsrankic([dict(vs)[d] for d in common], [vl[d] for d in common])
    else:
        ic = 0
    print(f"    val_TSRankIC={ic:.4f}", flush=True)
    if ic > best_ic:
        best_ic = ic
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state:
    model.load_state_dict(best_state)

print(f"\n=== 预测与回测 ===", flush=True)
val_scores = predict_daily(model, val_ds)
test_scores = predict_daily(model, test_ds)

arr = np.array([s for _, s in val_scores])
t_low = np.quantile(arr, 0.3194)
t_high = np.quantile(arr, 0.7353)

rows = []
for d, score in test_scores:
    if score > t_high:
        sig = 2
    elif score < t_low:
        sig = 0
    else:
        sig = 1
    rows.append({"date": d, "signal": sig, "score": float(score)})
signals = pd.DataFrame(rows)
print(f"  信号分布: {signals['signal'].value_counts().to_dict()}", flush=True)

bt = Backtester(idx)
result = bt.run(signals)
m = result["metrics"]
print(f"\n策略表现:")
for k, v in m.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
print(f"\n信号准确性:")
print(result["signal_accuracy"].to_string())
print(f"\n总耗时: {time.time()-t0:.1f}s", flush=True)

signals.to_csv(os.path.join(config.RESULT_DIR, "signals_test.csv"), index=False)
