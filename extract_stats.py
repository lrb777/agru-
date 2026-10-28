"""提取模型推理所需的标准化统计量和阈值（不重训）

用法:
  python extract_stats.py              # 默认提取 model_2026 的统计量
  python extract_stats.py --year 2025  # 指定模型年份

产出:
  checkpoints/stats_{year}.pkl
    - standardizer: 每个特征的 mean/std
    - threshold: (t_low, t_high) 验证集 score 分位数阈值
    - val_scores: 验证集每日 score 列表
    - train_range / val_range: 区间信息
"""
import os
import sys
import gc
import pickle
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import DATA_DIR, CHECKPOINT_DIR, SELECTED_FEATURES, MODEL, FEATURES, LABEL
from src.model import AGRU
from src.features import FeatureManager
from src.dataset import make_market_labels, TimingDataStore, TimingDataset


def load_fold_data(data_dir, feature_cols, year_start, year_end):
    keep = ["date", "code"] + [c for c in feature_cols]
    parts = []
    for yr in range(year_start, year_end + 1):
        slim = os.path.join(data_dir, f"factors_{yr}_slim.parquet")
        full = os.path.join(data_dir, f"factors_{yr}.parquet")
        if os.path.exists(slim):
            df = pd.read_parquet(slim)
        elif os.path.exists(full):
            df = pd.read_parquet(full)
            df = df[keep]
            for c in feature_cols:
                if c in df.columns:
                    df[c] = df[c].astype(np.float32)
        else:
            print(f"  !! 找不到 {yr} 的因子文件")
            continue
        parts.append(df)
        print(f"    {yr}: {len(df):,} 行")
    feat = pd.concat(parts, ignore_index=True)
    feat["date"] = pd.to_datetime(feat["date"])
    del parts
    gc.collect()
    return feat


def extract_stats(model_year: int):
    train_start = max(model_year - 10, 2015)
    train_end = model_year - 3
    val_start = model_year - 2
    val_end = model_year - 1

    print(f"=== 提取统计量: model_{model_year} ===")
    print(f"  训练集: {train_start}-{train_end}")
    print(f"  验证集: {val_start}-{val_end}")

    idx = pd.read_parquet(os.path.join(DATA_DIR, "index_daily.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx["year"] = idx["date"].dt.year

    train_dates = idx[(idx["year"] >= train_start) & (idx["year"] <= train_end)]["date"]
    labels = make_market_labels(
        idx,
        forecast_horizon=LABEL["forecast_horizon"],
        class_ratios=LABEL["class_ratios"],
        fit_dates=train_dates,
    )
    labels["date"] = pd.to_datetime(labels["date"])

    feature_cols = list(SELECTED_FEATURES)
    fm = FeatureManager()
    fm.feature_cols = feature_cols

    print(f"\n加载训练集因子 ({train_start}-{train_end})...")
    train_feat = load_fold_data(DATA_DIR, feature_cols, train_start, train_end)
    train_feat["year"] = train_feat["date"].dt.year
    FeatureManager.clean_inplace(train_feat, feature_cols)
    train_mask = train_feat["year"].between(train_start, train_end).values
    stats = fm.fit_standardizer(train_feat, mask=train_mask)
    del train_feat, train_mask
    gc.collect()
    print(f"  标准化统计量: {len(stats)} 个特征")

    print(f"\n加载验证集因子 ({val_start}-{val_end})...")
    val_feat = load_fold_data(DATA_DIR, feature_cols, val_start, val_end)
    val_feat["year"] = val_feat["date"].dt.year
    FeatureManager.clean_inplace(val_feat, feature_cols)
    val_feat = fm.apply_standardizer(val_feat, stats)
    gc.collect()

    T = FEATURES["lookback_window"]
    print(f"\n构建验证集数据存储 (lookback={T})...")
    store = TimingDataStore(val_feat, labels, T, feature_cols)
    del val_feat
    gc.collect()
    val_ds = TimingDataset(store, min_year=val_start, max_year=val_end)
    print(f"  验证集样本数: {len(val_ds):,}")

    model_path = os.path.join(CHECKPOINT_DIR, f"model_{model_year}.pt")
    if not os.path.exists(model_path):
        print(f"!! 模型文件不存在: {model_path}")
        return
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = AGRU(
        input_dim=len(feature_cols),
        hidden_dim=MODEL["hidden_dim"],
        num_gru_layers=MODEL["num_gru_layers"],
        num_classes=MODEL["num_classes"],
        dropout=MODEL["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  模型已加载: {model_path}")

    print(f"\n验证集推理...")
    loader = DataLoader(val_ds, batch_size=TRAIN_BATCH, shuffle=False, num_workers=0)
    sample_scores = []
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            probs = model.predict_proba(xb).numpy()
            scores = probs[:, 2] - probs[:, 0]
            sample_scores.append(scores)
            if (i + 1) % 50 == 0:
                print(f"    batch {i+1}...")
    sample_scores = np.concatenate(sample_scores)

    date_to_scores = {}
    for d, s in zip(val_ds.sample_dates, sample_scores):
        if d not in date_to_scores:
            date_to_scores[d] = []
        date_to_scores[d].append(float(s))
    daily_scores = [(d, np.mean(ss)) for d, ss in sorted(date_to_scores.items())]

    arr = np.array([s for _, s in daily_scores])
    r = LABEL["class_ratios"]
    t_low = float(np.quantile(arr, r[0]))
    t_high = float(np.quantile(arr, r[0] + r[1]))

    stats_path = os.path.join(CHECKPOINT_DIR, f"stats_{model_year}.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump({
            "standardizer": stats,
            "threshold": (t_low, t_high),
            "val_scores": daily_scores,
            "train_range": (train_start, train_end),
            "val_range": (val_start, val_end),
            "feature_cols": feature_cols,
        }, f)

    print(f"\n=== 完成 ===")
    print(f"  保存: {stats_path}")
    print(f"  阈值: t_low={t_low:.6f}, t_high={t_high:.6f}")
    print(f"  验证集天数: {len(daily_scores)}")
    print(f"  score 范围: [{arr.min():.6f}, {arr.max():.6f}]")


TRAIN_BATCH = 12000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取模型推理所需的统计量")
    parser.add_argument("--year", type=int, default=2026, help="模型年份 (默认 2026)")
    args = parser.parse_args()
    extract_stats(args.year)
