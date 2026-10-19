"""数据集构建: 个股样本 + 市场标签

样本: (股票 i, 日期 t) -> 过去 T 日特征序列 (T, F)
标签: 同一交易日所有股票共享市场标签 (中证全指未来1日收益三分类)

内存优化: TimingDataStore 持有特征数据的一份拷贝,
         TimingDataset 仅保存样本索引, train/val/test 共享同一 store。
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import LABEL


def make_market_labels(
    index_df: pd.DataFrame,
    forecast_horizon: int = None,
    class_ratios: list = None,
    fit_dates: pd.Series = None,
) -> pd.DataFrame:
    fh = forecast_horizon or LABEL["forecast_horizon"]
    ratios = class_ratios or LABEL["class_ratios"]

    df = index_df[["date", "close"]].sort_values("date").copy()
    df["future_ret"] = df["close"].shift(-fh) / df["close"] - 1
    df = df.dropna(subset=["future_ret"])

    q1 = ratios[0]
    q2 = ratios[0] + ratios[1]

    if fit_dates is not None:
        fit_mask = df["date"].isin(fit_dates)
        fit_rets = df.loc[fit_mask, "future_ret"]
        t1 = fit_rets.quantile(q1)
        t2 = fit_rets.quantile(q2)
    else:
        t1 = df["future_ret"].quantile(q1)
        t2 = df["future_ret"].quantile(q2)

    def classify(r):
        if r < t1:
            return 0
        elif r < t2:
            return 1
        else:
            return 2

    df["label"] = df["future_ret"].apply(classify)
    return df[["date", "future_ret", "label"]]


class TimingDataStore:
    """特征数据存储, 一次构建, train/val/test 共享引用, 避免内存重复"""

    def __init__(self, factor_df: pd.DataFrame, labels: pd.DataFrame,
                 lookback: int, feature_cols: list):
        self.lookback = lookback
        self.feature_cols = feature_cols
        self.label_map = labels.set_index("date")["label"].to_dict()

        self.code_data = {}
        df = factor_df.sort_values(["code", "date"])
        n_codes = df["code"].nunique()
        processed = 0
        for code, g in df.groupby("code", sort=False):
            g = g.sort_values("date")
            dates = g["date"].values
            feats = g[feature_cols].values.astype(np.float32)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            self.code_data[code] = (dates, feats)
            processed += 1
            if processed % 1000 == 0:
                print(f"    已处理 {processed}/{n_codes} 只股票...", flush=True)

    def n_stocks(self):
        return len(self.code_data)


class TimingDataset(Dataset):
    """TimingDataStore 的年份过滤视图, 仅保存样本索引, 不拷贝特征数据"""

    def __init__(self, store: TimingDataStore, min_year: int = None, max_year: int = None):
        self.store = store
        self.samples = []
        self.sample_dates = []

        for code, (dates, _) in store.code_data.items():
            if len(dates) < store.lookback:
                continue
            years = pd.DatetimeIndex(dates).year
            for i in range(store.lookback - 1, len(dates)):
                d = dates[i]
                if d not in store.label_map:
                    continue
                y = years[i]
                if min_year is not None and y < min_year:
                    continue
                if max_year is not None and y > max_year:
                    continue
                self.samples.append((code, i))
                self.sample_dates.append(d)

    @property
    def label_map(self):
        return self.store.label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, i = self.samples[idx]
        feats = self.store.code_data[code][1]
        window = feats[i - self.store.lookback + 1 : i + 1]
        label = self.store.label_map[self.sample_dates[idx]]
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.long)
