"""特征工程: 使用 DolphinDB 预计算因子 (alpha158 + cicc + custom, 已标准化)

内存优化: 所有操作原地修改, 不做 DataFrame.copy()
"""
import numpy as np
import pandas as pd


class FeatureManager:
    def __init__(self):
        self.feature_cols = None
        self.stats = None

    def setup(self, factor_df: pd.DataFrame):
        exclude = {"date", "code"}
        self.feature_cols = [c for c in factor_df.columns if c not in exclude]
        return self.feature_cols

    def fit_standardizer(self, df: pd.DataFrame, mask=None) -> dict:
        stats = {}
        for col in self.feature_cols:
            if col not in df.columns:
                continue
            if mask is not None:
                s = df.loc[mask, col]
            else:
                s = df[col]
            s = s.replace([np.inf, -np.inf], np.nan)
            stats[col] = {"mean": float(s.mean()), "std": float(s.std())}
        self.stats = stats
        return stats

    def apply_standardizer(self, df: pd.DataFrame, stats: dict = None) -> pd.DataFrame:
        st = stats or self.stats
        if st is None:
            return df
        for col in self.feature_cols:
            if col not in df.columns or col not in st:
                continue
            m = st[col]["mean"]
            s = st[col]["std"]
            if s > 0:
                df[col] = ((df[col] - m) / s).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            else:
                df[col] = (df[col] - m).fillna(0.0)
        return df

    @staticmethod
    def clean_inplace(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    @staticmethod
    def clean(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        return FeatureManager.clean_inplace(df, feature_cols)
