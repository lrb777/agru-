"""回测系统: 日频调仓, 开盘价成交, 单边万五手续费"""
import numpy as np
import pandas as pd
from config import BACKTEST
from .metrics import backtest_metrics, evaluate_signals, tsrankic


class Backtester:
    def __init__(self, index_df: pd.DataFrame):
        self.index_df = index_df.copy()
        self.index_df["date"] = pd.to_datetime(self.index_df["date"])
        self.index_df = self.index_df.sort_values("date").reset_index(drop=True)
        self.position_map = BACKTEST["position_map"]
        self.commission = BACKTEST["commission_rate"]

    def run(self, signals: pd.DataFrame) -> dict:
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"])
        sig = sig.sort_values("date").reset_index(drop=True)

        idx = self.index_df[["date", "open", "close"]].copy()
        merged = pd.merge_asof(
            sig[["date", "signal"]],
            idx,
            on="date",
            direction="nearest",
        )

        merged["ret"] = merged["close"].pct_change()
        merged["position"] = merged["signal"].map(self.position_map).fillna(0.0)
        merged["position_prev"] = merged["position"].shift(1).fillna(0.0)
        merged["turnover"] = (merged["position"] - merged["position_prev"]).abs()
        merged["strategy_ret"] = (
            merged["position"] * merged["ret"] - self.commission * merged["turnover"]
        )
        merged = merged.dropna(subset=["ret"])

        metrics = backtest_metrics(merged.set_index("date")["strategy_ret"])

        sig_eval = merged.set_index("date")["signal"]
        ret_eval = merged.set_index("date")["ret"]
        signal_acc = evaluate_signals(sig_eval, ret_eval)

        pred_score = sig[["date", "score"]].set_index("date")["score"]
        actual_ret = merged.set_index("date")["ret"]
        aligned = pd.concat([pred_score, actual_ret], axis=1, join="inner").dropna()
        if len(aligned) >= 2:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            ic = float(ic) if not np.isnan(ic) else 0.0
        else:
            ic = 0.0
        metrics["择时IC"] = ic

        return {
            "metrics": metrics,
            "signal_accuracy": signal_acc,
            "daily": merged,
        }

    def run_alt_price(self, signals: pd.DataFrame, price_col: str = "open") -> dict:
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"])
        idx = self.index_df[["date", price_col, "close"]].copy()
        idx["date"] = pd.to_datetime(idx["date"])

        merged = pd.merge_asof(
            sig[["date", "signal"]],
            idx,
            on="date",
            direction="nearest",
        )
        merged["ret"] = merged[price_col].pct_change()
        merged["position"] = merged["signal"].map(self.position_map).fillna(0.0)
        merged["position_prev"] = merged["position"].shift(1).fillna(0.0)
        merged["turnover"] = (merged["position"] - merged["position_prev"]).abs()
        merged["strategy_ret"] = (
            merged["position"] * merged["ret"] - self.commission * merged["turnover"]
        )
        merged = merged.dropna(subset=["ret"])
        return backtest_metrics(merged.set_index("date")["strategy_ret"])
