"""评估指标: TSRankIC, 信号准确性, 回测指标"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def tsrankic(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 2:
        return 0.0
    rho, _ = spearmanr(pred, true)
    return float(rho) if not np.isnan(rho) else 0.0


def evaluate_signals(signals: pd.Series, returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"signal": signals, "ret": returns}).dropna()
    rows = []
    for sig in sorted(df["signal"].unique()):
        sub = df[df["signal"] == sig]
        rows.append({
            "signal": int(sig),
            "出现比例": len(sub) / len(df),
            "收益率均值": sub["ret"].mean(),
            "收益率为正比例": (sub["ret"] > 0).mean(),
        })
    return pd.DataFrame(rows).set_index("signal")


def backtest_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()
    dd_end = drawdown.idxmin()
    dd_start = cum.loc[:dd_end].idxmax() if dd_end is not None else None
    return {
        "总收益率": total_ret,
        "年化收益率": ann_ret,
        "年化波动率": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "最大回撤开始日期": dd_start,
        "最大回撤结束日期": dd_end,
    }
