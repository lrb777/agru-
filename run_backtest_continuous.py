"""回测对比：三档离散仓位 vs 连续仓位，2020-2025，开盘价成交

信号时序:
  T日收盘 → 信号可用 → T+1开盘执行交易

收益计算:
  open-to-open: ret[T] = open[T+1] / open[T] - 1
  信号后移1天对齐

仓位规则:
  离散: position = {0: 0.0, 1: 0.5, 2: 1.0}
  连续: position = clip((score - t_low) / (t_high - t_low), 0, 1)

用法:
  python run_backtest_continuous.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RESULT_DIR, CHECKPOINT_DIR, BACKTEST

BT_START = "2020-01-01"
BT_END = "2025-12-31"
COMMISSION = BACKTEST["commission_rate"]
POSITION_MAP = BACKTEST["position_map"]
PERIODS_PER_YEAR = 252


def calc_metrics(returns: pd.Series, periods_per_year=PERIODS_PER_YEAR):
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
        "最大回撤区间": f"{dd_start} ~ {dd_end}" if dd_start else "N/A",
        "交易日数": len(returns),
    }


def run_backtest(df, position_col, label):
    """对指定仓位列做回测，返回指标 + 逐日结果"""
    d = df.copy()
    d["position_prev"] = d[position_col].shift(1).fillna(0.0)
    d["turnover"] = (d[position_col] - d["position_prev"]).abs()
    d["strategy_ret"] = d[position_col] * d["ret"] - COMMISSION * d["turnover"]
    d = d.dropna(subset=["ret"]).reset_index(drop=True)

    metrics = calc_metrics(d.set_index("date")["strategy_ret"])

    # IC
    idx_all = d[["date", "score"]].copy()
    idx_all["future_ret"] = d["ret"].shift(-1)
    aligned = idx_all.dropna(subset=["future_ret"])
    if len(aligned) >= 2:
        ic, _ = spearmanr(aligned["score"], aligned["future_ret"])
    else:
        ic = 0.0
    metrics["择时IC"] = ic

    d["strategy_cum"] = (1 + d["strategy_ret"]).cumprod()
    d["benchmark_cum"] = (1 + d["benchmark_ret"]).cumprod()

    return metrics, d


def main():
    print("=" * 70)
    print("回测对比: 离散仓位 vs 连续仓位 (2020-2025)")
    print("=" * 70)

    # 加载数据
    idx = pd.read_parquet(os.path.join(DATA_DIR, "index_daily.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)

    sig = pd.read_csv(os.path.join(RESULT_DIR, "signals.csv"))
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.sort_values("date").reset_index(drop=True)

    # 阈值
    with open(os.path.join(CHECKPOINT_DIR, "stats_2026.pkl"), "rb") as f:
        stats = pickle.load(f)
    t_low, t_high = stats["threshold"]

    # 合并
    bt_start = pd.Timestamp(BT_START)
    bt_end = pd.Timestamp(BT_END)
    idx_bt = idx[(idx["date"] >= bt_start) & (idx["date"] <= bt_end)].copy()
    sig_bt = sig[(sig["date"] >= bt_start) & (sig["date"] <= bt_end)].copy()
    df = pd.merge(sig_bt[["date", "signal", "score"]], idx_bt[["date", "open", "close"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    # 收益
    df["open_next"] = df["open"].shift(-1)
    df["ret"] = df["open_next"] / df["open"] - 1
    df["benchmark_ret"] = df["ret"]

    # 仓位
    # 离散: 信号后移1天 → map
    df["pos_discrete"] = df["signal"].shift(1).map(POSITION_MAP).fillna(0.0)

    # 连续: score后移1天 → 线性映射
    df["score_shifted"] = df["score"].shift(1)
    df["pos_continuous"] = np.clip(
        (df["score_shifted"] - t_low) / (t_high - t_low), 0, 1
    ).fillna(0.0)

    print(f"\n  回测区间: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  交易日数: {len(df)}")
    print(f"  阈值: t_low={t_low:.4f}, t_high={t_high:.4f}")
    print(f"\n  离散仓位分布:")
    print(f"    均值={df['pos_discrete'].mean():.4f}  std={df['pos_discrete'].std():.4f}")
    print(f"  连续仓位分布:")
    print(f"    均值={df['pos_continuous'].mean():.4f}  std={df['pos_continuous'].std():.4f}")
    print(f"    0%天数: {(df['pos_continuous'] == 0).sum()}  100%天数: {(df['pos_continuous'] >= 0.99).sum()}")

    # 回测
    m_disc, d_disc = run_backtest(df, "pos_discrete", "离散")
    m_cont, d_cont = run_backtest(df, "pos_continuous", "连续")
    m_bench = calc_metrics(df.set_index("date")["benchmark_ret"].dropna())

    # 对比表
    print("\n" + "=" * 70)
    print("策略对比 (2020-2025, 开盘价成交, 单边万五)")
    print("=" * 70)

    print(f"\n{'指标':<16} {'离散仓位':>12} {'连续仓位':>12} {'基准(持有)':>12} {'连续-离散':>10}")
    print("-" * 66)
    for key in ["总收益率", "年化收益率", "年化波动率", "夏普比率", "最大回撤"]:
        dv = m_disc[key]
        cv = m_cont[key]
        bv = m_bench[key]
        diff = cv - dv
        if key in ("总收益率", "年化收益率", "年化波动率", "最大回撤"):
            print(f"{key:<16} {dv:>11.2%} {cv:>11.2%} {bv:>11.2%} {diff:>+9.2%}")
        else:
            print(f"{key:<16} {dv:>12.4f} {cv:>12.4f} {bv:>12.4f} {diff:>+10.4f}")

    print(f"\n{'最大回撤区间(离散)':<18} {m_disc['最大回撤区间']}")
    print(f"{'最大回撤区间(连续)':<18} {m_cont['最大回撤区间']}")
    print(f"{'择时IC(离散)':<18} {m_disc['择时IC']:.4f}")
    print(f"{'择时IC(连续)':<18} {m_cont['择时IC']:.4f}")

    # 换手
    df_disc = d_disc
    df_cont = d_cont
    print(f"\n{'总换手(离散)':<18} {df_disc['turnover'].sum():.2f}  日均{df_disc['turnover'].mean():.4f}")
    print(f"{'总换手(连续)':<18} {df_cont['turnover'].sum():.2f}  日均{df_cont['turnover'].mean():.4f}")

    # 逐年对比
    print("\n" + "=" * 70)
    print("逐年对比")
    print("=" * 70)
    print(f"\n{'年份':<6} {'离散年化':>10} {'连续年化':>10} {'基准年化':>10} {'离散夏普':>10} {'连续夏普':>10} {'基准夏普':>10}")
    print("-" * 68)
    for yr in sorted(d_disc["date"].dt.year.unique()):
        s_d = d_disc[d_disc["date"].dt.year == yr]["strategy_ret"]
        s_c = d_cont[d_cont["date"].dt.year == yr]["strategy_ret"]
        s_b = d_disc[d_disc["date"].dt.year == yr]["benchmark_ret"]
        md = calc_metrics(s_d)
        mc = calc_metrics(s_c)
        mb = calc_metrics(s_b)
        print(f"{yr:<6} {md['年化收益率']:>9.2%} {mc['年化收益率']:>9.2%} {mb['年化收益率']:>9.2%} "
              f"{md['夏普比率']:>10.4f} {mc['夏普比率']:>10.4f} {mb['夏普比率']:>10.4f}")

    # 仓位分布对比
    print("\n" + "=" * 70)
    print("仓位分布对比")
    print("=" * 70)
    bins = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.01]
    labels = ["0%", "1-25%", "25-50%", "50-75%", "75-99%", "100%"]
    df_pos = df.dropna(subset=["pos_discrete"]).copy()
    df_pos["离散档"] = pd.cut(df_pos["pos_discrete"], bins=bins, labels=labels, include_lowest=True)
    df_pos["连续档"] = pd.cut(df_pos["pos_continuous"], bins=bins, labels=labels, include_lowest=True)
    print(f"\n  {'仓位区间':<10} {'离散天数':>8} {'连续天数':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8}")
    for label in labels:
        d_n = (df_pos["离散档"] == label).sum()
        c_n = (df_pos["连续档"] == label).sum()
        print(f"  {label:<10} {d_n:>8} {c_n:>8}")

    # 保存逐日结果
    daily_out = d_cont[["date", "signal", "score", "pos_continuous", "pos_discrete",
                         "open", "ret", "benchmark_ret", "turnover", "strategy_ret",
                         "strategy_cum", "benchmark_cum"]].copy()
    daily_path = os.path.join(RESULT_DIR, "backtest_continuous_2020_2025.csv")
    daily_out.to_csv(daily_path, index=False)
    print(f"\n逐日结果已保存: {daily_path}")


if __name__ == "__main__":
    main()
