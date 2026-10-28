"""回测：2020-2025，开盘价成交，避免未来函数

信号时序:
  T日收盘 → 信号可用 → T+1开盘执行交易

收益计算:
  open-to-open: ret[T] = open[T+1] / open[T] - 1
  position[T] = position_map[signal[T-1]]  (信号后移1天)
  strategy_ret[T] = position[T] * ret[T] - commission * |position[T] - position[T-1]|

对比基准: 中证全指买入持有 (position=1.0)

用法:
  python run_backtest_2020_2025.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RESULT_DIR, BACKTEST

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
    n_trades = (returns.index.to_series().diff().dt.days > 7).sum() if hasattr(returns.index, 'dtype') and returns.index.dtype != 'int64' else 0
    return {
        "总收益率": total_ret,
        "年化收益率": ann_ret,
        "年化波动率": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "最大回撤开始日期": dd_start,
        "最大回撤结束日期": dd_end,
        "交易日数": len(returns),
    }


def main():
    print("=" * 70)
    print("回测: 2020-2025, 开盘价成交, 中证全指")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    idx = pd.read_parquet(os.path.join(DATA_DIR, "index_daily.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)

    sig = pd.read_csv(os.path.join(RESULT_DIR, "signals.csv"))
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.sort_values("date").reset_index(drop=True)

    print(f"  指数: {len(idx):,} 行, {idx['date'].min().date()} ~ {idx['date'].max().date()}")
    print(f"  信号: {len(sig):,} 行, {sig['date'].min().date()} ~ {sig['date'].max().date()}")

    # 2. 合并信号和指数，筛选回测区间
    print("\n[2] 合并数据，筛选区间...")
    bt_start = pd.Timestamp(BT_START)
    bt_end = pd.Timestamp(BT_END)
    idx_bt = idx[(idx["date"] >= bt_start) & (idx["date"] <= bt_end)].copy()
    sig_bt = sig[(sig["date"] >= bt_start) & (sig["date"] <= bt_end)].copy()

    df = pd.merge(sig_bt[["date", "signal", "score"]], idx_bt[["date", "open", "close"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  回测区间: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  交易日数: {len(df)}")

    # 3. 信号后移1天 → 次日开盘建仓
    print("\n[3] 信号对齐 (后移1天)...")
    df["position"] = df["signal"].shift(1).map(POSITION_MAP).fillna(0.0)

    # 4. open-to-open 收益
    #    ret[T] = open[T+1] / open[T] - 1  (从T开盘持有到T+1开盘)
    df["open_next"] = df["open"].shift(-1)
    df["ret"] = df["open_next"] / df["open"] - 1

    # 5. 换手 & 策略收益
    df["position_prev"] = df["position"].shift(1).fillna(0.0)
    df["turnover"] = (df["position"] - df["position_prev"]).abs()
    df["strategy_ret"] = df["position"] * df["ret"] - COMMISSION * df["turnover"]
    df["benchmark_ret"] = df["ret"]  # 买入持有

    # 丢弃最后一行（无 open_next）
    df = df.dropna(subset=["ret"]).reset_index(drop=True)

    # 6. 计算指标
    print("\n[4] 计算指标...")
    strat_metrics = calc_metrics(df.set_index("date")["strategy_ret"])
    bench_metrics = calc_metrics(df.set_index("date")["benchmark_ret"])

    # 择时IC: score vs 次日收益 (close-to-close, 匹配模型训练口径)
    idx_all = idx[["date", "close"]].copy()
    idx_all["future_ret"] = idx_all["close"].shift(-1) / idx_all["close"] - 1
    ic_df = pd.merge(df[["date", "score"]], idx_all[["date", "future_ret"]], on="date", how="inner").dropna()
    if len(ic_df) >= 2:
        ic, _ = spearmanr(ic_df["score"], ic_df["future_ret"])
    else:
        ic = 0.0
    strat_metrics["择时IC"] = ic

    # 7. 信号准确性
    print("\n[5] 信号准确性...")
    # 信号预测的是次日收益 (close[T] → close[T+1])
    sig_acc_df = pd.merge(sig_bt[["date", "signal"]], idx_all[["date", "future_ret"]], on="date", how="inner").dropna()
    sig_acc_rows = []
    for s in sorted(sig_acc_df["signal"].unique()):
        sub = sig_acc_df[sig_acc_df["signal"] == s]
        sig_acc_rows.append({
            "信号": int(s),
            "出现天数": len(sub),
            "出现比例": len(sub) / len(sig_acc_df),
            "次日收益率均值": sub["future_ret"].mean(),
            "胜率": (sub["future_ret"] > 0).mean(),
        })
    sig_acc = pd.DataFrame(sig_acc_rows).set_index("信号")

    # 8. 输出结果
    print("\n" + "=" * 70)
    print("策略表现 (2020-2025, 开盘价成交, 单边万五)")
    print("=" * 70)

    print(f"\n{'指标':<16} {'策略':>12} {'基准(买入持有)':>14} {'差值':>12}")
    print("-" * 56)
    for key in ["总收益率", "年化收益率", "年化波动率", "夏普比率", "最大回撤"]:
        sv = strat_metrics[key]
        bv = bench_metrics[key]
        diff = sv - bv
        if key in ("总收益率", "年化收益率", "年化波动率", "最大回撤"):
            print(f"{key:<16} {sv:>11.2%} {bv:>13.2%} {diff:>11.2%}")
        else:
            print(f"{key:<16} {sv:>12.4f} {bv:>14.4f} {diff:>12.4f}")

    print(f"\n{'最大回撤区间':<16} {strat_metrics['最大回撤开始日期']} ~ {strat_metrics['最大回撤结束日期']}")
    print(f"{'基准回撤区间':<16} {bench_metrics['最大回撤开始日期']} ~ {bench_metrics['最大回撤结束日期']}")
    print(f"{'择时IC':<16} {ic:.4f}  (研报目标 0.1581)")
    print(f"{'交易日数':<16} {strat_metrics['交易日数']}")

    # 信号换手统计
    total_turnover = df["turnover"].sum()
    avg_turnover = total_turnover / len(df)
    print(f"{'总换手':<16} {total_turnover:.2f}")
    print(f"{'日均换手':<16} {avg_turnover:.4f}")

    print("\n" + "=" * 70)
    print("信号准确性 (信号 vs 次日close-to-close收益)")
    print("=" * 70)
    print(sig_acc.to_string())

    # 信号 vs 实际持仓收益
    print("\n" + "=" * 70)
    print("信号 vs 实际持仓收益 (open-to-open)")
    print("=" * 70)
    hold_acc_rows = []
    df_sig = df.dropna(subset=["signal"]).copy()
    # 信号在T日, 实际持仓在T+1, 收益为ret[T+1]=open[T+2]/open[T+1]-1
    df_sig["next_ret"] = df_sig["ret"].shift(-1)
    df_sig_hold = df_sig.dropna(subset=["next_ret"])
    for s in sorted(df_sig_hold["signal"].unique()):
        sub = df_sig_hold[df_sig_hold["signal"] == s]
        hold_acc_rows.append({
            "信号": int(s),
            "天数": len(sub),
            "实际持仓收益均值": sub["next_ret"].mean(),
            "胜率": (sub["next_ret"] > 0).mean(),
        })
    hold_acc = pd.DataFrame(hold_acc_rows).set_index("信号")
    print(hold_acc.to_string())

    # 9. 保存逐日结果
    daily_out = df[["date", "signal", "score", "position", "open", "open_next",
                     "ret", "benchmark_ret", "turnover", "strategy_ret"]].copy()
    daily_out["strategy_cum"] = (1 + daily_out["strategy_ret"]).cumprod()
    daily_out["benchmark_cum"] = (1 + daily_out["benchmark_ret"]).cumprod()
    daily_path = os.path.join(RESULT_DIR, "backtest_2020_2025.csv")
    daily_out.to_csv(daily_path, index=False)
    print(f"\n逐日结果已保存: {daily_path}")

    # 10. 逐年表现
    print("\n" + "=" * 70)
    print("逐年表现")
    print("=" * 70)
    df_yr = df.set_index("date")
    print(f"\n{'年份':<8} {'策略年化':>10} {'基准年化':>10} {'策略夏普':>10} {'基准夏普':>10} {'最大回撤':>10}")
    print("-" * 60)
    for yr in sorted(df_yr.index.year.unique()):
        sub = df_yr[df_yr.index.year == yr]
        sm = calc_metrics(sub["strategy_ret"])
        bm = calc_metrics(sub["benchmark_ret"])
        print(f"{yr:<8} {sm['年化收益率']:>9.2%} {bm['年化收益率']:>9.2%} "
              f"{sm['夏普比率']:>10.4f} {bm['夏普比率']:>10.4f} {sm['最大回撤']:>9.2%}")


if __name__ == "__main__":
    main()
