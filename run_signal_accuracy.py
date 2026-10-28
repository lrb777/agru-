"""逐日信号准确率统计：按信号分组，统计次日涨跌方向是否正确

用法:
  python run_signal_accuracy.py                  # 全部信号 (2020-2026)
  python run_signal_accuracy.py --start 2026-09  # 指定起始月份
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RESULT_DIR, DOLPHINDB, INDEX_CODE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="起始日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    # 加载信号
    sig = pd.read_csv(os.path.join(RESULT_DIR, "signals.csv"))
    sig["date"] = pd.to_datetime(sig["date"])

    # 加载指数行情
    idx = pd.read_parquet(os.path.join(DATA_DIR, "index_daily.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["future_ret"] = idx["close"].shift(-1) / idx["close"] - 1

    merged = pd.merge(sig[["date", "signal", "score"]], idx[["date", "future_ret"]], on="date", how="inner").dropna()

    if args.start:
        merged = merged[merged["date"] >= args.start].reset_index(drop=True)

    sig_labels = {0: "空仓", 1: "半仓", 2: "满仓"}

    # 按月统计
    merged["year_month"] = merged["date"].dt.to_period("M")

    print(f"{'='*90}")
    print(f"逐月信号准确率 ({merged['date'].min().date()} ~ {merged['date'].max().date()})")
    print(f"{'='*90}")
    print(f"\n{'月份':<10} {'天数':>5} {'空仓天数':>7} {'半仓天数':>7} {'满仓天数':>7}"
          f"  {'空仓准确率':>10} {'半仓准确率':>10} {'满仓准确率':>10}  {'整体准确率':>10}")
    print("-" * 90)

    for ym in sorted(merged["year_month"].unique()):
        sub = merged[merged["year_month"] == ym]
        n = len(sub)
        n0 = (sub["signal"] == 0).sum()
        n1 = (sub["signal"] == 1).sum()
        n2 = (sub["signal"] == 2).sum()

        # 空仓准确率: 次日跌=正确
        if n0 > 0:
            s0 = sub[sub["signal"] == 0]
            acc0 = (s0["future_ret"] < 0).mean()
            acc0_str = f"{acc0:.1%}({n0})"
        else:
            acc0_str = "-"

        # 半仓准确率: 次日涨=正确 (半仓偏多)
        if n1 > 0:
            s1 = sub[sub["signal"] == 1]
            acc1 = (s1["future_ret"] >= 0).mean()
            acc1_str = f"{acc1:.1%}({n1})"
        else:
            acc1_str = "-"

        # 满仓准确率: 次日涨=正确
        if n2 > 0:
            s2 = sub[sub["signal"] == 2]
            acc2 = (s2["future_ret"] > 0).mean()
            acc2_str = f"{acc2:.1%}({n2})"
        else:
            acc2_str = "-"

        # 整体: 空仓+跌, 半仓+涨, 满仓+涨
        correct = 0
        total = 0
        for _, r in sub.iterrows():
            if r["signal"] == 0 and r["future_ret"] < 0:
                correct += 1
                total += 1
            elif r["signal"] == 0 and r["future_ret"] >= 0:
                total += 1
            elif r["signal"] == 1 and r["future_ret"] >= 0:
                correct += 1
                total += 1
            elif r["signal"] == 1 and r["future_ret"] < 0:
                total += 1
            elif r["signal"] == 2 and r["future_ret"] > 0:
                correct += 1
                total += 1
            elif r["signal"] == 2 and r["future_ret"] <= 0:
                total += 1

        overall = f"{correct/total:.1%}" if total > 0 else "-"

        print(f"{str(ym):<10} {n:>5} {n0:>7} {n1:>7} {n2:>7}  {acc0_str:>10} {acc1_str:>10} {acc2_str:>10}  {overall:>10}")

    # 汇总
    print(f"\n{'='*90}")
    print(f"整体汇总")
    print(f"{'='*90}")

    n = len(merged)
    n0 = (merged["signal"] == 0).sum()
    n1 = (merged["signal"] == 1).sum()
    n2 = (merged["signal"] == 2).sum()

    for s in [0, 1, 2]:
        sub = merged[merged["signal"] == s]
        if s == 0:
            correct = (sub["future_ret"] < 0).sum()
        elif s == 2:
            correct = (sub["future_ret"] > 0).sum()
        else:
            correct = (sub["future_ret"] >= 0).sum()
        print(f"  {s} {sig_labels[s]}: {correct}/{len(sub)} = {correct/len(sub):.1%}  "
              f"次日均值={sub['future_ret'].mean()*100:+.4f}%  胜率={((sub['future_ret']>0).mean())*100:.1f}%")

    # 按score五等分看半仓信号
    print(f"\n{'='*90}")
    print(f"半仓信号(1)按score五等分的准确率")
    print(f"{'='*90}")
    half = merged[merged["signal"] == 1].copy()
    if len(half) >= 5:
        half["quintile"] = pd.qcut(half["score"], 5, labels=["Q1最低", "Q2", "Q3", "Q4", "Q5最高"])
        print(f"\n{'分组':<10} {'天数':>5} {'score范围':>22} {'次日均值':>10} {'胜率':>8} {'准确率(涨)':>10}")
        print("-" * 70)
        for q in ["Q1最低", "Q2", "Q3", "Q4", "Q5最高"]:
            sub = half[half["quintile"] == q]
            acc = (sub["future_ret"] >= 0).mean()
            print(f"{q:<10} {len(sub):>5} [{sub['score'].min():.4f}, {sub['score'].max():.4f}]"
                  f"  {sub['future_ret'].mean()*100:>+9.4f}% {(sub['future_ret']>0).mean()*100:>7.1f}% {acc*100:>9.1f}%")


if __name__ == "__main__":
    main()
