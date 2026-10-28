"""信号准确率验证：离散 vs 连续仓位，基于测试集分布定阈值

标准:
  仓位×涨跌 > +0.10%  → 正确（有实质正收益，扣费后仍赚）
  仓位×涨跌 < -0.10%  → 错误（有实质亏损）
  介于 ±0.10% 之间    → 无效（噪声区间，不计）

阈值依据: 测试集(2020-2025) |ret| 的10%分位 = 0.10%，等于双边手续费

时序: T-1日收盘数据 → 预测T日 → 用T日实际涨跌(close_ret)验证

用法:
  python verify_accuracy.py                    # 全部预测记录
  python verify_accuracy.py --start 2026-09-01 # 指定起始日期
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import dolphindb as ddb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR, CHECKPOINT_DIR, BACKTEST, DOLPHINDB, INDEX_CODE

THRESHOLD = 0.001  # 0.10%


def main():
    parser = argparse.ArgumentParser(description="信号准确率验证")
    parser.add_argument("--start", type=str, default=None, help="起始日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    # 加载阈值
    with open(os.path.join(CHECKPOINT_DIR, "stats_2026.pkl"), "rb") as f:
        stats = pickle.load(f)
    t_low, t_high = stats["threshold"]
    print(f"阈值: t_low={t_low:.4f}, t_high={t_high:.4f}")
    print(f"准确率标准: 仓位×涨跌 > +{THRESHOLD*100:.2f}% = 正确, < -{THRESHOLD*100:.2f}% = 错误, 中间 = 无效")

    # 加载预测结果
    dp = pd.read_csv(os.path.join(RESULT_DIR, "daily_predict.csv"))
    dp["date"] = pd.to_datetime(dp["date"])
    dp = dp.sort_values("date").reset_index(drop=True)

    if args.start:
        dp = dp[dp["date"] >= args.start].reset_index(drop=True)

    if len(dp) == 0:
        print("无预测记录")
        return

    # 拉取指数行情
    min_date = dp["date"].min() - pd.Timedelta(days=5)
    max_date = dp["date"].max() + pd.Timedelta(days=3)

    session = ddb.session()
    session.connect(DOLPHINDB["host"], DOLPHINDB["port"], DOLPHINDB["username"], DOLPHINDB["password"])
    script = (
        't = loadTable("dfs://d0_day", "jy_qt_indexquote")\n'
        's = loadTable("dfs://d0_day", "jy_secumain")\n'
        'select s.SecuCode as code, t.TradingDay as date, t.ClosePrice as close\n'
        'from t join s on t.InnerCode = s.InnerCode\n'
        f'where s.SecuCode = "{INDEX_CODE}"\n'
        f'  and t.TradingDay between {min_date.strftime("%Y.%m.%d")} and {max_date.strftime("%Y.%m.%d")}\n'
        'order by t.TradingDay'
    )
    idx = session.run(script)
    session.close()
    idx["date"] = pd.to_datetime(idx["date"])
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["close_ret"] = idx["close"].pct_change()

    merged = pd.merge(dp[["date", "signal", "score"]], idx[["date", "close_ret"]], on="date", how="left")
    merged = merged.dropna(subset=["close_ret"]).reset_index(drop=True)

    # 计算仓位
    pos_map = BACKTEST["position_map"]
    merged["pos_discrete"] = merged["signal"].map(pos_map)
    merged["pos_continuous"] = np.clip((merged["score"] - t_low) / (t_high - t_low), 0, 1)

    # 计算收益贡献
    merged["disc_contrib"] = merged["pos_discrete"] * merged["close_ret"]
    merged["cont_contrib"] = merged["pos_continuous"] * merged["close_ret"]

    # 判定
    # 有仓位时: 仓位×涨跌 > +0.10% = 正确, < -0.10% = 错误, 中间 = 无效
    # 空仓时: 当日跌 > 0.10% = 正确(避损), 当日涨 > 0.10% = 错误(踏空), 中间 = 无效
    def judge(pos, ret):
        if pos == 0:
            if ret < -THRESHOLD:
                return "正确"
            elif ret > THRESHOLD:
                return "错误"
            else:
                return "无效"
        else:
            contrib = pos * ret
            if contrib > THRESHOLD:
                return "正确"
            elif contrib < -THRESHOLD:
                return "错误"
            else:
                return "无效"

    merged["disc_verdict"] = merged.apply(lambda r: judge(r["pos_discrete"], r["close_ret"]), axis=1)
    merged["cont_verdict"] = merged.apply(lambda r: judge(r["pos_continuous"], r["close_ret"]), axis=1)

    # 逐日输出
    sig_labels = {0: "空仓", 1: "半仓", 2: "满仓"}
    print(f"\n{'日期':<12} {'信号':<8} {'score':>10} {'离散仓位':>8} {'连续仓位':>8} {'当日涨跌':>10} {'离散收益':>10} {'连续收益':>10} {'离散':<6} {'连续':<6}")
    print("-" * 110)

    for _, r in merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        sig = int(r["signal"])
        ret = r["close_ret"]
        pd_ = r["pos_discrete"]
        pc = r["pos_continuous"]
        dc = r["disc_contrib"]
        cc = r["cont_contrib"]
        dv = r["disc_verdict"]
        cv = r["cont_verdict"]
        print(f"{d:<12} {sig} {sig_labels[sig]:<5} {r['score']:>10.6f} {pd_:>7.0%} {pc:>7.1%} {ret:>+9.2%} {dc:>+9.2%} {cc:>+9.2%} {dv:<6} {cv:<6}")

    # 汇总
    n = len(merged)
    print(f"\n{'='*70}")
    print(f"汇总 ({merged['date'].min().date()} ~ {merged['date'].max().date()}, {n} 天)")
    print(f"{'='*70}")

    for label, verdict_col, contrib_col in [("离散仓位", "disc_verdict", "disc_contrib"), ("连续仓位", "cont_verdict", "cont_contrib")]:
        valid = merged[merged[verdict_col].isin(["正确", "错误"])]
        invalid = merged[merged[verdict_col] == "无效"]
        n_correct = (merged[verdict_col] == "正确").sum()
        n_wrong = (merged[verdict_col] == "错误").sum()
        n_invalid = (merged[verdict_col] == "无效").sum()
        n_valid = n_correct + n_wrong

        print(f"\n  {label}:")
        print(f"    正确: {n_correct}  错误: {n_wrong}  无效: {n_invalid}")
        if n_valid > 0:
            print(f"    准确率: {n_correct}/{n_valid} = {n_correct/n_valid*100:.1f}%")
        print(f"    累计收益贡献: {merged[contrib_col].sum():+.2%}")
        print(f"    有效样本占比: {n_valid}/{n} = {n_valid/n*100:.1f}%")

    # 按信号分组
    print(f"\n{'='*70}")
    print(f"按信号分组")
    print(f"{'='*70}")

    for sig in [0, 1, 2]:
        sub = merged[merged["signal"] == sig]
        if len(sub) == 0:
            continue
        label = sig_labels[sig]
        n_sub = len(sub)

        print(f"\n  {sig} {label} ({n_sub} 天):")

        for vlabel, vcol, ccol in [("离散", "disc_verdict", "disc_contrib"), ("连续", "cont_verdict", "cont_contrib")]:
            valid = sub[sub[vcol].isin(["正确", "错误"])]
            n_c = (sub[vcol] == "正确").sum()
            n_w = (sub[vcol] == "错误").sum()
            n_i = (sub[vcol] == "无效").sum()
            n_v = n_c + n_w
            if n_v > 0:
                print(f"    {vlabel}: 正确{n_c} 错误{n_w} 无效{n_i}  准确率={n_c/n_v*100:.1f}%  收益={sub[ccol].sum():+.2%}")
            else:
                print(f"    {vlabel}: 正确{n_c} 错误{n_w} 无效{n_i}  准确率=N/A  收益={sub[ccol].sum():+.2%}")

    # 上涨/下跌日对比
    print(f"\n{'='*70}")
    print(f"上涨/下跌日对比")
    print(f"{'='*70}")

    up = merged[merged["close_ret"] > 0]
    dn = merged[merged["close_ret"] < 0]

    print(f"\n  上涨日 ({len(up)} 天):")
    print(f"    离散: 仓位均值={up['pos_discrete'].mean():.1%}  收益={up['disc_contrib'].sum():+.2%}")
    print(f"    连续: 仓位均值={up['pos_continuous'].mean():.1%}  收益={up['cont_contrib'].sum():+.2%}")

    print(f"\n  下跌日 ({len(dn)} 天):")
    print(f"    离散: 仓位均值={dn['pos_discrete'].mean():.1%}  收益={dn['disc_contrib'].sum():+.2%}")
    print(f"    连续: 仓位均值={dn['pos_continuous'].mean():.1%}  收益={dn['cont_contrib'].sum():+.2%}")

    print(f"\n  累计: 离散={merged['disc_contrib'].sum():+.2%}  连续={merged['cont_contrib'].sum():+.2%}  基准={merged['close_ret'].sum():+.2%}")


if __name__ == "__main__":
    main()
