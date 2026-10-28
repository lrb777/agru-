"""对比验证：离散仓位 vs 连续仓位，同一时间段

流程:
  T-1日收盘数据 → 预测T日信号 → 获取T日实际涨跌(close_ret) → 验证

准确率定义:
  离散: 空仓+跌=正确, 满仓+涨=正确, 半仓=中性
  连续: 仓位<50%+跌=正确(减仓避险), 仓位>50%+涨=正确(持仓获利)
        仓位=0+跌=正确, 仓位>0+涨=正确
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import dolphindb as ddb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR, CHECKPOINT_DIR, BACKTEST, DOLPHINDB, INDEX_CODE


def main():
    # 加载阈值
    with open(os.path.join(CHECKPOINT_DIR, "stats_2026.pkl"), "rb") as f:
        stats = pickle.load(f)
    t_low, t_high = stats["threshold"]
    print(f"阈值: t_low={t_low:.4f}, t_high={t_high:.4f}")

    # 加载预测结果
    dp = pd.read_csv(os.path.join(RESULT_DIR, "daily_predict.csv"))
    dp["date"] = pd.to_datetime(dp["date"])
    dp = dp.sort_values("date").reset_index(drop=True)

    # 拉取指数行情 (需要预测日前一天的close来算close_ret)
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

    # 离散仓位
    pos_map = BACKTEST["position_map"]
    merged["pos_discrete"] = merged["signal"].map(pos_map)

    # 连续仓位
    merged["pos_continuous"] = np.clip(
        (merged["score"] - t_low) / (t_high - t_low), 0, 1
    )

    # 逐日对比
    sig_labels = {0: "空仓", 1: "半仓", 2: "满仓"}
    print(f"\n{'日期':<12} {'信号':<8} {'score':>10} {'离散仓位':>8} {'连续仓位':>8} {'实际涨跌':>10} {'离散验证':<8} {'连续验证':<8}")
    print("-" * 90)

    disc_correct = 0
    disc_valid = 0
    cont_correct = 0
    cont_valid = 0

    for _, r in merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        sig = int(r["signal"])
        ret = r["close_ret"]
        pd_ = r["pos_discrete"]
        pc = r["pos_continuous"]

        # 离散验证: 空仓+跌=正确, 满仓+涨=正确, 半仓=中性
        if sig == 0:
            dv = "正确" if ret < 0 else "错误"
            disc_valid += 1
            if ret < 0:
                disc_correct += 1
        elif sig == 2:
            dv = "正确" if ret > 0 else "错误"
            disc_valid += 1
            if ret > 0:
                disc_correct += 1
        else:
            dv = "中性"

        # 连续验证: 仓位与涨跌方向是否匹配
        # 仓位>50% 且 涨 = 正确 (持仓获利)
        # 仓位<50% 且 跌 = 正确 (减仓避险)
        # 仓位>50% 且 跌 = 错误
        # 仓位<50% 且 涨 = 错误
        # 仓位=0 且 跌 = 正确 (空仓避险)
        if pc > 0.5:
            cv = "正确" if ret > 0 else "错误"
            cont_valid += 1
            if ret > 0:
                cont_correct += 1
        elif pc < 0.5:
            cv = "正确" if ret < 0 else "错误"
            cont_valid += 1
            if ret < 0:
                cont_correct += 1
        else:
            cv = "中性"

        print(f"{d:<12} {sig} {sig_labels[sig]:<5} {r['score']:>10.6f} {pd_:>7.0%} {pc:>7.1%} {ret:>+9.2%} {dv:<8} {cv:<8}")

    # 汇总
    print(f"\n{'='*70}")
    print(f"汇总 ({merged['date'].min().date()} ~ {merged['date'].max().date()}, {len(merged)} 天)")
    print(f"{'='*70}")

    print(f"\n  离散仓位方向准确率: {disc_correct}/{disc_valid} = {disc_correct/disc_valid*100:.1f}%")
    print(f"  连续仓位方向准确率: {cont_correct}/{cont_valid} = {cont_correct/cont_valid*100:.1f}%")

    # 按信号分组
    print(f"\n  按信号分组:")
    for sig in [0, 1, 2]:
        sub = merged[merged["signal"] == sig]
        if len(sub) == 0:
            continue
        if sig == 0:
            disc_right = (sub["close_ret"] < 0).sum()
        elif sig == 2:
            disc_right = (sub["close_ret"] > 0).sum()
        else:
            disc_right = -1

        # 连续: 看仓位和涨跌是否匹配
        cont_right = 0
        cont_total = 0
        for _, r in sub.iterrows():
            pc = r["pos_continuous"]
            ret = r["close_ret"]
            if pc > 0.5:
                cont_total += 1
                if ret > 0:
                    cont_right += 1
            elif pc < 0.5:
                cont_total += 1
                if ret < 0:
                    cont_right += 1

        label = sig_labels[sig]
        n = len(sub)
        if disc_right >= 0:
            print(f"    {sig} {label} ({n}天): 离散 {disc_right}/{n}={disc_right/n*100:.1f}%"
                  f"  连续 {cont_right}/{cont_total}={cont_right/cont_total*100:.1f}%" if cont_total > 0 else
                  f"    {sig} {label} ({n}天): 离散 {disc_right}/{n}={disc_right/n*100:.1f}%  连续 N/A")
        else:
            cont_str = f"  连续 {cont_right}/{cont_total}={cont_right/cont_total*100:.1f}%" if cont_total > 0 else "  连续 N/A"
            print(f"    {sig} {label} ({n}天): 离散 中性{cont_str}")

    # 收益贡献对比
    print(f"\n  收益贡献:")
    merged["disc_contrib"] = merged["pos_discrete"] * merged["close_ret"]
    merged["cont_contrib"] = merged["pos_continuous"] * merged["close_ret"]

    up = merged[merged["close_ret"] > 0]
    dn = merged[merged["close_ret"] < 0]

    print(f"    上涨日 ({len(up)}天):")
    print(f"      离散: 仓位均值={up['pos_discrete'].mean():.1%}  收益贡献={up['disc_contrib'].sum():+.2%}")
    print(f"      连续: 仓位均值={up['pos_continuous'].mean():.1%}  收益贡献={up['cont_contrib'].sum():+.2%}")
    print(f"    下跌日 ({len(dn)}天):")
    print(f"      离散: 仓位均值={dn['pos_discrete'].mean():.1%}  收益贡献={dn['disc_contrib'].sum():+.2%}")
    print(f"      连续: 仓位均值={dn['pos_continuous'].mean():.1%}  收益贡献={dn['cont_contrib'].sum():+.2%}")

    disc_total = (1 + merged["disc_contrib"]).prod() - 1
    cont_total = (1 + merged["cont_contrib"]).prod() - 1
    bench_total = merged["close_ret"].sum()
    print(f"\n    累计收益: 离散={disc_total:+.2%}  连续={cont_total:+.2%}  基准={bench_total:+.2%}")


if __name__ == "__main__":
    main()
