"""连续仓位验证：用已有预测结果，对比离散 vs 连续仓位的实际表现

用 t_low/t_high 把 score 线性映射为连续仓位，验证 2026-09-01 ~ 2026-10-27
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULT_DIR, CHECKPOINT_DIR, BACKTEST, INDEX_CODE, DOLPHINDB

T_LOW = None
T_HIGH = None
COMMISSION = BACKTEST["commission_rate"]


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

    # 拉取指数实际收益
    import dolphindb as ddb
    session = ddb.session()
    session.connect(DOLPHINDB["host"], DOLPHINDB["port"], DOLPHINDB["username"], DOLPHINDB["password"])
    script = f"""
        t = loadTable("dfs://d0_day", "jy_qt_indexquote")
        s = loadTable("dfs://d0_day", "jy_secumain")
        select s.SecuCode as code, t.TradingDay as date, t.ClosePrice as close
        from t join s on t.InnerCode = s.InnerCode
        where s.SecuCode = "{INDEX_CODE}"
          and t.TradingDay between 2026.08.28 and 2026.10.30
        order by t.TradingDay
    """
    idx = session.run(script)
    session.close()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["close_ret"] = idx["close"].pct_change()

    merged = pd.merge(dp[["date", "signal", "score"]], idx[["date", "close_ret"]], on="date", how="left")

    # 计算连续仓位
    merged["pos_continuous"] = np.clip(
        (merged["score"] - t_low) / (t_high - t_low), 0, 1
    )
    # 离散仓位
    pos_map = {0: 0.0, 1: 0.5, 2: 1.0}
    merged["pos_discrete"] = merged["signal"].map(pos_map)

    # 逐日收益贡献 = 仓位 × 实际涨跌
    merged["disc_ret"] = merged["pos_discrete"] * merged["close_ret"]
    merged["cont_ret"] = merged["pos_continuous"] * merged["close_ret"]
    merged["bench_ret"] = merged["close_ret"]

    merged = merged.dropna(subset=["close_ret"]).reset_index(drop=True)

    # 输出对比表
    sig_labels = {0: "空仓", 1: "半仓", 2: "满仓"}
    print(f"\n{'日期':<12} {'信号':<8} {'score':>10} {'连续仓位':>8} {'离散仓位':>8} {'实际涨跌':>10} {'连续收益':>10} {'离散收益':>10}")
    print("-" * 90)

    for _, r in merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        sig = int(r["signal"])
        ret = r["close_ret"]
        pc = r["pos_continuous"]
        pd_ = r["pos_discrete"]
        cr = r["cont_ret"]
        dr = r["disc_ret"]
        print(f"{d:<12} {sig} {sig_labels[sig]:<5} {r['score']:>10.6f} {pc:>7.1%} {pd_:>7.0%} {ret:>+9.2%} {cr:>+9.2%} {dr:>+9.2%}")

    # 汇总
    n = len(merged)
    disc_total = (1 + merged["disc_ret"]).prod() - 1
    cont_total = (1 + merged["cont_ret"]).prod() - 1
    bench_total = (1 + merged["bench_ret"]).prod() - 1

    disc_vol = merged["disc_ret"].std() * np.sqrt(252)
    cont_vol = merged["cont_ret"].std() * np.sqrt(252)
    bench_vol = merged["bench_ret"].std() * np.sqrt(252)

    disc_sharpe = disc_total / (disc_vol * n / 252) if disc_vol > 0 else 0
    cont_sharpe = cont_total / (cont_vol * n / 252) if cont_vol > 0 else 0
    bench_sharpe = bench_total / (bench_vol * n / 252) if bench_vol > 0 else 0

    # 最大回撤
    def max_dd(rets):
        cum = (1 + rets).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()

    print(f"\n{'='*70}")
    print(f"汇总 ({merged['date'].min().date()} ~ {merged['date'].max().date()}, {n} 天)")
    print(f"{'='*70}")
    print(f"{'指标':<16} {'离散仓位':>12} {'连续仓位':>12} {'基准(持有)':>12} {'连续-离散':>10}")
    print("-" * 66)
    print(f"{'累计收益率':<16} {disc_total:>11.2%} {cont_total:>11.2%} {bench_total:>11.2%} {cont_total-disc_total:>+9.2%}")
    print(f"{'年化波动率':<16} {disc_vol:>11.2%} {cont_vol:>11.2%} {bench_vol:>11.2%} {cont_vol-disc_vol:>+9.2%}")
    print(f"{'夏普比率(年化)':<16} {disc_sharpe:>12.4f} {cont_sharpe:>12.4f} {bench_sharpe:>12.4f} {cont_sharpe-disc_sharpe:>+10.4f}")
    print(f"{'最大回撤':<16} {max_dd(merged['disc_ret']):>11.2%} {max_dd(merged['cont_ret']):>11.2%} {max_dd(merged['bench_ret']):>11.2%}")

    # 仓位均值
    print(f"\n{'平均仓位':<16} {merged['pos_discrete'].mean():>11.1%} {merged['pos_continuous'].mean():>11.1%} {'100.0%':>11}")

    # 上涨/下跌日分别的仓位
    up = merged[merged["close_ret"] > 0]
    dn = merged[merged["close_ret"] < 0]
    print(f"\n{'上涨日平均仓位':<16} {'':<12} {'离散':<8} {'连续':<8}")
    print(f"  上涨({len(up)}天): 离散={up['pos_discrete'].mean():.1%}  连续={up['pos_continuous'].mean():.1%}")
    print(f"  下跌({len(dn)}天): 离散={dn['pos_discrete'].mean():.1%}  连续={dn['pos_continuous'].mean():.1%}")


if __name__ == "__main__":
    main()
