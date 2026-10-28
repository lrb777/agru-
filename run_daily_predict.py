"""每日择时信号预测：从 DolphinDB 实时拉取因子，输出择时信号

用法:
  python run_daily_predict.py                        # 预测今日信号
  python run_daily_predict.py --date 2026-09-01      # 指定预测日期
  python run_daily_predict.py --year 2026            # 指定模型年份
  python run_daily_predict.py verify                 # 验证历史预测
  python run_daily_predict.py verify --last 5        # 验证最近5条

时序逻辑:
  T日收盘后因子可用 → 模型输入[T-19, T] → 预测T+1收益方向 → T+1开盘前调仓

产出:
  results/daily_predict.csv (追加模式)
"""
import os
import sys
import pickle
import argparse
from datetime import datetime, timedelta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, CHECKPOINT_DIR, RESULT_DIR,
    SELECTED_FEATURES, MODEL, FEATURES, LABEL, BACKTEST,
    DOLPHINDB, DB_D2, FACTOR_POOL,
)
from src.model import AGRU
from src.features import FeatureManager
from src.data_loader import DolphinDBLoader

INFER_BATCH = 5000
FETCH_CAL_DAYS = 45


def load_model_and_stats(model_year: int):
    model_path = os.path.join(CHECKPOINT_DIR, f"model_{model_year}.pt")
    stats_path = os.path.join(CHECKPOINT_DIR, f"stats_{model_year}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"统计量文件不存在: {stats_path}\n请先运行: python extract_stats.py --year {model_year}"
        )

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    with open(stats_path, "rb") as f:
        stats_data = pickle.load(f)

    model = AGRU(
        input_dim=len(SELECTED_FEATURES),
        hidden_dim=MODEL["hidden_dim"],
        num_gru_layers=MODEL["num_gru_layers"],
        num_classes=MODEL["num_classes"],
        dropout=MODEL["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, stats_data


def fetch_latest_factors(predict_date: str, feature_cols: list):
    end_date = pd.Timestamp(predict_date) - timedelta(days=1)
    start_date = end_date - timedelta(days=FETCH_CAL_DAYS)

    loader = DolphinDBLoader()
    print(f"  从 DolphinDB 拉取因子: {start_date.date()} ~ {end_date.date()} ...")
    df = loader.fetch_factors_wide(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        pool=FACTOR_POOL,
    )
    loader.close()

    if len(df) == 0:
        raise ValueError("DolphinDB 未返回任何数据")

    keep = ["date", "code"] + [c for c in feature_cols if c in df.columns]
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  !! 缺失特征: {missing}")

    print(f"  拉取完成: {len(df):,} 行, {df['code'].nunique():,} 只股票")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def build_windows(df: pd.DataFrame, feature_cols: list, lookback: int):
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    windows = []
    codes_used = []
    for code, g in df.groupby("code", sort=False):
        if len(g) < lookback:
            continue
        feats = g[feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        window = feats[-lookback:]
        windows.append(window)
        codes_used.append(code)

    if len(windows) == 0:
        raise ValueError(f"没有股票有足够的数据 ({lookback} 日)")

    tensor = np.stack(windows)
    print(f"  构建时序窗口: {tensor.shape[0]} 只股票 x {lookback} 日 x {len(feature_cols)} 特征")
    return tensor, codes_used


def is_trading_day(date_str: str) -> bool:
    """检查是否为交易日：周末一定不是，工作日默认是（不依赖本地数据范围）"""
    d = pd.Timestamp(date_str)
    if d.weekday() >= 5:
        return False
    # 检查是否在本地交易日历中（仅在本地数据覆盖范围内校验节假日）
    idx_path = os.path.join(DATA_DIR, "index_daily.parquet")
    if not os.path.exists(idx_path):
        return True
    idx = pd.read_parquet(idx_path, columns=["date"])
    idx["date"] = pd.to_datetime(idx["date"])
    max_local = idx["date"].max()
    if d <= max_local:
        return d in set(idx["date"])
    # 超出本地数据范围，只按周末判断（节假日无法校验）
    return True


def next_trading_day(date_str: str) -> str:
    """获取下一个交易日（用本地指数行情数据，若超出范围则用因子数据最后日期+1天）"""
    idx_path = os.path.join(DATA_DIR, "index_daily.parquet")
    idx = pd.read_parquet(idx_path, columns=["date"])
    idx["date"] = pd.to_datetime(idx["date"])
    d = pd.Timestamp(date_str)
    future = idx[idx["date"] >= d]["date"]
    if len(future) > 0:
        return future.iloc[0].strftime("%Y-%m-%d")
    # 本地交易日历不够用，往后推一个工作日
    from pandas.tseries.offsets import BDay
    return (d + BDay(1)).strftime("%Y-%m-%d")


def predict(predict_date: str = None, model_year: int = 2026):
    if predict_date is None:
        predict_date = datetime.now().strftime("%Y-%m-%d")

    # 检查是否为交易日
    if not is_trading_day(predict_date):
        next_td = next_trading_day(predict_date)
        if next_td != predict_date:
            print(f"  !! {predict_date} 不是交易日，自动调整为下一个交易日: {next_td}")
            predict_date = next_td
        else:
            print(f"  !! {predict_date} 不在本地交易日历中（可能超出数据范围），继续预测")

    print(f"=== 每日择时预测 ===")
    print(f"  预测日期: {predict_date}")
    print(f"  模型: model_{model_year}.pt")

    print(f"\n[1/5] 加载模型和统计量...")
    model, stats_data = load_model_and_stats(model_year)
    standardizer = stats_data["standardizer"]
    t_low, t_high = stats_data["threshold"]
    feature_cols = stats_data.get("feature_cols", list(SELECTED_FEATURES))
    print(f"  阈值: t_low={t_low:.6f}, t_high={t_high:.6f}")

    print(f"\n[2/5] 拉取最新因子数据...")
    df = fetch_latest_factors(predict_date, feature_cols)
    data_end_date = df["date"].max().date()

    print(f"\n[3/5] 清洗 + 标准化...")
    FeatureManager.clean_inplace(df, feature_cols)
    fm = FeatureManager()
    fm.feature_cols = feature_cols
    fm.apply_standardizer(df, standardizer)

    print(f"\n[4/5] 构建时序窗口 + 模型推理...")
    lookback = FEATURES["lookback_window"]
    tensor, codes_used = build_windows(df, feature_cols, lookback)

    all_scores = []
    with torch.no_grad():
        for i in range(0, len(tensor), INFER_BATCH):
            batch = torch.from_numpy(tensor[i:i + INFER_BATCH])
            probs = model.predict_proba(batch).numpy()
            scores = probs[:, 2] - probs[:, 0]
            all_scores.append(scores)
    all_scores = np.concatenate(all_scores)

    daily_score = float(np.mean(all_scores))
    if daily_score > t_high:
        signal = 2
    elif daily_score < t_low:
        signal = 0
    else:
        signal = 1

    position = BACKTEST["position_map"][signal]

    print(f"\n[5/5] 结果")
    print(f"  数据截止: {data_end_date}")
    print(f"  预测日期: {predict_date}")
    print(f"  股票数量: {len(codes_used)}")
    print(f"  score:    {daily_score:.6f}")
    print(f"  signal:   {signal} ({'空仓' if signal == 0 else '半仓' if signal == 1 else '满仓'})")
    print(f"  position: {position:.1f}")

    result = pd.DataFrame([{
        "date": predict_date,
        "signal": signal,
        "score": round(daily_score, 6),
        "position": position,
        "model_year": model_year,
        "n_stocks": len(codes_used),
        "data_end_date": str(data_end_date),
        "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])

    output_path = os.path.join(RESULT_DIR, "daily_predict.csv")
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        existing = existing[existing["date"] != predict_date]
        result = pd.concat([existing, result], ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)
    result.to_csv(output_path, index=False)
    print(f"\n  已保存: {output_path}")


def verify(last_n: int = 0):
    """验证历史预测：用实际指数收益检验信号方向是否正确

    时序: T-1日收盘数据 → 预测T日涨跌方向 → 用T日实际涨跌验证
    """
    from config import INDEX_CODE

    predict_path = os.path.join(RESULT_DIR, "daily_predict.csv")
    if not os.path.exists(predict_path):
        print("无历史预测记录: results/daily_predict.csv")
        return

    preds = pd.read_csv(predict_path)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.sort_values("date").reset_index(drop=True)

    if last_n > 0:
        preds = preds.tail(last_n).reset_index(drop=True)

    if len(preds) == 0:
        print("无预测记录可验证")
        return

    print(f"=== 验证历史预测 ({len(preds)} 条) ===")

    min_date = preds["date"].min()
    max_date = preds["date"].max() + timedelta(days=3)

    print(f"  预测区间: {min_date.date()} ~ {preds['date'].max().date()}")
    print(f"  拉取指数行情...")

    loader = DolphinDBLoader()
    idx = loader.fetch_index_daily(
        index_code=INDEX_CODE,
        start=(min_date - timedelta(days=5)).strftime("%Y-%m-%d"),
        end=max_date.strftime("%Y-%m-%d"),
    )
    loader.close()

    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["close_ret"] = idx["close"].pct_change()
    idx["open_ret"] = idx["open"].pct_change()

    merged = pd.merge(preds, idx[["date", "close", "close_ret", "open", "open_ret"]], on="date", how="left")

    sig_labels = {0: "空仓", 1: "半仓", 2: "满仓"}
    rows = []
    for _, r in merged.iterrows():
        ret = r["close_ret"]
        signal = int(r["signal"])
        if pd.isna(ret):
            result = "无数据"
        elif signal == 0 and ret < 0:
            result = "正确"
        elif signal == 0 and ret >= 0:
            result = "错误"
        elif signal == 2 and ret > 0:
            result = "正确"
        elif signal == 2 and ret <= 0:
            result = "错误"
        else:
            result = "中性"
        rows.append({
            "预测日期": r["date"].strftime("%Y-%m-%d"),
            "信号": f"{signal} {sig_labels[signal]}",
            "score": round(r["score"], 6),
            "实际涨跌": f"{ret:+.2%}" if not pd.isna(ret) else "N/A",
            "验证": result,
        })

    df_out = pd.DataFrame(rows)
    # 格式化对齐输出
    print()
    print(f"  {'预测日期':<12} {'信号':<8} {'score':>10} {'实际涨跌':>10} {'验证':<6}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
    for _, r in df_out.iterrows():
        print(f"  {r['预测日期']:<12} {r['信号']:<8} {r['score']:>10.6f} {r['实际涨跌']:>10} {r['验证']:<6}")

    valid = df_out[df_out["验证"].isin(["正确", "错误"])]
    if len(valid) > 0:
        n_correct = (valid["验证"] == "正确").sum()
        print(f"\n  方向预测准确率: {n_correct}/{len(valid)} = {n_correct/len(valid):.1%}")

        for sig in ["0 空仓", "2 满仓"]:
            sub = valid[valid["信号"] == sig]
            if len(sub) > 0:
                n_right = (sub["验证"] == "正确").sum()
                print(f"    {sig}: {n_right}/{len(sub)} = {n_right/len(sub):.1%}")

    print(f"\n  详细数据已保存: {predict_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="每日择时信号预测")
    parser.add_argument("command", nargs="?", default="predict", choices=["predict", "verify"],
                        help="predict=预测信号 (默认), verify=验证历史预测")
    parser.add_argument("--date", type=str, default=None, help="预测日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--year", type=int, default=2026, help="模型年份 (默认 2026)")
    parser.add_argument("--last", type=int, default=0, help="verify时只看最近N条 (0=全部)")
    args = parser.parse_args()

    if args.command == "verify":
        verify(last_n=args.last)
    else:
        predict(predict_date=args.date, model_year=args.year)
