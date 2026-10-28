"""批量补齐 2026-09-02 ~ 2026-10-18 的逐日预测并验证

流程（每个交易日）:
  1. 拉取截止到预测日前一日的因子（确保无未来函数）
  2. 模型推理 → 信号
  3. 获取预测日的实际涨跌
  4. 验证信号方向
"""
import os
import sys
import pickle
from datetime import timedelta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, CHECKPOINT_DIR, RESULT_DIR,
    SELECTED_FEATURES, MODEL, FEATURES, LABEL, BACKTEST,
    DOLPHINDB, INDEX_CODE,
)
from src.model import AGRU
from src.features import FeatureManager
from src.data_loader import DolphinDBLoader

INFER_BATCH = 5000
FETCH_CAL_DAYS = 45


def load_model_and_stats():
    model_path = os.path.join(CHECKPOINT_DIR, "model_2026.pt")
    stats_path = os.path.join(CHECKPOINT_DIR, "stats_2026.pkl")
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


def fetch_factors(start, end, feature_cols):
    loader = DolphinDBLoader()
    df = loader.fetch_factors_wide(start=start, end=end, pool="filtered")
    loader.close()
    keep = ["date", "code"] + [c for c in feature_cols if c in df.columns]
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_index(start, end):
    loader = DolphinDBLoader()
    idx = loader.fetch_index_daily(index_code=INDEX_CODE, start=start, end=end)
    loader.close()
    idx["date"] = pd.to_datetime(idx["date"])
    return idx


def build_windows(df, feature_cols, lookback):
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    windows = []
    for code, g in df.groupby("code", sort=False):
        if len(g) < lookback:
            continue
        feats = g[feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        windows.append(feats[-lookback:])
    return np.stack(windows)


def predict_one_day(predict_date, model, stats_data, feature_cols):
    """预测单日信号（只用 predict_date 前一天及之前的数据）"""
    end_date = pd.Timestamp(predict_date) - timedelta(days=1)
    start_date = end_date - timedelta(days=FETCH_CAL_DAYS)

    df = fetch_factors(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        feature_cols,
    )
    if len(df) == 0:
        return None

    FeatureManager.clean_inplace(df, feature_cols)
    fm = FeatureManager()
    fm.feature_cols = feature_cols
    fm.apply_standardizer(df, stats_data["standardizer"])

    lookback = FEATURES["lookback_window"]
    tensor = build_windows(df, feature_cols, lookback)

    all_scores = []
    with torch.no_grad():
        for i in range(0, len(tensor), INFER_BATCH):
            batch = torch.from_numpy(tensor[i:i + INFER_BATCH])
            probs = model.predict_proba(batch).numpy()
            scores = probs[:, 2] - probs[:, 0]
            all_scores.append(scores)
    all_scores = np.concatenate(all_scores)

    daily_score = float(np.mean(all_scores))
    t_low, t_high = stats_data["threshold"]
    if daily_score > t_high:
        signal = 2
    elif daily_score < t_low:
        signal = 0
    else:
        signal = 1

    data_end = df["date"].max().date()
    return {
        "date": predict_date,
        "signal": signal,
        "score": round(daily_score, 6),
        "position": BACKTEST["position_map"][signal],
        "n_stocks": len(tensor),
        "data_end_date": str(data_end),
    }


def main():
    print("=" * 70)
    print("批量预测: 2026-09-02 ~ 2026-10-18")
    print("=" * 70)

    # 获取交易日历 + 指数行情
    idx_all = fetch_index("2026-08-01", "2026-10-20")
    idx_all = idx_all.sort_values("date").reset_index(drop=True)
    idx_all["close_ret"] = idx_all["close"].pct_change()

    trading_days = idx_all[(idx_all["date"] >= "2026-09-02") & (idx_all["date"] <= "2026-10-18")]["date"].tolist()
    print(f"  交易日数: {len(trading_days)}")

    # 加载模型
    print("\n加载模型和统计量...")
    model, stats_data = load_model_and_stats()
    feature_cols = stats_data.get("feature_cols", list(SELECTED_FEATURES))
    t_low, t_high = stats_data["threshold"]
    print(f"  阈值: t_low={t_low:.6f}, t_high={t_high:.6f}")

    # 一次性拉取全部因子数据（从最早预测日前45天到最后预测日前1天）
    first_end = pd.Timestamp(trading_days[0]) - timedelta(days=1)
    last_end = pd.Timestamp(trading_days[-1]) - timedelta(days=1)
    fetch_start = (first_end - timedelta(days=FETCH_CAL_DAYS)).strftime("%Y-%m-%d")
    fetch_end = last_end.strftime("%Y-%m-%d")
    print(f"\n一次性拉取因子: {fetch_start} ~ {fetch_end} ...")
    all_factors = fetch_factors(fetch_start, fetch_end, feature_cols)
    all_factors = all_factors.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  因子数据: {len(all_factors):,} 行, 日期 {all_factors['date'].min().date()} ~ {all_factors['date'].max().date()}")

    # 清洗 + 标准化
    FeatureManager.clean_inplace(all_factors, feature_cols)
    fm = FeatureManager()
    fm.feature_cols = feature_cols
    fm.apply_standardizer(all_factors, stats_data["standardizer"])

    lookback = FEATURES["lookback_window"]
    predict_path = os.path.join(RESULT_DIR, "daily_predict.csv")

    # 逐日预测
    results = []
    sig_labels = {0: "空仓", 1: "半仓", 2: "满仓"}

    for i, d in enumerate(trading_days):
        d_str = d.strftime("%Y-%m-%d")
        d_ts = pd.Timestamp(d)

        # 筛选该预测日可用的因子数据（截止前一日）
        factors_day = all_factors[all_factors["date"] < d_ts].copy()
        if len(factors_day) == 0:
            print(f"  [{i+1}/{len(trading_days)}] {d_str}: 无数据，跳过")
            continue

        # 构建时序窗口
        tensor = build_windows(factors_day, feature_cols, lookback)
        if len(tensor) == 0:
            print(f"  [{i+1}/{len(trading_days)}] {d_str}: 无足够窗口，跳过")
            continue

        # 推理
        all_scores = []
        with torch.no_grad():
            for j in range(0, len(tensor), INFER_BATCH):
                batch = torch.from_numpy(tensor[j:j + INFER_BATCH])
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

        # 获取实际涨跌
        row = idx_all[idx_all["date"] == d]
        actual_ret = float(row.iloc[0]["close_ret"]) if len(row) > 0 and not pd.isna(row.iloc[0]["close_ret"]) else None

        ret_str = f"{actual_ret:+.2%}" if actual_ret is not None else "N/A"
        print(f"  [{i+1}/{len(trading_days)}] {d_str}  信号={signal}({sig_labels[signal]})  score={daily_score:.6f}  实际={ret_str}")

        r = {
            "date": d_str,
            "signal": signal,
            "score": round(daily_score, 6),
            "position": BACKTEST["position_map"][signal],
            "model_year": 2026,
            "n_stocks": len(tensor),
            "data_end_date": str(factors_day["date"].max().date()),
            "predict_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "actual_ret": round(actual_ret, 6) if actual_ret is not None else None,
        }
        results.append(r)

        # 每条预测后立即保存（防止中断丢失）
        df_new = pd.DataFrame(results)
        if os.path.exists(predict_path):
            existing = pd.read_csv(predict_path)
            existing = existing[~existing["date"].isin(df_new["date"].tolist())]
            combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            combined = df_new
        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_csv(predict_path, index=False)

    # 验证汇总
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)
    print(f"\n  {'日期':<12} {'信号':<8} {'score':>10} {'实际涨跌':>10} {'验证':<6}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")

    n_correct = 0
    n_valid = 0
    sig_stats = {0: [0, 0], 2: [0, 0]}

    for r in results:
        signal = r["signal"]
        ret = r.get("actual_ret")

        if ret is None:
            verdict = "无数据"
        elif signal == 0 and ret < 0:
            verdict = "正确"
            n_valid += 1
            n_correct += 1
            sig_stats[0][0] += 1
            sig_stats[0][1] += 1
        elif signal == 0 and ret >= 0:
            verdict = "错误"
            n_valid += 1
            sig_stats[0][1] += 1
        elif signal == 2 and ret > 0:
            verdict = "正确"
            n_valid += 1
            n_correct += 1
            sig_stats[2][0] += 1
            sig_stats[2][1] += 1
        elif signal == 2 and ret <= 0:
            verdict = "错误"
            n_valid += 1
            sig_stats[2][1] += 1
        else:
            verdict = "中性"

        ret_str = f"{ret:+.2%}" if ret is not None else "N/A"
        print(f"  {r['date']:<12} {signal} {sig_labels[signal]:<5} {r['score']:>10.6f} {ret_str:>10} {verdict:<6}")

    if n_valid > 0:
        print(f"\n  方向预测准确率: {n_correct}/{n_valid} = {n_correct/n_valid:.1%}")
    else:
        print(f"\n  无可验证的方向信号")
    for sig, (c, t) in sig_stats.items():
        if t > 0:
            label = sig_labels[sig]
            print(f"    {sig} {label}: {c}/{t} = {c/t:.1%}")

    print(f"\n  已保存: {predict_path}")


if __name__ == "__main__":
    main()
