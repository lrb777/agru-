"""主流程: 数据探查 -> 取数 -> 预处理 -> 训练 -> 回测 -> 对标

用法:
  python run.py smoke        # 合成数据验证全流程
  python run.py discover     # 探查 DolphinDB
  python run.py fetch        # 拉取数据到本地 parquet
  python run.py preprocess   # 预处理: 精简 parquet (仅40特征, float32)
  python run.py train        # 全量训练 (保存信号到 results/)
  python run.py backtest     # 仅回测 (读取已保存信号)
  python run.py run          # 训练 + 回测
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(os.cpu_count() or 4)

import numpy as np
import pandas as pd

from config import *
from config import DB_D0, DB_D2, FACTOR_POOL, SELECTED_FEATURES, CHECKPOINT_DIR, RESULT_DIR, DATA_DIR
from src.data_loader import DolphinDBLoader
from src.features import FeatureManager
from src.train import RollingTrainer, mem_mb
from src.backtest import Backtester


def setup_logging():
    log_file = os.path.join(RESULT_DIR, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info(f"日志文件: {log_file}")
    return log_file


def discover():
    loader = DolphinDBLoader()
    print("=== DolphinDB 连接成功 ===")
    print(f"  {loader.host}:{loader.port}")
    print(f"\n  指数行情: {DB_D0}.jy_qt_indexquote (join jy_secumain)")
    print(f"  个股日线: {DB_D0}.ts_daily")
    print(f"  因子数据: {DB_D2}.factors_std (pool={FACTOR_POOL})")
    factors = loader.fetch_factor_list()
    print(f"\n  可用因子: {len(factors)} 个")
    for i in range(0, len(factors), 10):
        print(f"    {', '.join(factors[i:i+10])}")
    loader.close()


def fetch():
    loader = DolphinDBLoader()
    idx, factors = loader.fetch_all()
    loader.close()
    print(f"\n数据拉取完成: 指数 {len(idx):,} 行, 因子 {len(factors):,} 行 x {len(factors.columns)-2} 因子")


def preprocess():
    """预处理: 将每年 parquet 精简为仅40特征 + float32, 减小文件和内存"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    keep = ["date", "code"] + [c for c in SELECTED_FEATURES]
    print(f"=== 预处理: 精简 parquet ({len(SELECTED_FEATURES)} 特征, float32) ===\n")

    total_saved = 0
    for yr in range(2015, 2026):
        src = os.path.join(DATA_DIR, f"factors_{yr}.parquet")
        dst = os.path.join(DATA_DIR, f"factors_{yr}_slim.parquet")
        if not os.path.exists(src):
            print(f"  {yr}: 源文件不存在, 跳过")
            continue
        if os.path.exists(dst):
            print(f"  {yr}: slim 已存在, 跳过")
            continue

        src_size = os.path.getsize(src) / 1e6
        print(f"  {yr}: 读取 ({src_size:.0f}MB)...", end=" ", flush=True)
        df = pd.read_parquet(src, columns=keep)
        for c in SELECTED_FEATURES:
            if c in df.columns:
                df[c] = df[c].astype(np.float32)
        df.to_parquet(dst, index=False)
        dst_size = os.path.getsize(dst) / 1e6
        saved = src_size - dst_size
        total_saved += saved
        print(f"{len(df):,} 行 -> {dst_size:.0f}MB (省 {saved:.0f}MB)")

    print(f"\n预处理完成。总节省磁盘: {total_saved/1000:.1f}GB")
    print(f"内存占用: {mem_mb():.0f} MB")


def load_index():
    idx = pd.read_parquet(os.path.join(DATA_DIR, "index_daily.parquet"))
    print(f"  指数: {len(idx):,} 行")
    return idx


def train_only():
    log_file = setup_logging()
    logging.info("=" * 70)
    logging.info("AGRU 神经网络大盘择时 - 全量训练")
    logging.info("=" * 70)

    idx = load_index()

    has_slim = os.path.exists(os.path.join(DATA_DIR, "factors_2015_slim.parquet"))
    if not has_slim:
        logging.warning("未发现 slim 预处理文件, 将使用原始 parquet (内存较大)")
        logging.warning("建议先运行: python run.py preprocess")

    logging.info("启动滚动训练 (按折加载, 不一次性加载全部数据)...")
    trainer = RollingTrainer(index_df=idx, data_dir=DATA_DIR)
    signals = trainer.run()

    if len(signals) > 0:
        signals_path = os.path.join(RESULT_DIR, "signals.csv")
        signals.to_csv(signals_path, index=False)
        logging.info(f"全部信号已保存: {signals_path} ({len(signals)} 天)")
        logging.info(f"信号分布: {signals['signal'].value_counts().to_dict()}")
    else:
        logging.warning("未生成任何信号!")

    logging.info(f"训练完成。日志: {log_file}")


def backtest_only():
    signals_path = os.path.join(RESULT_DIR, "signals.csv")
    if not os.path.exists(signals_path):
        print(f"信号文件不存在: {signals_path}")
        print("请先运行: python run.py train")
        return

    idx = load_index()
    signals = pd.read_csv(signals_path)
    signals["date"] = pd.to_datetime(signals["date"])

    print(f"\n加载信号: {len(signals)} 天")
    print(f"信号分布: {signals['signal'].value_counts().to_dict()}")

    run_backtest(idx, signals)


def train_and_backtest():
    log_file = setup_logging()
    logging.info("=" * 70)
    logging.info("AGRU 神经网络大盘择时 - 训练 + 回测")
    logging.info("=" * 70)

    idx = load_index()

    has_slim = os.path.exists(os.path.join(DATA_DIR, "factors_2015_slim.parquet"))
    if not has_slim:
        logging.warning("未发现 slim 预处理文件, 将使用原始 parquet (内存较大)")
        logging.warning("建议先运行: python run.py preprocess")

    logging.info("启动滚动训练 (按折加载)...")
    trainer = RollingTrainer(index_df=idx, data_dir=DATA_DIR)
    signals = trainer.run()

    if len(signals) == 0:
        logging.warning("未生成任何信号!")
        return

    signals_path = os.path.join(RESULT_DIR, "signals.csv")
    signals.to_csv(signals_path, index=False)
    logging.info(f"信号已保存: {signals_path} ({len(signals)} 天)")

    run_backtest(idx, signals, logger=logging.getLogger())


def run_backtest(idx, signals, logger=None):
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    log("\n" + "=" * 70)
    log("回测")
    log("=" * 70)

    bt = Backtester(idx)
    result = bt.run(signals)
    m = result["metrics"]

    log("\n" + "=" * 70)
    log("策略表现 (对标验收基线 A)")
    log("=" * 70)
    log(f"  总收益率:     {m['总收益率']:.2%}   (目标 131.87%)")
    log(f"  年化收益率:   {m['年化收益率']:.2%}   (目标 15.86%)")
    log(f"  年化波动率:   {m['年化波动率']:.2%}   (目标 13.44%)")
    log(f"  夏普比率:     {m['夏普比率']:.2f}   (目标 1.18)")
    log(f"  最大回撤:     {m['最大回撤']:.2%}   (目标 -12.66%)")
    log(f"  最大回撤区间: {m['最大回撤开始日期']} ~ {m['最大回撤结束日期']}")
    log(f"  择时IC:       {m.get('择时IC', 0):.2%}   (目标 15.81%)")

    log("\n信号准确性 (对标验收基线 B)")
    log(result["signal_accuracy"].to_string())

    daily_path = os.path.join(RESULT_DIR, "daily_result.csv")
    result["daily"].to_csv(daily_path, index=False)
    log(f"\n逐日结果已保存: {daily_path}")


def smoke_test():
    import numpy as np
    import pandas as pd
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    print("=== Smoke Test: 合成数据验证 ===\n")

    np.random.seed(42)
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    n_stocks = 50
    n_features = 10
    feature_names = [f"f{i}" for i in range(n_features)]
    rows = []
    for code in range(n_stocks):
        for d in dates:
            row = {"code": f"S{code:04d}", "date": d}
            for fi in range(n_features):
                row[f"f{fi}"] = np.random.randn()
            rows.append(row)
    factors = pd.DataFrame(rows)
    idx = pd.DataFrame({
        "date": dates,
        "open": np.cumprod(1 + np.random.randn(len(dates)) * 0.01) * 3000,
        "close": np.cumprod(1 + np.random.randn(len(dates)) * 0.01) * 3000,
        "high": 0, "low": 0,
    })
    idx["high"] = idx["close"] * 1.01
    idx["low"] = idx["close"] * 0.99

    import config
    config.SELECTED_FEATURES = feature_names
    smoke_ckpt = os.path.join(config.PROJECT_ROOT, "checkpoints_smoke")
    os.makedirs(smoke_ckpt, exist_ok=True)
    config.CHECKPOINT_DIR = smoke_ckpt

    old_rolling = config.ROLLING_YEARS
    old_batch = config.TRAIN["batch_size"]
    old_epochs = config.TRAIN["max_epochs"]
    config.ROLLING_YEARS = [2023, 2024]
    config.TRAIN["batch_size"] = 1024
    config.TRAIN["max_epochs"] = 5

    trainer = RollingTrainer(index_df=idx, feature_df=factors)
    signals = trainer.run()

    config.ROLLING_YEARS = old_rolling
    config.TRAIN["batch_size"] = old_batch
    config.TRAIN["max_epochs"] = old_epochs

    if len(signals) > 0:
        print(f"\n生成信号: {len(signals)} 天")
        print(f"信号分布: {signals['signal'].value_counts().to_dict()}")
        bt = Backtester(idx)
        result = bt.run(signals)
        m = result["metrics"]
        print(f"\n回测指标:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("\n信号准确性:")
        print(result["signal_accuracy"].to_string())
    else:
        print("!! 未生成信号")

    import shutil
    shutil.rmtree(smoke_ckpt, ignore_errors=True)
    print("\n=== Smoke Test 完成 (已清理临时checkpoint) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGRU 神经网络大盘择时策略")
    parser.add_argument("mode", choices=["smoke", "discover", "fetch", "preprocess", "train", "backtest", "run"],
                        default="smoke", nargs="?",
                        help="smoke=合成验证, discover=探查DB, fetch=拉取数据, "
                             "preprocess=精简parquet, train=仅训练, backtest=仅回测, run=训练+回测")
    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
    elif args.mode == "discover":
        discover()
    elif args.mode == "fetch":
        fetch()
    elif args.mode == "preprocess":
        preprocess()
    elif args.mode == "train":
        train_only()
    elif args.mode == "backtest":
        backtest_only()
    elif args.mode == "run":
        train_and_backtest()
