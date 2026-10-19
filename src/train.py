"""滚动训练流水线: 8年训练+2年验证+1年测试, 逐年滚动, 每日聚合个股预测为择时信号

内存优化策略:
  1. 按折加载: 每折只从 parquet 读取所需年份, 不一次性加载全部
  2. float32: 加载后立即转 float32, 内存减半
  3. 原地标准化: apply_standardizer 不 copy DataFrame
  4. 每折结束释放 store + model
  5. slim parquet: 优先读取预处理后的精简版 (仅40特征+date+code, float32)
"""
import os
import gc
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from config import MODEL, FEATURES, LABEL, RANDOM_SEED, GRAD_CLIP
from .model import AGRU
from .features import FeatureManager
from .dataset import make_market_labels, TimingDataStore, TimingDataset
from .metrics import tsrankic

logger = logging.getLogger(__name__)


def mem_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


class RollingTrainer:
    def __init__(self, index_df: pd.DataFrame, feature_df: pd.DataFrame = None,
                 data_dir: str = None):
        self.index_df = index_df
        self.data_dir = data_dir or config.DATA_DIR
        self._all_data = feature_df
        self.feature_cols = list(config.SELECTED_FEATURES)
        self.fm = FeatureManager()
        self.fm.feature_cols = self.feature_cols
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(RANDOM_SEED)

        if feature_df is not None:
            self._n_stocks = feature_df["code"].nunique()
            self._n_rows = len(feature_df)
        else:
            self._n_stocks = None
            self._n_rows = None

        logger.info(f"  设备: {self.device}")
        logger.info(f"  特征数: {len(self.feature_cols)}")
        if self._n_stocks:
            logger.info(f"  股票数: {self._n_stocks:,}")
            logger.info(f"  总行数: {self._n_rows:,}")
        logger.info(f"  数据目录: {self.data_dir}")
        logger.info(f"  滚动折: {config.ROLLING_YEARS}")
        logger.info(f"  训练参数: batch={config.TRAIN['batch_size']}, epochs={config.TRAIN['max_epochs']}, "
                     f"lr={config.TRAIN['lr']}, patience={config.TRAIN['patience']}, grad_clip={GRAD_CLIP}")
        logger.info(f"  内存占用: {mem_mb():.0f} MB")

    def _load_fold_data(self, year_start: int, year_end: int) -> pd.DataFrame:
        if self._all_data is not None:
            feat = self._all_data
            mask = feat["date"].dt.year.between(year_start, year_end)
            return feat.loc[mask].copy()

        keep = ["date", "code"] + [c for c in self.feature_cols]
        parts = []
        for yr in range(year_start, year_end + 1):
            slim = os.path.join(self.data_dir, f"factors_{yr}_slim.parquet")
            full = os.path.join(self.data_dir, f"factors_{yr}.parquet")
            if os.path.exists(slim):
                df = pd.read_parquet(slim)
            elif os.path.exists(full):
                df = pd.read_parquet(full)
                df = df[keep]
                for c in self.feature_cols:
                    if c in df.columns:
                        df[c] = df[c].astype(np.float32)
            else:
                logger.warning(f"  !! 找不到 {yr} 的因子文件")
                continue
            parts.append(df)
            logger.info(f"    加载 {yr}: {len(df):,} 行, mem={mem_mb():.0f}MB")

        feat = pd.concat(parts, ignore_index=True)
        feat["date"] = pd.to_datetime(feat["date"])
        del parts
        gc.collect()
        return feat

    def run(self) -> pd.DataFrame:
        results = []
        for test_year in config.ROLLING_YEARS:
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"  滚动训练: 测试年 = {test_year}")
            logger.info("=" * 70)

            ckpt_signals = os.path.join(config.CHECKPOINT_DIR, f"signals_{test_year}.csv")

            if os.path.exists(ckpt_signals):
                logger.info(f"  [checkpoint] 发现已完成折, 跳过")
                fold_result = pd.read_csv(ckpt_signals)
                fold_result["date"] = pd.to_datetime(fold_result["date"])
                results.append(fold_result)
                logger.info(f"  信号分布: {fold_result['signal'].value_counts().to_dict()} ({len(fold_result)} 天)")
                continue

            try:
                fold_result = self._train_one_fold(test_year)
                if len(fold_result) > 0:
                    results.append(fold_result)
                    fold_result.to_csv(ckpt_signals, index=False)
                    logger.info(f"  [checkpoint] 信号已保存: {ckpt_signals}")
            except Exception as e:
                logger.error(f"  !! 错误: {e}", exc_info=True)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"  内存占用: {mem_mb():.0f} MB")

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def _train_one_fold(self, test_year: int) -> pd.DataFrame:
        train_start = max(test_year - 10, 2015)
        train_end = test_year - 3
        val_start = test_year - 2
        val_end = test_year - 1

        logger.info(f"  区间: train={train_start}-{train_end}, val={val_start}-{val_end}, test={test_year}")
        logger.info(f"  内存占用 (折开始): {mem_mb():.0f} MB")

        idx = self.index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx["year"] = idx["date"].dt.year

        train_dates = idx[(idx["year"] >= train_start) & (idx["year"] <= train_end)]["date"]
        labels = make_market_labels(
            self.index_df,
            forecast_horizon=LABEL["forecast_horizon"],
            class_ratios=LABEL["class_ratios"],
            fit_dates=train_dates,
        )
        labels["date"] = pd.to_datetime(labels["date"])
        del idx

        logger.info(f"  加载数据 ({train_start}-{test_year})...")
        t0 = time.time()
        feat = self._load_fold_data(train_start, test_year)
        feat["year"] = feat["date"].dt.year
        logger.info(f"  数据加载完成: {len(feat):,} 行, {feat['code'].nunique():,} 只股票, {time.time()-t0:.1f}s")
        logger.info(f"  内存占用 (加载后): {mem_mb():.0f} MB")

        FeatureManager.clean_inplace(feat, self.feature_cols)

        train_mask = feat["year"].between(train_start, train_end).values
        stats = self.fm.fit_standardizer(feat, mask=train_mask)
        del train_mask
        feat = self.fm.apply_standardizer(feat, stats)
        del stats
        gc.collect()
        logger.info(f"  内存占用 (标准化后): {mem_mb():.0f} MB")

        T = FEATURES["lookback_window"]
        logger.info(f"  构建共享数据存储...")
        t0 = time.time()
        store = TimingDataStore(feat, labels, T, self.feature_cols)
        del feat
        gc.collect()
        logger.info(f"  数据存储构建完成: {store.n_stocks():,} 只股票, {time.time()-t0:.1f}s")
        logger.info(f"  内存占用 (store后): {mem_mb():.0f} MB")

        logger.info(f"  构建训练/验证/测试子集...")
        train_ds = TimingDataset(store, min_year=train_start, max_year=train_end)
        val_ds = TimingDataset(store, min_year=val_start, max_year=val_end)
        test_ds = TimingDataset(store, min_year=test_year, max_year=test_year)

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            logger.warning(f"  !! 样本量不足: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
            del store
            return pd.DataFrame()

        logger.info(f"  样本量: train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")

        model = AGRU(
            input_dim=len(self.feature_cols),
            hidden_dim=MODEL["hidden_dim"],
            num_gru_layers=MODEL["num_gru_layers"],
            num_classes=MODEL["num_classes"],
            dropout=MODEL["dropout"],
        ).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  模型参数量: {n_params:,}")

        best_ic = self._train_model(model, train_ds, val_ds, test_year)

        model_path = os.path.join(config.CHECKPOINT_DIR, f"model_{test_year}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "test_year": test_year,
            "best_ic": best_ic,
            "input_dim": len(self.feature_cols),
            "feature_cols": self.feature_cols,
        }, model_path)
        logger.info(f"  [checkpoint] 模型权重已保存: {model_path}")

        logger.info(f"  生成验证集阈值...")
        val_scores, val_thresh = self._predict_daily_scores(model, val_ds, fit_thresholds=True)
        logger.info(f"  生成测试集信号...")
        test_scores, _ = self._predict_daily_scores(model, test_ds, fit_thresholds=False)

        del model, train_ds, val_ds, test_ds, store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t_low, t_high = val_thresh
        rows = []
        for d, score in test_scores:
            if score > t_high:
                sig = 2
            elif score < t_low:
                sig = 0
            else:
                sig = 1
            rows.append({"date": d, "signal": sig, "score": float(score)})
        result = pd.DataFrame(rows)
        logger.info(f"  信号分布: {result['signal'].value_counts().to_dict()}")
        logger.info(f"  内存占用 (折结束): {mem_mb():.0f} MB")
        return result

    def _train_model(self, model: nn.Module, train_ds: TimingDataset,
                     val_ds: TimingDataset, test_year: int) -> float:
        batch_size = config.TRAIN["batch_size"]
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN["lr"])
        criterion = nn.CrossEntropyLoss()

        best_ic = -999.0
        best_state = None
        patience_counter = 0
        n_batches = len(train_loader)
        fold_start = time.time()

        for epoch in range(config.TRAIN["max_epochs"]):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            n_done = 0

            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                total_loss += loss.item()
                n_done += 1

                if (batch_idx + 1) % 200 == 0:
                    elapsed = time.time() - epoch_start
                    pct = (batch_idx + 1) / n_batches * 100
                    eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
                    logger.info(f"    batch {batch_idx+1}/{n_batches} ({pct:.0f}%) | "
                                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, mem={mem_mb():.0f}MB")

            epoch_train_time = time.time() - epoch_start

            model.eval()
            val_start = time.time()
            val_scores, _ = self._predict_daily_scores(model, val_ds, fit_thresholds=False)
            val_time = time.time() - val_start

            val_labels = val_ds.label_map
            score_dict = dict(val_scores)
            common = sorted(set(score_dict.keys()) & set(val_labels.keys()))
            if len(common) >= 2:
                preds = np.array([score_dict[d] for d in common])
                actuals = np.array([val_labels[d] for d in common])
                ic = tsrankic(preds, actuals)
            else:
                ic = 0.0

            avg_loss = total_loss / max(n_done, 1)
            total_elapsed = time.time() - fold_start
            logger.info(
                f"  Epoch {epoch+1:3d}/{config.TRAIN['max_epochs']} | "
                f"loss={avg_loss:.4f} | IC={ic:.4f} | "
                f"train={epoch_train_time:.0f}s val={val_time:.0f}s | "
                f"total={total_elapsed:.0f}s | mem={mem_mb():.0f}MB"
            )

            if ic > best_ic:
                best_ic = ic
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                logger.info(f"    ^ new best IC={best_ic:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.TRAIN["patience"]:
                    logger.info(f"  早停 @ epoch {epoch+1}, best IC={best_ic:.4f}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        logger.info(f"  训练完成: best val TSRankIC = {best_ic:.4f}, "
                     f"总耗时 {time.time()-fold_start:.0f}s")
        return best_ic

    def _predict_daily_scores(self, model: nn.Module, ds: TimingDataset, fit_thresholds: bool):
        model.eval()
        loader = DataLoader(ds, batch_size=config.TRAIN["batch_size"], shuffle=False, num_workers=0)

        sample_scores = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                probs = model.predict_proba(xb).cpu().numpy()
                scores = probs[:, 2] - probs[:, 0]
                sample_scores.append(scores)

        sample_scores = np.concatenate(sample_scores)
        sample_dates = ds.sample_dates

        date_to_scores = {}
        for d, s in zip(sample_dates, sample_scores):
            if d not in date_to_scores:
                date_to_scores[d] = []
            date_to_scores[d].append(float(s))

        daily_scores = [(d, np.mean(ss)) for d, ss in sorted(date_to_scores.items())]

        thresh = None
        if fit_thresholds and len(daily_scores) > 10:
            arr = np.array([s for _, s in daily_scores])
            r = LABEL["class_ratios"]
            t_low = np.quantile(arr, r[0])
            t_high = np.quantile(arr, r[0] + r[1])
            thresh = (t_low, t_high)

        return daily_scores, thresh
