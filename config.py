"""AGRU 神经网络大盘择时策略 - 全局配置"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
for _d in (DATA_DIR, CHECKPOINT_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

DOLPHINDB = {
    "host": "10.65.0.75",
    "port": 8902,
    "username": "admin",
    "password": "123456",
}

DB_D0 = "dfs://d0_day"
DB_D2 = "dfs://d2_day"

INDEX_CODE = "000985"
INDEX_INNER_CODE = 14110
ALT_INDICES = {"000300": 3145, "000852": 39144}

DATA_START = "2010-01-01"
DATA_END = "2026-08-31"
BT_START = "2020-01-01"
BT_END = "2025-09-30"

FACTOR_POOL = "filtered"

SELECTED_FEATURES = [
    "alpha158_kbar_KMID", "alpha158_kbar_KLEN", "alpha158_kbar_KUP", "alpha158_kbar_KLOW", "alpha158_kbar_KSFT",
    "alpha158_price_OPEN0", "alpha158_price_VWAP0",
    "alpha158_rolling_MA5", "alpha158_rolling_MA20", "alpha158_rolling_MA60",
    "alpha158_rolling_STD5", "alpha158_rolling_STD20", "alpha158_rolling_STD60",
    "alpha158_rolling_ROC5", "alpha158_rolling_ROC20", "alpha158_rolling_ROC60",
    "alpha158_rolling_RSV5", "alpha158_rolling_RSV20",
    "alpha158_rolling_MAX5", "alpha158_rolling_MIN5", "alpha158_rolling_MAX20", "alpha158_rolling_MIN20",
    "alpha158_rolling_RANK5", "alpha158_rolling_RANK20",
    "alpha158_volume_VMA5", "alpha158_volume_VMA20", "alpha158_volume_VSTD5", "alpha158_volume_WVMA20",
    "alpha158_vol_price_CORR5", "alpha158_vol_price_CORR20",
    "alpha158_reg_BETA5", "alpha158_reg_RESI20",
    "alpha158_updown_CNTP5", "alpha158_updown_CNTN20",
    "cicc_mmt_normal_M", "cicc_mmt_route_M", "cicc_vol_std_1M", "cicc_liq_turn_avg_1M",
    "custom_SIZE_LN", "custom_SIZE_RANK",
]

MODEL = {
    "hidden_dim": 64,
    "num_gru_layers": 2,
    "num_classes": 3,
    "dropout": 0.3,
    "activation": "tanh",
}

TRAIN = {
    "batch_size": 12000,
    "lr": 1e-3,
    "max_epochs": 100,
    "patience": 10,
    "val_metric": "tsrankic",
}

ROLLING_YEARS = list(range(2020, 2027))

LABEL = {
    "forecast_horizon": 1,
    "n_classes": 3,
    "class_ratios": [0.3194, 0.4159, 0.2647],
}

FEATURES = {
    "lookback_window": 20,
    "standardize": False,
}

BACKTEST = {
    "trade_price": "open",
    "commission_rate": 0.0005,
    "position_map": {0: 0.0, 1: 0.5, 2: 1.0},
}

GRAD_CLIP = 1.0
RANDOM_SEED = 42
