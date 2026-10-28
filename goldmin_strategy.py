# -*- coding: utf-8 -*-
"""
AGRU 神经网络大盘择时策略 — 掘金量化终端版

策略逻辑:
  1. 从 signals.csv 加载预计算的择时信号 (0=空仓 / 1=半仓 / 2=满仓)
  2. T日收盘生成信号 → T+1开盘前调仓
  3. 信号后移1天: 用最近一个交易日的信号指导当日仓位
  4. 开盘价市价成交, 单边万五手续费

交易标的: 中证全指ETF (SHSE.510980)
回测区间: 2020-01-01 ~ 2025-09-30

使用方式:
  1. 将 signals.csv 复制到掘金终端策略目录
  2. 修改 TOKEN 和 STRATEGY_ID
  3. 在掘金终端中运行回测
"""

from gm.api import *
import pandas as pd
import numpy as np
import os
import logging

# ======================== 策略参数 ========================

TOKEN = "your_token_here"
STRATEGY_ID = "agru_timing_strategy"

SYMBOL = "SHSE.510980"          # 中证全指ETF
SIGNALS_FILE = "signals.csv"    # 信号文件路径
INITIAL_CASH = 1_000_000        # 初始资金 100万
COMMISSION_RATE = 0.0005        # 单边万五
SLIPPAGE = 0.0                  # 无滑点

POSITION_MAP = {0: 0.0, 1: 0.5, 2: 1.0}   # 信号 → 目标仓位
SIGNAL_LABELS = {0: "空仓", 1: "半仓", 2: "满仓"}

BT_START = "2020-01-01 09:00:00"
BT_END = "2025-09-30 15:00:00"

# ========================================================

logger = logging.getLogger(__name__)


def init(context):
    """策略初始化"""
    context.symbol = SYMBOL
    context.position_map = POSITION_MAP

    # 加载信号文件
    signals_path = SIGNALS_FILE
    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"信号文件不存在: {signals_path}")

    df = pd.read_csv(signals_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 构建排序数组, 用于快速查找最近信号
    context.signal_dates = df["date"].values
    context.signal_values = df["signal"].values
    context.last_signal = 1  # 默认半仓
    context.last_signal_date = None

    logger.info(f"信号加载完成: {len(df)} 天, {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 订阅日K行情
    subscribe(symbols=[context.symbol], frequency="1d", count=1)

    # 每日 09:30 (开盘时) 调仓
    schedule(schedule_func=trade_at_open, date_rule="1d", time_rule="09:30")

    logger.info(f"策略初始化完成 | 标的: {context.symbol} | 初始资金: {INITIAL_CASH:,.0f}")


def trade_at_open(context):
    """每日开盘调仓"""
    today = context.now.normalize()

    # 查找今日之前最近的信号 (T日信号 → T+1执行)
    idx = np.searchsorted(context.signal_dates, today, side="right") - 1
    if idx >= 0:
        new_signal = int(context.signal_values[idx])
        signal_date = pd.Timestamp(context.signal_dates[idx])
    else:
        new_signal = context.last_signal
        signal_date = context.last_signal_date

    # 信号变化时调仓
    if new_signal != context.last_signal or context.last_signal_date is None:
        target_pos = context.position_map.get(new_signal, 0.5)
        order_target_percent(
            symbol=context.symbol,
            percent=target_pos,
            position_side=PositionSide_Long,
            order_type=OrderType_Market,
        )
        logger.info(
            f"{today.strftime('%Y-%m-%d')} | "
            f"信号={new_signal}({SIGNAL_LABELS[new_signal]}) | "
            f"信号日期={signal_date.strftime('%Y-%m-%d') if signal_date else 'N/A'} | "
            f"目标仓位={target_pos:.0%}"
        )
        context.last_signal = new_signal
        context.last_signal_date = signal_date


def on_bar(context, bars):
    """K线事件 (本策略不使用, 调仓在 schedule 中完成)"""
    pass


def on_backtest_finished(context, indicator):
    """回测结束, 输出统计"""
    logger.info("=" * 60)
    logger.info("回测完成")
    logger.info(f"  累计收益率:   {indicator['pnl_ratio']:.2%}")
    logger.info(f"  年化收益率:   {indicator['pnl_ratio_annual']:.2%}")
    logger.info(f"  最大回撤:     {indicator['max_drawdown']:.2%}")
    logger.info(f"  夏普比率:     {indicator['sharp_ratio']:.4f}")
    logger.info(f"  交易次数:     {indicator['trade_count']}")
    logger.info(f"  胜率:         {indicator['win_rate']:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run(
        strategy_id=STRATEGY_ID,
        filename="goldmin_strategy.py",
        mode=MODE_BACKTEST,
        token=TOKEN,
        backtest_start_time=BT_START,
        backtest_end_time=BT_END,
        backtest_initial_cash=INITIAL_CASH,
        backtest_transaction_ratio=1.0,
        backtest_adjust=ADJUST_PREV,          # 前复权
        backtest_slippage_ratio=SLIPPAGE,
        backtest_commission_ratio=COMMISSION_RATE,
    )
