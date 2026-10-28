"""DolphinDB 数据接口: 指数行情、预计算因子(260个alpha158+cicc+custom)"""
import os
import pandas as pd
from config import DOLPHINDB, DB_D0, DB_D2, INDEX_CODE, INDEX_INNER_CODE, DATA_START, DATA_END, FACTOR_POOL, DATA_DIR


class DolphinDBLoader:
    def __init__(self, **kwargs):
        cfg = {**DOLPHINDB, **kwargs}
        self.host = cfg["host"]
        self.port = cfg["port"]
        self.user = cfg["username"]
        self.pwd = cfg["password"]
        self._session = None

    @property
    def session(self):
        if self._session is None:
            # 本地缓存处理无需 DolphinDB，仅建立数据库会话时导入。
            import dolphindb as ddb
            self._session = ddb.session()
            self._session.connect(self.host, self.port, self.user, self.pwd)
        return self._session

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def fetch_index_daily(self, index_code: str = INDEX_CODE, start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
        script = f"""
            t = loadTable("{DB_D0}", "jy_qt_indexquote")
            s = loadTable("{DB_D0}", "jy_secumain")
            select s.SecuCode as code, t.TradingDay as date,
                   t.OpenPrice as open, t.HighPrice as high,
                   t.LowPrice as low, t.ClosePrice as close,
                   t.TurnoverVolume as volume, t.TurnoverValue as amount
            from t join s on t.InnerCode = s.InnerCode
            where s.SecuCode = "{index_code}"
              and t.TradingDay between {start.replace("-", ".")} and {end.replace("-", ".")}
            order by t.TradingDay
        """
        df = self.session.run(script)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def fetch_factor_list(self) -> list:
        script = f"""
            exec distinct factor_name from loadTable("{DB_D2}", "factors_std")
            where trade_date = 2024.01.02, pool = "{FACTOR_POOL}"
        """
        return list(self.session.run(script))

    def fetch_factors_wide(self, start: str, end: str, pool: str = FACTOR_POOL) -> pd.DataFrame:
        script = f"""
            t = loadTable("{DB_D2}", "factors_std")
            select factor_value from t
            where pool = "{pool}"
              and trade_date between {start.replace("-", ".")} and {end.replace("-", ".")}
            pivot by trade_date, ts_code, factor_name
        """
        df = self.session.run(script)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.rename(columns={"trade_date": "date"})
        df = df.rename(columns={"ts_code": "code"})
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        return df

    def fetch_factors_yearly(self, pool: str = FACTOR_POOL, start_year: int = 2015, end_year: int = None) -> pd.DataFrame:
        if end_year is None:
            end_year = pd.Timestamp(DATA_END).year
        all_dfs = []
        for yr in range(start_year, end_year + 1):
            cache = os.path.join(DATA_DIR, f"factors_{yr}.parquet")
            if os.path.exists(cache):
                df = pd.read_parquet(cache)
                print(f"  {yr}: cached {len(df):,} rows")
            else:
                s = f"{yr}-01-01"
                if yr == end_year:
                    e = DATA_END
                else:
                    e = f"{yr}-12-31"
                print(f"  {yr}: fetching from DolphinDB...")
                df = self.fetch_factors_wide(s, e, pool)
                df.to_parquet(cache, index=False)
                print(f"  {yr}: {len(df):,} rows, {len(df.columns)-2} factors -> cached")
            all_dfs.append(df)
        result = pd.concat(all_dfs, ignore_index=True)
        return result

    def fetch_all(self):
        print("=== 拉取指数行情 ===")
        idx = self.fetch_index_daily()
        cache = os.path.join(DATA_DIR, "index_daily.parquet")
        idx.to_parquet(cache, index=False)
        print(f"  指数: {len(idx):,} 行 -> {cache}")

        print("\n=== 拉取预计算因子 (逐年) ===")
        factors = self.fetch_factors_yearly()
        print(f"\n  因子总行数: {len(factors):,}, 因子数: {len(factors.columns)-2}")

        return idx, factors

    @staticmethod
    def load_local(columns=None):
        idx = pd.read_parquet(os.path.join(DATA_DIR, "index_daily.parquet"))
        factors_dir = DATA_DIR
        factor_files = sorted([
            os.path.join(factors_dir, f) for f in os.listdir(factors_dir)
            if f.startswith("factors_") and f.endswith(".parquet")
        ])
        parts = []
        for f in factor_files:
            df = pd.read_parquet(f)
            if columns is not None:
                keep = ["date", "code"] + [c for c in columns if c in df.columns]
                df = df[keep]
            parts.append(df)
        factors = pd.concat(parts, ignore_index=True)
        return idx, factors
