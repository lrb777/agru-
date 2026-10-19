import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, r"D:\agru择时")
from src.data_loader import DolphinDBLoader

loader = DolphinDBLoader()

t0 = time.time()
df = loader.fetch_factors_wide("2020-01-01", "2020-12-31")
t1 = time.time()
print(f"2020: {len(df):,} rows, {len(df.columns)-2} factors, {t1-t0:.1f}s")

cache = os.path.join(r"D:\agru择时\data", "factors_2020.parquet")
df.to_parquet(cache, index=False)
print(f"Cached. File size: {os.path.getsize(cache)/1e6:.1f} MB")

loader.close()
