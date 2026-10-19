import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, r"D:\agru择时")
from src.data_loader import DolphinDBLoader

loader = DolphinDBLoader()

print("Checking factor date range...")
r = loader.session.run("""
    select min(trade_date) as min_date, max(trade_date) as max_date
    from loadTable("dfs://d2_day", "factors_std")
    where pool = "filtered"
""")
print(r)

print("\nChecking 2025 data...")
r = loader.session.run("""
    select count(*) as cnt, min(trade_date) as min_d, max(trade_date) as max_d
    from loadTable("dfs://d2_day", "factors_std")
    where pool = "filtered" and trade_date >= 2025.01.01
""")
print(r)

print("\nRemoving 0-byte files...")
data_dir = r"D:\agru择时\data"
for f in os.listdir(data_dir):
    if f.startswith("factors_") and f.endswith(".parquet"):
        path = os.path.join(data_dir, f)
        if os.path.getsize(path) == 0:
            os.remove(path)
            print(f"  removed {f}")

print("\nFetching 2025...")
df = loader.fetch_factors_wide("2025-01-01", "2025-09-30")
cache = os.path.join(data_dir, "factors_2025.parquet")
df.to_parquet(cache, index=False)
print(f"2025: {len(df):,} rows, {len(df.columns)-2} factors, {os.path.getsize(cache)/1e6:.1f} MB")

loader.close()
