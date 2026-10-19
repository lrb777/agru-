import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import dolphindb as ddb
s = ddb.session()
s.connect('10.65.0.75', 8902, 'admin', '123456')

print("=== factor_info ===")
r = s.run('select top 20 * from loadTable("dfs://d2_day", "factor_info")')
print(r.columns.tolist())
print(r.to_string())
print("Rows:", s.run('exec count(*) from loadTable("dfs://d2_day", "factor_info")'))

print("\n=== distinct pools in factors_std ===")
r = s.run('exec distinct pool from loadTable("dfs://d2_day", "factors_std")')
print(f"Pools: {list(r)}")

print("\n=== distinct factor_names (sample) ===")
r = s.run('exec distinct factor_name from loadTable("dfs://d2_day", "factors_std")')
names = list(r)
print(f"Total factor_names: {len(names)}")
print(f"First 30: {names[:30]}")
print(f"Last 10: {names[-10:]}")

print("\n=== factors_raw check ===")
r = s.run('select top 3 * from loadTable("dfs://d2_day", "factors_raw")')
print(r.columns.tolist())
print(r.to_string())
print("Rows:", s.run('exec count(*) from loadTable("dfs://d2_day", "factors_raw")'))

print("\n=== distinct pools in factors_raw ===")
r = s.run('exec distinct pool from loadTable("dfs://d2_day", "factors_raw")')
print(f"Pools: {list(r)}")

print("\n=== date range in factors_std ===")
r = s.run('exec min(trade_date), max(trade_date) from loadTable("dfs://d2_day", "factors_std")')
print(r)

s.close()
