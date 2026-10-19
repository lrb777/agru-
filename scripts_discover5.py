import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import dolphindb as ddb
s = ddb.session()
s.connect('10.65.0.75', 8902, 'admin', '123456')

print("=== factor_info ===")
r = s.run('select * from loadTable("dfs://d2_day", "factor_info")')
print(r.columns.tolist())
print(f"Rows: {len(r)}")
print(r.to_string())

print("\n=== factors_std sample (1 date) ===")
r = s.run("""
    select top 20 * from loadTable("dfs://d2_day", "factors_std")
    where trade_date = 2024.01.02
""")
print(r.columns.tolist())
print(r.to_string())

print("\n=== distinct pools (from 1 date) ===")
r = s.run('exec distinct pool from loadTable("dfs://d2_day", "factors_std") where trade_date = 2024.01.02')
print(f"Pools: {list(r)}")

print("\n=== distinct factor_names (from 1 date, 1 stock) ===")
r = s.run("""
    exec distinct factor_name from loadTable("dfs://d2_day", "factors_std")
    where trade_date = 2024.01.02 and ts_code = "000001.SZ"
""")
names = list(r)
print(f"Total: {len(names)}")
for n in names:
    print(f"  {n}")

s.close()
