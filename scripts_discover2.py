import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import dolphindb as ddb
s = ddb.session()
s.connect('10.65.0.75', 8902, 'admin', '123456')

print("=== ts_daily (stock daily) ===")
r = s.run('select top 3 * from loadTable("dfs://d0_day", "ts_daily")')
print(r.columns.tolist())
print(r)
print("Rows:", s.run('exec count(*) from loadTable("dfs://d0_day", "ts_daily")'))

print("\n=== ts_daily_basic ===")
r = s.run('select top 3 * from loadTable("dfs://d0_day", "ts_daily_basic")')
print(r.columns.tolist())
print(r)
print("Rows:", s.run('exec count(*) from loadTable("dfs://d0_day", "ts_daily_basic")'))

print("\n=== jy_qt_indexquote (index daily) ===")
r = s.run('select top 5 * from loadTable("dfs://d0_day", "jy_qt_indexquote")')
print(r.columns.tolist())
print(r)

print("\n=== jy_secumain (code mapping) ===")
r = s.run('select top 10 * from loadTable("dfs://d0_day", "jy_secumain")')
print(r.columns.tolist())
if 'SecuCode' in r.columns:
    print(r[['SecuCode','InnerCode','SecuAbbr','ChiName']].to_string())
else:
    print(r)

print("\n=== factor_analysis table count ===")
tables = s.run('listTables("dfs://factor_analysis")')
print(f"Total factor tables: {len(tables)}")
print(f"Sample: {tables['tableName'].tolist()[:10]}")

print("\n=== check 000985 in index quote ===")
try:
    r = s.run("""
        select top 5 * from loadTable("dfs://d0_day", "jy_qt_indexquote")
        where TradingDay > 2025.01.01
        order by TradingDay
    """)
    print(r)
except Exception as e:
    print(f"FAIL: {e}")

s.close()
