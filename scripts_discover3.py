import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import dolphindb as ddb
s = ddb.session()
s.connect('10.65.0.75', 8902, 'admin', '123456')

print("=== Find 中证全指 InnerCode ===")
r = s.run("""
    select SecuCode, InnerCode, SecuAbbr, ChiName from loadTable("dfs://d0_day", "jy_secumain")
    where SecuCode in ["000985", "000300", "000852"]
""")
print(r.to_string())

print("\n=== Index quote for 000985 ===")
r = s.run("""
    t = loadTable("dfs://d0_day", "jy_qt_indexquote")
    s = loadTable("dfs://d0_day", "jy_secumain")
    select top 5 s.SecuCode, t.TradingDay, t.OpenPrice, t.ClosePrice, t.HighPrice, t.LowPrice
    from t join s on t.InnerCode = s.InnerCode
    where s.SecuCode = "000985" and t.TradingDay > 2025.01.01
    order by t.TradingDay
""")
print(r.to_string())

print("\n=== Check alpha158 in each DB ===")
for db in ['dfs://d1_day', 'dfs://d2_day', 'dfs://rangedb', 'dfs://valuedb', 'dfs://dataAcquisition']:
    try:
        tables = s.run(f'listTables("{db}")')
        names = tables['tableName'].tolist()
        alpha_tables = [t for t in names if 'alpha' in t.lower() or 'cicc' in t.lower() or 'factor' in t.lower()]
        if alpha_tables:
            print(f"\n{db}: {len(alpha_tables)} factor tables (of {len(names)} total)")
            print(f"  Sample: {alpha_tables[:15]}")
            cnt = s.run(f'exec count(*) from loadTable("{db}", "{alpha_tables[0]}")')
            print(f"  {alpha_tables[0]} rows: {cnt}")
            sample = s.run(f'select top 3 * from loadTable("{db}", "{alpha_tables[0]}")')
            print(f"  Cols: {sample.columns.tolist()}")
            print(sample.to_string())
    except Exception as e:
        print(f"{db}: FAIL {str(e)[:100]}")

s.close()
