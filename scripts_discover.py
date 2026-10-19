import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import dolphindb as ddb
import pandas as pd

s = ddb.session()
s.connect('10.65.0.75', 8902, 'admin', '123456')

dbs = [
    'dfs://d0_day', 'dfs://d1_day', 'dfs://d2_day',
    'dfs://valuedb', 'dfs://factor_analysis', 'dfs://rangedb',
    'dfs://dataAcquisition', 'dfs://test_d2_day',
]

for db in dbs:
    print(f"\n{'='*60}")
    print(f"Database: {db}")
    print(f"{'='*60}")

    try:
        tables_df = s.run(f'listTables("{db}")')
        if isinstance(tables_df, pd.DataFrame):
            table_names = tables_df['tableName'].tolist()
        else:
            table_names = list(tables_df)
        print(f"  Tables: {table_names}")

        for t in table_names:
            print(f"\n  --- {t} ---")
            try:
                cols = s.run(f'schema(loadTable("{db}", "{t}")).colDefs')
                if isinstance(cols, pd.DataFrame):
                    col_info = [(row['name'], row['typeString']) for _, row in cols.iterrows()]
                    print(f"    Cols: {col_info}")
            except Exception as e:
                print(f"    schema FAIL: {str(e)[:120]}")

            try:
                cnt = s.run(f'exec count(*) from loadTable("{db}", "{t}")')
                print(f"    Rows: {cnt:,}")
            except Exception as e:
                print(f"    count FAIL: {str(e)[:120]}")

            try:
                sample = s.run(f'select top 2 * from loadTable("{db}", "{t}")')
                print(f"    Sample:")
                print(sample.to_string())
            except Exception as e:
                print(f"    sample FAIL: {str(e)[:120]}")
    except Exception as e:
        print(f"  listTables FAIL: {str(e)[:200]}")

s.close()
