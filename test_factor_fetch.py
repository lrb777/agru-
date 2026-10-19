import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, r"D:\agru择时")
from src.data_loader import DolphinDBLoader

loader = DolphinDBLoader()

print("Testing factor fetch (1 week)...")
df = loader.fetch_factors_wide("2024-01-02", "2024-01-08")
print(f"Rows: {len(df)}")
print(f"Cols: {len(df.columns)} (2 meta + {len(df.columns)-2} factors)")
print(f"Factor names: {list(df.columns[2:12])}...")
print(df.head(5))
loader.close()
