import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocess import preprocess_csv

# ===================== TEST 1: Data nguyên gốc =====================
print("=" * 60)
print("TEST 1: Phân tích file gốc")
print("=" * 60)

raw_path = 'data/sample/PJME_hourly.csv'
raw_df = pd.read_csv(raw_path)
print(f"Số dòng gốc từ file: {len(raw_df)}")
print(f"Cột: {list(raw_df.columns)}")

# ===================== TEST 2: Dữ liệu sau parse time =====================
print("\n" + "=" * 60)
print("TEST 2: Sau khi parse Datetime")
print("=" * 60)

df_time = raw_df.copy()
df_time['Datetime'] = pd.to_datetime(df_time['Datetime'], errors='coerce')
print(f"Dòng có NaT (datetime lỗi): {df_time['Datetime'].isna().sum()}")

df_time = df_time.dropna(subset=['Datetime']).sort_values('Datetime')
print(f"Dòng còn lại sau dropna: {len(df_time)}")

# ===================== TEST 3: Dữ liệu trùng lặp =====================
print("\n" + "=" * 60)
print("TEST 3: Phân tích thời gian trùng lặp")
print("=" * 60)

dup_count_default = df_time['Datetime'].duplicated().sum()
dup_count_keep_false = df_time['Datetime'].duplicated(keep=False).sum()
unique_times = df_time['Datetime'].nunique()

print(f"Dòng trùng (keep='first'): {dup_count_default}")
print(f"Dòng trùng (keep=False): {dup_count_keep_false}")
print(f"Mốc thời gian duy nhất: {unique_times}")

if dup_count_keep_false > 0:
    dup_df = df_time[df_time['Datetime'].duplicated(keep=False)]
    print(f"\nTop 10 dòng trùng:")
    print(dup_df[['Datetime', 'PJME_MW']].head(10).to_string(index=False))

# ===================== TEST 4: Thời gian thiếu =====================
print("\n" + "=" * 60)
print("TEST 4: Phân tích thời gian thiếu")
print("=" * 60)

full_range = pd.date_range(
    start=df_time['Datetime'].min(),
    end=df_time['Datetime'].max(),
    freq='h'
)
print(f"Dải giờ đầy đủ từ {df_time['Datetime'].min()} đến {df_time['Datetime'].max()}")
print(f"Số mốc trong dải: {len(full_range)}")
print(f"Số mốc thiếu (full_range - unique_times): {len(full_range) - unique_times}")

# ===================== TEST 5: Sau preprocess_csv =====================
print("\n" + "=" * 60)
print("TEST 5: Sau preprocess_csv")
print("=" * 60)

try:
    df_processed, meta = preprocess_csv(raw_path, 'Datetime', 'PJME_MW')
    print(f"\nMetadata từ preprocess_csv:")
    for key, val in meta.items():
        print(f"  {key}: {val:,}")
    print(f"\nSố dòng kết quả: {len(df_processed)}")
    print(f"Cột: {list(df_processed.columns)}")
    print(f"Index type: {type(df_processed.index)}")
    print(f"Có NaN trong PJME_MW: {df_processed['PJME_MW'].isna().sum()}")
    print(f"Min datetime: {df_processed.index.min()}")
    print(f"Max datetime: {df_processed.index.max()}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# ===================== TEST 6: Tóm tắt =====================
print("\n" + "=" * 60)
print("TÓM TẮT - So sánh con số")
print("=" * 60)

print(f"orig       = {len(raw_df):>8}  (dòng gốc từ file)")
print(f"raw_parse  = {len(df_time):>8}  (sau parse Datetime + dropna)")
print(f"unique_times = {unique_times:>8}  (mốc thời gian duy nhất)")
print(f"dup_full   = {dup_count_keep_false:>8}  (dòng trùng với keep=False)")
print(f"full_range = {len(full_range):>8}  (dải giờ hoàn chỉnh)")
print(f"missing    = {len(full_range) - unique_times:>8}  (mốc giờ thiếu)")

if 'df_processed' in locals():
    print(f"\nuse        = {len(df_processed):>8}  (sau preprocess)")
    expected_use = len(full_range) - (len(full_range) - unique_times)
    print(f"expected_use = {expected_use:>8}  (full_range - missing = dùng được)")
