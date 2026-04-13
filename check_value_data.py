import pandas as pd; from pathlib import Path; 

path = Path('data/sample/PJME_hourly.csv'); 
df = pd.read_csv(path);
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce'); 
df = df.dropna(subset=['Datetime']).sort_values('Datetime'); 
full_range = pd.date_range(start=df['Datetime'].min(), 
                           end=df['Datetime'].max(), freq='h'); orig = len(df); use = len(df.drop_duplicates(subset=['Datetime'], keep='last')); 
missing = len(full_range) - len(set(df['Datetime'])); 
dup_full = df['Datetime'].duplicated(keep=False).sum(); 
print(f"orig={orig}");
print(f"use={use}"); print(f"missing={missing}"); 
print(f"dup_full={dup_full}"); 
print(f"unique_times={len(set(df['Datetime']))}");
print(f"full_range={len(full_range)}")