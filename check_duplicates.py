import pandas as pd
import os
import glob

# List of CSV files to check
csv_files = glob.glob('D:\ĐACN\Electricity_forecasting\data\sample\PJME_hourly.csv')

for file_path in csv_files:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'Datetime' in df.columns:
                # Check for duplicates in 'Datetime' column
                duplicate_mask = df['Datetime'].duplicated(keep=False)
                duplicates = df[duplicate_mask]
                num_duplicates = len(duplicates)
                print(f"\nFile: {file_path}")
                print(f"Số dòng có thời gian trùng lặp: {num_duplicates}")
                if num_duplicates > 0:
                    print("Các dòng trùng lặp:")
                    print(duplicates[['Datetime', 'PJME_MW']].head(20))  # Show first 20 duplicates with relevant columns
                    if num_duplicates > 20:
                        print(f"... và {num_duplicates - 20} dòng khác")
            else:
                print(f"File {file_path} không có cột 'Datetime'")
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
    else:
        print(f"File không tồn tại: {file_path}")