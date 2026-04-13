import pandas as pd
import os

# File cần kiểm tra
file_path = 'data/sample/PJME_hourly.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime'])

        # Tạo range đầy đủ từ min đến max với freq='h'
        full_range = pd.date_range(
            start=df['Datetime'].min(),
            end=df['Datetime'].max(),
            freq='h'
        )

        # Set các mốc có trong dữ liệu
        existing_times = set(df['Datetime'])

        # Tìm các mốc thiếu
        missing_times = [ts for ts in full_range if ts not in existing_times]

        print(f"Tổng số mốc thời gian từ {df['Datetime'].min()} đến {df['Datetime'].max()}: {len(full_range)}")
        print(f"Số mốc có trong dữ liệu: {len(existing_times)}")
        print(f"Số mốc bị thiếu: {len(missing_times)}")

        if missing_times:
            print("\nCác mốc thời gian bị thiếu:")
            for ts in missing_times[:20]:  # Hiển thị 20 đầu
                print(ts)
            if len(missing_times) > 20:
                print(f"... và {len(missing_times) - 20} mốc khác")
    else:
        print("File không có cột 'Datetime'")
else:
    print(f"File không tồn tại: {file_path}")