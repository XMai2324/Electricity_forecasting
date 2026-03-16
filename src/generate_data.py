import pandas as pd
import numpy as np
import os

def generate_electricity_data(output_file, years=5):
    """
    Generate dữ liệu tiêu thụ điện Hà Nội giả lập 5 năm gần nhất.
    Pattern dựa trên PJME nhưng điều chỉnh cho Hà Nội.
    """
    # Tạo dải thời gian 5 năm gần nhất (2021-2025)
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2025-12-31 23:00:00')
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Tạo base consumption với seasonality
    np.random.seed(42)
    n_hours = len(date_range)
    
    # Base load với trend tăng dần
    base_load = 25000 + np.linspace(0, 5000, n_hours)  # Tăng từ 25k đến 30k MW
    
    # Seasonal theo giờ (giống PJME nhưng điều chỉnh)
    hour_seasonal = np.array([
        0.6, 0.55, 0.5, 0.45, 0.4, 0.45,  # 0-5h: thấp
        0.6, 0.8, 1.0, 1.1, 1.2, 1.3,  # 6-11h: tăng
        1.4, 1.5, 1.6, 1.7, 1.8, 1.9,  # 12-17h: cao
        2.0, 1.8, 1.6, 1.3, 1.0, 0.8   # 18-23h: giảm
    ])
    
    # Seasonal theo tháng (mùa hè cao hơn)
    month_seasonal = np.array([
        0.9, 0.85, 0.8, 0.75, 0.8, 0.9,  # Jan-Jun
        1.2, 1.3, 1.4, 1.2, 1.0, 0.95   # Jul-Dec
    ])
    
    # Tạo consumption
    consumption = []
    for i, dt in enumerate(date_range):
        hour_factor = hour_seasonal[dt.hour]
        month_factor = month_seasonal[dt.month - 1]
        day_factor = 1.1 if dt.weekday() < 5 else 0.9  # Ngày thường cao hơn cuối tuần
        
        load = base_load[i] * hour_factor * month_factor * day_factor
        load += np.random.normal(0, 500)  # Noise
        
        consumption.append(max(0, load))  # Đảm bảo không âm
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'Datetime': date_range,
        'PJME_MW': consumption
    })
    
    # Thêm một số giá trị 0 và missing để test preprocess
    zero_indices = np.random.choice(len(df), size=int(0.01 * len(df)), replace=False)
    df.loc[zero_indices, 'PJME_MW'] = 0
    
    nan_indices = np.random.choice(len(df), size=int(0.005 * len(df)), replace=False)
    df.loc[nan_indices, 'PJME_MW'] = np.nan
    
    # Lưu file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} rows of Saigon electricity data to {output_file}")
    
    return df

if __name__ == "__main__":
    output_file = "uploads/raw/saigon_electricity.csv"
    generate_electricity_data(output_file, years=5)