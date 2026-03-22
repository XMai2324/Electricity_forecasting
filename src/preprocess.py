import pandas as pd
import numpy as np


def preprocess_csv(
    file_path: str,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Làm sạch dữ liệu từ file CSV raw.
    Xử lý: giá trị 0, thiếu dữ liệu, trùng lặp, đảm bảo liên tục theo giờ.
    """
    df = pd.read_csv(file_path)

    if time_col not in df.columns:
        raise ValueError(f"Thiếu cột thời gian: {time_col}")
    if target_col not in df.columns:
        raise ValueError(f"Thiếu cột target: {target_col}")

    # Chuyển cột thời gian thành datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Loại bỏ trùng lặp: giữ lại bản ghi cuối cùng nếu có nhiều bản ghi cùng thời điểm
    df = df.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")

    # Chuyển target thành numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Xử lý giá trị 0: giả sử 0 là lỗi, thay bằng NaN
    df[target_col] = df[target_col].replace(0, np.nan)

    # Xử lý thiếu dữ liệu: interpolate linear
    df[target_col] = df[target_col].interpolate(method="linear")

    # Set index
    df = df.set_index(time_col)

    # Đảm bảo dữ liệu liên tục theo giờ (nếu thiếu giờ, thêm NaN rồi interpolate)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="H")
    df = df.reindex(full_range)
    df[target_col] = df[target_col].interpolate(method="time")

    # Loại bỏ NaN còn lại (nếu có)
    df = df.dropna()

    return df