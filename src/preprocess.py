import pandas as pd
import numpy as np


def preprocess_csv(
    file_path: str,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if time_col not in df.columns:
        raise ValueError(f"Thiếu cột thời gian: {time_col}")
    if target_col not in df.columns:
        raise ValueError(f"Thiếu cột target: {target_col}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    df = df.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].replace(0, np.nan)
    df[target_col] = df[target_col].interpolate(method="linear")

    df = df.set_index(time_col)

    if df.empty or df.index.isna().all():
        raise ValueError("Cột thời gian không có dữ liệu hợp lệ sau khi chuyển sang datetime.")

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=pd.Timedelta(hours=1)
    )

    df = df.reindex(full_range)
    df.index.name = time_col
    df[target_col] = df[target_col].interpolate(method="time")
    df = df.dropna()

    return df