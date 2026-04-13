import pandas as pd
import numpy as np
from typing import Tuple, Dict


def preprocess_csv(
    file_path: str,
    time_col: str,
    target_col: str,
    treat_zero_as_missing: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Tiền xử lý file CSV và trả về:
    1) DataFrame đã làm sạch, có DatetimeIndex liên tục theo giờ
    2) Metadata thống kê quá trình xử lý

    Parameters
    ----------
    file_path : str
        Đường dẫn file CSV
    time_col : str
        Tên cột thời gian
    target_col : str
        Tên cột mục tiêu
    treat_zero_as_missing : bool, default=False
        Nếu True thì giá trị 0 ở target sẽ được xem là thiếu và nội suy lại

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        df đã xử lý và metadata
    """

    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    raw_lines = len(df)

    # Kiểm tra cột bắt buộc
    if time_col not in df.columns:
        raise ValueError(f"Thiếu cột thời gian: {time_col}")
    if target_col not in df.columns:
        raise ValueError(f"Thiếu cột target: {target_col}")

    # Chuẩn hóa cột thời gian
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    invalid_time_count = df[time_col].isna().sum()
    df = df.dropna(subset=[time_col]).copy()
    after_parse = len(df)

    if df.empty:
        raise ValueError("Cột thời gian không có dữ liệu hợp lệ sau khi chuyển sang datetime.")

    # Sắp xếp theo thời gian
    df = df.sort_values(time_col).copy()

    # Chuẩn hóa cột target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    target_nan_before_fill = int(df[target_col].isna().sum())
    zero_count = int((df[target_col] == 0).sum())

    if treat_zero_as_missing:
        df[target_col] = df[target_col].replace(0, np.nan)

    # Loại trùng thời gian, giữ dòng cuối cùng
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[time_col], keep="last").copy()
    dup_removed = before_dedup - len(df)

    # Đưa time_col thành index
    df = df.set_index(time_col).sort_index()
    df.index.name = time_col

    if df.empty or df.index.isna().all():
        raise ValueError("Không còn dữ liệu hợp lệ sau khi loại bản ghi lỗi và trùng lặp.")

    # Tạo dải thời gian đầy đủ theo giờ
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=pd.Timedelta(hours=1)
    )
    full_range_len = len(full_range)

    # Số mốc giờ thiếu trong dữ liệu gốc
    missing_count = len(full_range.difference(df.index))

    # Reindex để tạo chuỗi liên tục theo giờ
    df = df.reindex(full_range)
    df.index.name = time_col

    # Nội suy dữ liệu thiếu theo thời gian
    df[target_col] = df[target_col].interpolate(method="time")

    # Nếu đầu hoặc cuối chuỗi vẫn còn NaN thì lấp bằng giá trị gần nhất
    df[target_col] = df[target_col].ffill().bfill()

    # Nếu vẫn còn NaN thì bỏ
    df = df.dropna(subset=[target_col]).copy()

    final_lines = len(df)
    remaining_nan_after_fill = int(df[target_col].isna().sum())

    metadata = {
        "raw_lines": int(raw_lines),
        "after_parse": int(after_parse),
        "invalid_time_count": int(invalid_time_count),
        "dup_removed": int(dup_removed),
        "full_range_len": int(full_range_len),
        "missing_count": int(missing_count),
        "final_lines": int(final_lines),
        "target_nan_before_fill": int(target_nan_before_fill),
        "zero_count": int(zero_count),
        "treat_zero_as_missing": bool(treat_zero_as_missing),
        "remaining_nan_after_fill": int(remaining_nan_after_fill),
    }

    return df, metadata