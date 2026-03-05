import pandas as pd


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
    df[target_col] = df[target_col].ffill().bfill()

    df = df.set_index(time_col)
    return df