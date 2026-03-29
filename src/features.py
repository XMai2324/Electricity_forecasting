import re
import math
import numpy as np
import pandas as pd

SEASON_MAP = {
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
    12: "Winter", 1: "Winter", 2: "Winter",
}

VN_FIXED_HOLIDAYS = {
    (1, 1),
    (4, 30),
    (5, 1),
    (9, 2),
}


def _base_time_features_from_ts(ts: pd.Timestamp) -> dict:
    hour = ts.hour
    dayofweek = ts.dayofweek

    return {
        "hour": hour,
        "dayofweek": dayofweek,
        "day": ts.day,
        "month": ts.month,
        "year": ts.year,
        "weekofyear": int(ts.isocalendar().week),
        "is_weekend": 1 if dayofweek >= 5 else 0,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dayofweek / 7),
        "dow_cos": math.cos(2 * math.pi * dayofweek / 7),
    }


def add_common_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy().sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("df phải có DatetimeIndex.")

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[target_col])

    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["day"] = idx.day
    out["month"] = idx.month
    out["year"] = idx.year
    out["weekofyear"] = idx.isocalendar().week.astype(int)

    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def add_analysis_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = add_common_features(df, target_col)

    out["day_name"] = out.index.day_name()
    out["season"] = out["month"].map(SEASON_MAP)
    out["is_holiday"] = [(ts.month, ts.day) in VN_FIXED_HOLIDAYS for ts in out.index]
    out["day_type"] = np.where(
        out["is_holiday"],
        "Holiday",
        np.where(out["is_weekend"] == 1, "Weekend", "Weekday")
    )

    return out


def build_feature_row(
    ts: pd.Timestamp,
    history: pd.Series,
    feature_names: list[str]
) -> dict:
    base = _base_time_features_from_ts(ts)
    row = {name: base[name] for name in feature_names if name in base}

    for name in feature_names:
        m = re.fullmatch(r"lag_(\d+)", name)
        if m:
            k = int(m.group(1))
            row[name] = float("nan") if len(history) < k else float(history.iloc[-k])

    for name in feature_names:
        m_mean = re.fullmatch(r"roll_mean_(\d+)", name)
        m_std = re.fullmatch(r"roll_std_(\d+)", name)

        if m_mean:
            w = int(m_mean.group(1))
            row[name] = float("nan") if len(history) < w else float(history.iloc[-w:].mean())

        if m_std:
            w = int(m_std.group(1))
            row[name] = float("nan") if len(history) < w else float(history.iloc[-w:].std(ddof=0))

    return row