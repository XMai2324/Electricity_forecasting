import re
import pandas as pd


def _weekofyear(idx: pd.DatetimeIndex) -> pd.Series:
    # pandas version compatibility
    return idx.isocalendar().week.astype(int)


def build_feature_row(ts: pd.Timestamp, history: pd.Series, feature_names: list[str]) -> dict:
    row = {}

    # time features
    if "hour" in feature_names:
        row["hour"] = ts.hour
    if "dayofweek" in feature_names:
        row["dayofweek"] = ts.dayofweek
    if "day" in feature_names:
        row["day"] = ts.day
    if "month" in feature_names:
        row["month"] = ts.month
    if "weekofyear" in feature_names:
        row["weekofyear"] = int(ts.isocalendar().week)

    # lag_k
    for name in feature_names:
        m = re.fullmatch(r"lag_(\d+)", name)
        if m:
            k = int(m.group(1))
            if len(history) < k:
                row[name] = float("nan")
            else:
                row[name] = float(history.iloc[-k])

    # rolling
    for name in feature_names:
        m_mean = re.fullmatch(r"roll_mean_(\d+)", name)
        m_std = re.fullmatch(r"roll_std_(\d+)", name)

        if m_mean:
            w = int(m_mean.group(1))
            if len(history) < w:
                row[name] = float("nan")
            else:
                row[name] = float(history.iloc[-w:].mean())

        if m_std:
            w = int(m_std.group(1))
            if len(history) < w:
                row[name] = float("nan")
            else:
                row[name] = float(history.iloc[-w:].std(ddof=0))

    return row