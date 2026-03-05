from __future__ import annotations

import pandas as pd

from io_untils import load_model, load_feature_config
from features import build_feature_row


def _infer_freq(index: pd.DatetimeIndex) -> str:
    freq = pd.infer_freq(index)
    if freq:
        return freq

    # fallback: lấy median delta
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return "H"
    med = deltas.median()
    # quy về 1H hoặc 1D nếu gần nhất
    if abs(med - pd.Timedelta(hours=1)) <= pd.Timedelta(minutes=5):
        return "H"
    if abs(med - pd.Timedelta(days=1)) <= pd.Timedelta(minutes=30):
        return "D"
    # nếu lạ thì dùng median seconds
    secs = int(med.total_seconds())
    return f"{secs}S"


def forecast_by_date(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    start_date: str,
    end_date: str,
    model_path: str = "artifacts/model.pkl",
    feature_config_path: str = "artifacts/feature_config.json",
) -> pd.DataFrame:
    cfg = load_feature_config(feature_config_path)
    feature_names = cfg.get("features") or cfg.get("feature_names")
    if not feature_names:
        raise ValueError("feature_config.json thiếu key features (hoặc feature_names).")

    model = load_model(model_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df phải có index là DatetimeIndex (đã set_index cột thời gian).")

    df = df.sort_index()
    last_ts = df.index.max()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date phải lớn hơn hoặc bằng start_date.")

    # Nếu user chọn start_date trước hoặc bằng dữ liệu lịch sử, ta vẫn dự báo từ max(last_ts, start_ts)
    run_start = max(last_ts, start_ts)

    freq = cfg.get("frequency_hint") or _infer_freq(df.index)

    # tạo future index, bắt đầu sau last_ts 1 bước
    first_future = last_ts + pd.tseries.frequencies.to_offset(freq)
    future_index = pd.date_range(start=first_future, end=end_ts, freq=freq)

    # Nếu user chọn end_date <= last_ts thì không có tương lai để dự báo
    if len(future_index) == 0:
        return pd.DataFrame(columns=["Datetime", "yhat"])

    # Lịch sử để tạo lag/rolling
    history = df[target_col].astype(float).copy()

    preds = []
    for ts in future_index:
        row_dict = build_feature_row(ts, history, feature_names)
        X_one = pd.DataFrame([row_dict], columns=feature_names)

        # Nếu vẫn còn NaN do thiếu lịch sử, fill theo cách an toàn
        X_one = X_one.fillna(method="ffill", axis=1).fillna(0)

        yhat = float(model.predict(X_one)[0])
        preds.append((ts, yhat))

        # cập nhật history để bước sau có lag/rolling
        history = pd.concat([history, pd.Series([yhat], index=[ts])])

    out = pd.DataFrame(preds, columns=["Datetime", "yhat"])

    # lọc theo khoảng user muốn (start_date -> end_date)
    out = out[out["Datetime"] >= start_ts]
    out = out[out["Datetime"] <= end_ts]
    out = out.reset_index(drop=True)
    return out