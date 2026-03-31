import matplotlib.pyplot as plt
import sys
import os
import json
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

try:
    import shap
except ImportError:
    shap = None

from sklearn.ensemble import IsolationForest

from preprocess import preprocess_csv
from forecast import forecast_by_date
from features import add_analysis_features, build_feature_row

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Electricity Forecast XGBoost", layout="wide")
st.title("Electricity Analysis & Forecasting System")

TIME_COL = "Datetime"
TARGET_COL = "PJME_MW"

MODEL_PATH = ROOT_DIR / "artifacts" / "model.pkl"
CFG_PATH = ROOT_DIR / "artifacts" / "feature_config.json"

UPLOAD_RAW_DIR = ROOT_DIR / "uploads" / "raw"
UPLOAD_PROCESSED_DIR = ROOT_DIR / "uploads" / "processed"
OUTPUT_FORECAST_DIR = ROOT_DIR / "outputs" / "forecasts"

UPLOAD_RAW_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FORECAST_DIR.mkdir(parents=True, exist_ok=True)

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]

# ===================== SESSION STATE =====================
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "forecast_start_date" not in st.session_state:
    st.session_state.forecast_start_date = None
if "forecast_end_date" not in st.session_state:
    st.session_state.forecast_end_date = None
if "forecast_source_signature" not in st.session_state:
    st.session_state.forecast_source_signature = None
if "selected_forecast_idx" not in st.session_state:
    st.session_state.selected_forecast_idx = 0


# ===================== INSIGHT HELPERS =====================
def show_ai_insight(title: str, insights: list[str]) -> None:
    st.markdown(f"#### 🤖 {title}")
    for text in insights:
        st.write(f"- {text}")


def generate_trend_insights(series: pd.Series) -> list[str]:
    s = series.dropna()
    if s.empty:
        return ["Chưa đủ dữ liệu để tạo insight."]

    overall_avg = s.mean()
    peak_time = s.idxmax()
    low_time = s.idxmin()
    peak_value = s.max()
    low_value = s.min()

    first_avg = s.head(min(24 * 30, len(s))).mean()
    last_avg = s.tail(min(24 * 30, len(s))).mean()

    if first_avg != 0:
        pct_change = (last_avg - first_avg) / first_avg * 100
    else:
        pct_change = 0

    if pct_change > 5:
        trend_text = "xu hướng tăng"
    elif pct_change < -5:
        trend_text = "xu hướng giảm"
    else:
        trend_text = "xu hướng tương đối ổn định"

    return [
        f"Mức tiêu thụ điện nhìn chung có {trend_text} theo thời gian.",
        f"Giá trị trung bình toàn bộ chuỗi đạt khoảng {overall_avg:,.2f} MW.",
        f"Điểm cao nhất xuất hiện vào {peak_time} với khoảng {peak_value:,.2f} MW.",
        f"Điểm thấp nhất xuất hiện vào {low_time} với khoảng {low_value:,.2f} MW.",
    ]


def generate_hourly_insights(hourly_mean: pd.Series) -> list[str]:
    peak_hour = int(hourly_mean.idxmax())
    low_hour = int(hourly_mean.idxmin())
    peak_value = float(hourly_mean.max())
    low_value = float(hourly_mean.min())

    insights = [
        f"Khung giờ dùng điện cao nhất trung bình là {peak_hour}:00 với khoảng {peak_value:,.2f} MW.",
        f"Khung giờ dùng điện thấp nhất trung bình là {low_hour}:00 với khoảng {low_value:,.2f} MW.",
    ]

    if 17 <= peak_hour <= 21:
        insights.append("Nhu cầu điện tập trung mạnh vào buổi tối, phù hợp với thời điểm sinh hoạt cao.")
    elif 9 <= peak_hour <= 16:
        insights.append("Nhu cầu điện cao tập trung vào giờ làm việc ban ngày.")
    else:
        insights.append("Đỉnh tải xuất hiện ở khung giờ ít phổ biến hơn, nên xem thêm theo từng ngày cụ thể.")

    if low_value > 0 and peak_value / low_value >= 1.3:
        insights.append("Chênh lệch giữa giờ cao điểm và thấp điểm khá rõ, cho thấy phụ tải thay đổi mạnh trong ngày.")

    return insights


def generate_selected_day_insights(day_df: pd.DataFrame, target_col: str, selected_date) -> list[str]:
    valid = day_df[target_col].dropna()
    if valid.empty:
        return [f"Không có dữ liệu hợp lệ để phân tích ngày {selected_date}."]

    avg_day = valid.mean()
    max_time = valid.idxmax()
    min_time = valid.idxmin()
    max_val = valid.max()
    min_val = valid.min()

    return [
        f"Trong ngày {selected_date}, mức tiêu thụ trung bình đạt khoảng {avg_day:,.2f} MW.",
        f"Đỉnh trong ngày rơi vào {max_time.hour}:00 với khoảng {max_val:,.2f} MW.",
        f"Mức thấp nhất trong ngày rơi vào {min_time.hour}:00 với khoảng {min_val:,.2f} MW.",
    ]


def generate_monthly_insights(monthly_year_df: pd.DataFrame, target_col: str) -> list[str]:
    max_row = monthly_year_df.loc[monthly_year_df[target_col].idxmax()]
    min_row = monthly_year_df.loc[monthly_year_df[target_col].idxmin()]

    return [
        f"Tháng có mức tiêu thụ trung bình cao nhất là {int(max_row['month'])}/{int(max_row['year'])} với khoảng {max_row[target_col]:,.2f} MW.",
        f"Tháng có mức tiêu thụ trung bình thấp nhất là {int(min_row['month'])}/{int(min_row['year'])} với khoảng {min_row[target_col]:,.2f} MW.",
        "Sự khác biệt giữa các tháng cho thấy dữ liệu có tính mùa vụ khá rõ.",
    ]


def generate_season_insights(season_month_df: pd.DataFrame, target_col: str) -> list[str]:
    if season_month_df.empty:
        return ["Chưa đủ dữ liệu để phân tích theo mùa."]

    max_row = season_month_df.loc[season_month_df[target_col].idxmax()]
    min_row = season_month_df.loc[season_month_df[target_col].idxmin()]

    return [
        f"Giai đoạn theo mùa cao nhất nằm ở {max_row['season']} - tháng {int(max_row['month'])} với khoảng {max_row[target_col]:,.2f} MW.",
        f"Giai đoạn thấp nhất nằm ở {min_row['season']} - tháng {int(min_row['month'])} với khoảng {min_row[target_col]:,.2f} MW.",
        "Biểu đồ cho thấy nhu cầu điện thay đổi theo mùa, thường chịu ảnh hưởng của thời tiết và thói quen sử dụng điện.",
    ]


def generate_day_type_insights(day_type_mean: pd.Series) -> list[str]:
    insights = []

    if "Weekday" in day_type_mean.index and "Weekend" in day_type_mean.index:
        if day_type_mean["Weekday"] > day_type_mean["Weekend"]:
            insights.append("Ngày thường có mức tiêu thụ cao hơn cuối tuần, phản ánh hoạt động làm việc và sản xuất.")
        else:
            insights.append("Cuối tuần không thấp hơn ngày thường, cho thấy đặc điểm sử dụng điện của khu vực khá đặc biệt.")

    if "Holiday" in day_type_mean.index:
        insights.append(f"Ngày lễ có mức tiêu thụ trung bình khoảng {day_type_mean['Holiday']:,.2f} MW.")

    return insights if insights else ["Chưa đủ dữ liệu để tạo insight cho loại ngày."]


def generate_forecast_insights(fc: pd.DataFrame) -> list[str]:
    avg_value = float(fc["yhat"].mean())
    max_row = fc.loc[fc["yhat"].idxmax()]
    min_row = fc.loc[fc["yhat"].idxmin()]

    daily = fc.groupby(fc["Datetime"].dt.date)["yhat"].mean()
    hourly = fc.groupby(fc["Datetime"].dt.hour)["yhat"].mean()

    insights = [
        f"Giai đoạn dự báo có mức tiêu thụ trung bình khoảng {avg_value:,.2f} MW.",
        f"Điểm dự báo cao nhất xuất hiện vào {max_row['Datetime']} với khoảng {max_row['yhat']:,.2f} MW.",
        f"Điểm dự báo thấp nhất xuất hiện vào {min_row['Datetime']} với khoảng {min_row['yhat']:,.2f} MW.",
        f"Ngày có mức trung bình cao nhất là {daily.idxmax()}, còn ngày thấp nhất là {daily.idxmin()}.",
        f"Khung giờ dự báo cao nhất là {int(hourly.idxmax())}:00, khung giờ thấp nhất là {int(hourly.idxmin())}:00.",
    ]

    if max_row["yhat"] > avg_value * 1.1:
        insights.append("Có thời điểm phụ tải vượt khá xa mức trung bình, nên chú ý khi theo dõi hoặc điều phối công suất.")

    return insights


# ===================== SIMPLE SHAP HELPERS =====================
FEATURE_LABELS = {
    "hour": "Giờ trong ngày",
    "dayofweek": "Thứ trong tuần",
    "day": "Ngày trong tháng",
    "month": "Tháng",
    "year": "Năm",
    "weekofyear": "Tuần trong năm",
    "is_weekend": "Cuối tuần",
    "hour_sin": "Chu kỳ giờ sin",
    "hour_cos": "Chu kỳ giờ cos",
    "dow_sin": "Chu kỳ thứ sin",
    "dow_cos": "Chu kỳ thứ cos",
}


def get_feature_names_from_cfg(cfg: dict) -> list[str]:
    for key in ["feature_names", "features", "selected_features", "model_features"]:
        value = cfg.get(key)
        if isinstance(value, list) and len(value) > 0:
            return value
    raise ValueError("Không tìm thấy danh sách feature trong feature_config.json")


@st.cache_resource(show_spinner=False)
def load_model_and_feature_names(model_path, feature_config_path):
    model = joblib.load(model_path)
    with open(feature_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_names = get_feature_names_from_cfg(cfg)
    return model, feature_names


def human_feature_name(name: str) -> str:
    if name.startswith("lag_"):
        step = name.split("_")[1]
        return f"Nhu cầu trễ {step} giờ"
    if name.startswith("roll_mean_"):
        step = name.split("_")[2]
        return f"Trung bình trượt {step} giờ"
    if name.startswith("roll_std_"):
        step = name.split("_")[2]
        return f"Độ lệch chuẩn trượt {step} giờ"
    return FEATURE_LABELS.get(name, name)


def format_feature_value(v):
    if pd.isna(v):
        return "NaN"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return str(int(v))
        return f"{float(v):,.2f}"
    return str(v)


def build_forecast_feature_matrix(history_df, forecast_df, target_col, feature_names):
    history = history_df[target_col].dropna().astype(float).copy()
    rows = []

    for ts, pred in forecast_df[["Datetime", "yhat"]].itertuples(index=False, name=None):
        ts = pd.Timestamp(ts)
        row = build_feature_row(ts, history, feature_names)
        rows.append({"Datetime": ts, **row})
        history.loc[ts] = float(pred)

    feature_frame = pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)
    X_future = feature_frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feature_frame, X_future


def make_shap_explainer(model, X_background):
    if shap is None:
        raise ImportError("Chưa cài thư viện shap. Hãy chạy: pip install shap")

    try:
        return shap.TreeExplainer(model)
    except Exception:
        return shap.Explainer(model, X_background)


def explain_single_prediction_simple(shap_row, feature_row, pred_value):
    base_value = float(np.array(shap_row.base_values).reshape(-1)[0])

    impacts = pd.DataFrame({
        "feature": feature_row.index,
        "feature_name": [human_feature_name(col) for col in feature_row.index],
        "feature_value": [format_feature_value(v) for v in feature_row.values],
        "shap_value": shap_row.values
    })

    positive_df = impacts[impacts["shap_value"] > 0].sort_values("shap_value", ascending=False).head(3)
    negative_df = impacts[impacts["shap_value"] < 0].sort_values("shap_value", ascending=True).head(3)

    top_abs = impacts.copy()
    top_abs["abs_value"] = top_abs["shap_value"].abs()
    top_abs = top_abs.sort_values("abs_value", ascending=False).head(6)

    summary_parts = []

    if not positive_df.empty:
        inc_names = ", ".join(positive_df["feature_name"].tolist()[:2])
        summary_parts.append(f"các yếu tố làm tăng dự báo mạnh nhất là {inc_names}")

    if not negative_df.empty:
        dec_names = ", ".join(negative_df["feature_name"].tolist()[:2])
        summary_parts.append(f"các yếu tố làm giảm dự báo mạnh nhất là {dec_names}")

    if summary_parts:
        summary_text = (
            f"Tại thời điểm này, mô hình có giá trị gốc khoảng {base_value:,.2f} MW. "
            + ", đồng thời ".join(summary_parts)
            + f". Sau khi cộng trừ các tác động, giá trị dự báo cuối cùng là {pred_value:,.2f} MW."
        )
    else:
        summary_text = (
            f"Tại thời điểm này, mô hình có giá trị gốc khoảng {base_value:,.2f} MW "
            f"và dự báo cuối cùng là {pred_value:,.2f} MW."
        )

    return base_value, positive_df, negative_df, top_abs, summary_text


def plot_simple_shap_contributions(top_abs_df):
    plot_df = top_abs_df.copy().sort_values("shap_value")
    labels = [
        f"{row.feature_name}\n({row.feature_value})"
        for _, row in plot_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(labels, plot_df["shap_value"].values, alpha=0.8)

    for bar, value in zip(bars, plot_df["shap_value"].values):
        if value >= 0:
            bar.set_color("red")
        else:
            bar.set_color("green")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Các yếu tố đang làm dự báo tăng hoặc giảm")
    ax.set_xlabel("Mức tác động đến dự báo (MW)")
    ax.set_ylabel("Biến")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    return fig


def generate_simple_global_shap_insights(shap_values, feature_names: list[str]) -> list[str]:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    top_features = [human_feature_name(x) for x in ranking.head(3).index.tolist()]

    return [
        f"Ba biến quan trọng nhất của mô hình là: {', '.join(top_features)}.",
        "Điều này cho thấy mô hình chủ yếu dựa vào lịch sử tiêu thụ gần nhất và đặc trưng thời gian để dự báo.",
        "Phần giải thích bên dưới sẽ cho biết cụ thể ở từng thời điểm, biến nào làm giá trị dự báo tăng hoặc giảm.",
    ]


# ===================== ANOMALY DETECTION HELPERS =====================
ANOMALY_VALUE_COL = "load_value"

#Tạo đặc trưng cho mô hình Isolation Forest, bao gồm các đặc trưng thời gian và các đặc trưng trễ, trung bình trượt để giúp mô hình học được các mẫu bất thường trong dữ liệu dự báo
def build_anomaly_feature_frame(df_input: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df_input.copy().sort_index()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])

    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["lag_1"] = out[value_col].shift(1)
    out["lag_24"] = out[value_col].shift(24)
    out["roll_mean_24"] = out[value_col].rolling(24, min_periods=1).mean()
    out["roll_std_24"] = out[value_col].rolling(24, min_periods=1).std(ddof=0).fillna(0.0)
    out["diff_from_roll_mean_24"] = out[value_col] - out["roll_mean_24"]

    return out


def fit_isolation_forest(history_df: pd.DataFrame, target_col: str):
    train_source = history_df[[target_col]].copy().rename(columns={target_col: ANOMALY_VALUE_COL})
    hist = build_anomaly_feature_frame(train_source, ANOMALY_VALUE_COL)
    feature_cols = [
        ANOMALY_VALUE_COL,
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "lag_1",
        "lag_24",
        "roll_mean_24",
        "roll_std_24",
        "diff_from_roll_mean_24",
    ]
    train_df = hist[feature_cols].dropna().copy()

    if len(train_df) < 100:
        raise ValueError("Chưa đủ dữ liệu lịch sử để huấn luyện Isolation Forest.")
    
#Khỏi tạo mô hình Isolation Forest của thư viện sklearn để phát hiện điểm bất thường trong dữ liệu dự báo
    model = IsolationForest(
        n_estimators=200,           #Số cây trong rừng
        contamination=0.05,         #Tỷ lệ điểm bất thường dự kiến trong dữ liệu (có thể điều chỉnh nếu cần)
        random_state=42,            #Đặt random_state để đảm bảo kết quả có thể tái lập
    )
    model.fit(train_df)
    return model, feature_cols, hist


def detect_forecast_anomalies(history_df: pd.DataFrame, forecast_df: pd.DataFrame, target_col: str):
    model, feature_cols, hist = fit_isolation_forest(history_df, target_col)

    history_series = history_df[target_col].dropna().astype(float).copy()
    rows = []

    for ts, pred in forecast_df[["Datetime", "yhat"]].itertuples(index=False, name=None):
        ts = pd.Timestamp(ts)
        series_for_ts = pd.concat([
            history_series,
            pd.Series([float(pred)], index=[ts])
        ])
        tmp = build_anomaly_feature_frame(
            pd.DataFrame({ANOMALY_VALUE_COL: series_for_ts}),
            ANOMALY_VALUE_COL,
        )
        row = tmp.loc[[ts]].copy()
        rows.append(row)
        history_series.loc[ts] = float(pred)

    future_features = pd.concat(rows).sort_index()
    X_future = future_features[feature_cols].copy().ffill().bfill().fillna(0.0)

    preds = model.predict(X_future)                # Hàm thư viện sklearn: 1 = bình thường, -1 = bất thường
    scores = model.decision_function(X_future)     # Hàm thư viện sklearn: điểm càng thấp càng bất thường

    result = forecast_df.copy()
    result = result.sort_values("Datetime").reset_index(drop=True)
    result["anomaly_flag"] = preds
    result["anomaly_score"] = scores
    result["is_anomaly"] = result["anomaly_flag"] == -1

    hourly_stats = hist.groupby("hour")[ANOMALY_VALUE_COL].agg(["mean", "std"]).reset_index()
    hourly_stats.columns = ["hour", "hour_mean", "hour_std"]
    hourly_stats["hour_std"] = hourly_stats["hour_std"].fillna(1.0).replace(0, 1.0)

    if "hour" not in result.columns:
        result["hour"] = pd.to_datetime(result["Datetime"]).dt.hour

    result = result.merge(hourly_stats, on="hour", how="left")
    result["hour_zscore"] = (result["yhat"] - result["hour_mean"]) / result["hour_std"]

    return result


def generate_anomaly_insights(anomaly_df: pd.DataFrame) -> list[str]:
    flagged = anomaly_df[anomaly_df["is_anomaly"]].copy()

    if flagged.empty:
        return [
            "Không phát hiện điểm bất thường rõ rệt trong giai đoạn dự báo.",
            "Các giá trị dự báo nhìn chung vẫn nằm trong vùng hành vi quen thuộc của dữ liệu lịch sử."
        ]

    top = flagged.nsmallest(3, "anomaly_score")
    insights = [
        f"Phát hiện {len(flagged)} điểm dự báo có dấu hiệu bất thường cần theo dõi."
    ]

    for _, row in top.iterrows():
        direction = "cao hơn" if row["hour_zscore"] >= 0 else "thấp hơn"
        insights.append(
            f"Tại {row['Datetime']}, phụ tải dự báo {row['yhat']:,.2f} MW, {direction} mức thông thường của cùng khung giờ khoảng {abs(row['hour_zscore']):,.2f} độ lệch chuẩn."
        )

    return insights


def plot_anomaly_chart(forecast_with_anomaly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        forecast_with_anomaly["Datetime"],
        forecast_with_anomaly["yhat"],
        linewidth=1.8,
        label="Forecast"
    )

    flagged = forecast_with_anomaly[forecast_with_anomaly["is_anomaly"]]
    if not flagged.empty:
        ax.scatter(
            flagged["Datetime"],
            flagged["yhat"],
            s=60,
            color="red",
            label="Anomaly"
        )

    ax.set_title("Forecast with Anomaly Warnings")
    ax.set_xlabel("Time")
    ax.set_ylabel("MW")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    return fig

def render_welcome_screen():
    st.markdown("""
    <style>
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #132238 45%, #16304d 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 28px 30px;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
        line-height: 1.2;
    }

    .hero-desc {
        font-size: 16px;
        color: #d7e3f4;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .mini-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        min-height: 150px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .mini-card-title {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .mini-card-desc {
        font-size: 14px;
        color: #cbd5e1;
        line-height: 1.6;
    }

    .feature-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px;
        height: 100%;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .feature-text {
        font-size: 14px;
        color: #d1d5db;
        line-height: 1.7;
    }

    .note-box {
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 16px;
        padding: 16px 18px;
        margin-top: 12px;
        color: #dbeafe;
        line-height: 1.7;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">⚡ Chào mừng đến với hệ thống phân tích và dự báo điện năng</div>
        <div class="hero-desc">
            Hệ thống hỗ trợ tải dữ liệu điện năng từ file CSV, thực hiện phân tích khám phá dữ liệu,
            trực quan hóa xu hướng tiêu thụ và dự báo nhu cầu điện bằng mô hình AI.
            Hãy bắt đầu bằng cách tải file dữ liệu ở thanh bên trái.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="mini-card">
            <div style="font-size:32px;">📊</div>
            <div class="mini-card-title">Phân tích EDA</div>
            <div class="mini-card-desc">
                Khám phá xu hướng tiêu thụ điện theo giờ, ngày, tháng, mùa và loại ngày.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="mini-card">
            <div style="font-size:32px;">🔮</div>
            <div class="mini-card-title">Dự báo nhu cầu</div>
            <div class="mini-card-desc">
                Dự đoán phụ tải điện trong tương lai theo khoảng ngày người dùng lựa chọn.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="mini-card">
            <div style="font-size:32px;">🤖</div>
            <div class="mini-card-title">Giải thích mô hình</div>
            <div class="mini-card-desc">
                Hỗ trợ giải thích kết quả dự báo và cảnh báo bất thường trong dữ liệu dự báo.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">📁 Định dạng file đầu vào</div>
            <div class="feature-text">
                File CSV nên có tối thiểu 2 cột chính:
                <br><br>
                <b>1. Datetime</b>: thời gian theo định dạng ngày giờ
                <br>
                <b>2. PJME_MW</b>: giá trị điện năng tiêu thụ
            </div>
        </div>
        """, unsafe_allow_html=True)

        sample_df = pd.DataFrame({
            "Datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
            "PJME_MW": [24567.0, 23891.0, 23125.0]
        })
        st.markdown("#### CSV mẫu")
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🚀 Hệ thống hỗ trợ</div>
            <div class="feature-text">
                • Hiển thị tổng quan dữ liệu đầu vào<br>
                • Phân tích mức tiêu thụ theo nhiều góc nhìn<br>
                • Dự báo điện năng theo khoảng ngày<br>
                • Giải thích dự báo bằng SHAP<br>
                • Cảnh báo điểm bất thường bằng Isolation Forest
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="note-box">
            <b>Gợi ý:</b> Sau khi upload file, hệ thống sẽ tự động làm sạch dữ liệu, xử lý cột thời gian
            và hiển thị các biểu đồ phân tích trực quan ở hai tab EDA và Forecast.
        </div>
        """, unsafe_allow_html=True)

# ===================== SIDEBAR =====================
st.sidebar.header("SETTING")
st.sidebar.caption("Upload file data")

uploaded = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded is not None:
    show_raw = st.sidebar.toggle("Show raw data (head)", value=False)
    show_fc_table = st.sidebar.toggle("Show forecast table", value=True)
    save_name = st.sidebar.text_input("Export file name", value="forecast_result.csv")
else:
    show_raw = False
    show_fc_table = True
    save_name = "forecast_result.csv"

bar_label_mode = "Trong cột"

# ===================== LOAD DATA =====================
df = None
current_raw_path = None

if uploaded is None:
    render_welcome_screen()
    st.stop()

raw_path = UPLOAD_RAW_DIR / uploaded.name

with open(raw_path, "wb") as f:
    f.write(uploaded.getbuffer())

current_raw_path = raw_path
df = preprocess_csv(str(raw_path), TIME_COL, TARGET_COL)

if df is not None and not df.empty:
    processed_path = UPLOAD_PROCESSED_DIR / uploaded.name
    df.to_csv(processed_path)
    st.sidebar.success(f"✓ Uploaded file processed: {uploaded.name}")
else:
    st.error("File upload không hợp lệ hoặc không có dữ liệu.")
    st.stop()

# ===================== TABS =====================
tab_eda, tab_forecast = st.tabs(["📊 EDA", "🔮 Forecast"])


# ===================== TAB EDA =====================
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA)")

    if df is None:
        st.warning("Choose a file.")
        st.stop()

    if current_raw_path is None:
        st.warning("Raw file not found.")
        st.stop()

    df_analysis = add_analysis_features(df, TARGET_COL)
    raw_df = pd.read_csv(current_raw_path)

    raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], errors="coerce")
    raw_df = raw_df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

    full_range = pd.date_range(
        start=raw_df[TIME_COL].min(),
        end=raw_df[TIME_COL].max(),
        freq="H"
    )

    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Raw Rows", f"{len(raw_df):,}")
    info2.metric("Rows Used", f"{len(df_analysis):,}")
    info3.metric("Missing Timestamps", f"{len(full_range) - len(raw_df):,}")
    info4.metric("Duplicated Timestamps", f"{raw_df[TIME_COL].duplicated().sum():,}")

    st.markdown("### Basic Information")
    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Start", str(df_analysis.index.min()))
    meta2.metric("End", str(df_analysis.index.max()))
    meta3.metric("Target", TARGET_COL)

    if show_raw:
        with st.expander("Show raw data (first 30 rows)", expanded=False):
            st.dataframe(raw_df.head(30), use_container_width=True)

    # ===== TREND =====
    st.markdown("### Electricity Consumption Trend")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_analysis.index, df_analysis[TARGET_COL], linewidth=0.9)
    ax.set_title("Electricity Consumption Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    show_ai_insight("Auto Insight for Trend Chart", generate_trend_insights(df_analysis[TARGET_COL]))

    # ===== HOURLY =====
    st.markdown("### Average Electricity Load by Hour")
    hourly_mean = df_analysis.groupby("hour")[TARGET_COL].mean()

    peak_hour = int(hourly_mean.idxmax())
    peak_value = float(hourly_mean.max())
    low_hour = int(hourly_mean.idxmin())
    low_value = float(hourly_mean.min())

    hcol1, hcol2 = st.columns(2)
    hcol1.metric("⬆️ Peak Hour", f"{peak_hour}:00", f"{peak_value:,.2f} MW")
    hcol2.metric("⬇️ Low Hour", f"{low_hour}:00", f"{low_value:,.2f} MW")

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(hourly_mean.index, hourly_mean.values, alpha=0.75)
    bars[peak_hour].set_color("red")
    bars[low_hour].set_color("green")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.set_title("Average Electricity Load by Hour")
    ax.set_xticks(range(24))
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    show_ai_insight("Auto Insight for Hourly Chart", generate_hourly_insights(hourly_mean))

    # ===== DAILY =====
    st.markdown("### Analysis of Electricity Consumption by Day")
    min_date = df_analysis.index.min().date()
    max_date = df_analysis.index.max().date()

    selected_date = st.date_input(
        "Select a date to analyze",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    selected_day_df = df_analysis[df_analysis.index.date == selected_date].copy()
    full_day = pd.date_range(start=pd.Timestamp(selected_date), periods=24, freq="h")
    selected_day_df = selected_day_df.reindex(full_day)
    selected_day_df["hour"] = selected_day_df.index.hour

    if not selected_day_df[TARGET_COL].dropna().empty:
        valid = selected_day_df[TARGET_COL].dropna()

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Daily Load", f"{valid.mean():,.2f} MW")
        col2.metric("Maximum Daily Load", f"{valid.max():,.2f} MW")
        col3.metric("Minimum Daily Load", f"{valid.min():,.2f} MW")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(selected_day_df["hour"], selected_day_df[TARGET_COL], marker="o", linewidth=2)
        ax.set_title(f"Electricity Consumption by Hour on {selected_date}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("MW")
        ax.set_xticks(range(24))
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        show_ai_insight(
            "Auto Insight for Daily Chart",
            generate_selected_day_insights(selected_day_df, TARGET_COL, selected_date)
        )
    else:
        st.warning(f"No data available for {selected_date}")

        # ===== MONTHLY =====
    st.markdown("### Analysis of Electricity Consumption by Month")

    available_years = sorted(df_analysis["year"].dropna().unique().tolist())
    selected_year = st.selectbox(
        "Select year to analyze by month",
        options=available_years,
        index=len(available_years) - 1
    )

    df_year = df_analysis[df_analysis["year"] == selected_year].copy()

    monthly_mean = (
        df_year.groupby("month")[TARGET_COL]
        .mean()
        .reindex(range(1, 13))
    )

    valid_monthly = monthly_mean.dropna()

    if not valid_monthly.empty:
        max_month = int(valid_monthly.idxmax())
        min_month = int(valid_monthly.idxmin())
        max_value = float(valid_monthly.max())
        min_value = float(valid_monthly.min())

        mcol1, mcol2 = st.columns(2)
        mcol1.metric(
            "🔥 The Highest Month",
            f"Month {max_month}/{selected_year}",
            f"{max_value:,.2f} MW"
        )
        mcol2.metric(
            "❄️ The Lowest Month",
            f"Month {min_month}/{selected_year}",
            f"{min_value:,.2f} MW"
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(monthly_mean.index, monthly_mean.values, alpha=0.75)

        if not np.isnan(monthly_mean.loc[max_month]):
            bars[max_month - 1].set_color("red")
        if not np.isnan(monthly_mean.loc[min_month]):
            bars[min_month - 1].set_color("green")

        ax.set_xlabel("Month")
        ax.set_ylabel("MW")
        ax.set_title(f"Average Electricity Load by Month in {selected_year}")
        ax.set_xticks(range(1, 13))
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        monthly_year_df = df_year.groupby(["year", "month"])[TARGET_COL].mean().reset_index()
        show_ai_insight(
            "Auto Insight for Monthly Chart",
            generate_monthly_insights(monthly_year_df, TARGET_COL)
        )
    else:
        st.warning(f"No monthly data available for year {selected_year}.")

    # ===== SEASON =====
    st.markdown("### Analysis of Electricity Consumption by Season")

    def generate_selected_season_insights(season_month_daytype_df: pd.DataFrame, target_col: str, selected_season: str) -> list[str]:
            if season_month_daytype_df.empty:
                return [f"Chưa đủ dữ liệu để phân tích mùa {selected_season}."]

            max_row = season_month_daytype_df.loc[season_month_daytype_df[target_col].idxmax()]
            min_row = season_month_daytype_df.loc[season_month_daytype_df[target_col].idxmin()]

            insights = [
                f"Trong mùa {selected_season}, mức tiêu thụ cao nhất nằm ở tháng {int(max_row['month'])} thuộc nhóm {max_row['day_type']} với khoảng {max_row[target_col]:,.2f} MW.",
                f"Trong mùa {selected_season}, mức tiêu thụ thấp nhất nằm ở tháng {int(min_row['month'])} thuộc nhóm {min_row['day_type']} với khoảng {min_row[target_col]:,.2f} MW.",
                "Biểu đồ cho thấy trong cùng một mùa, mức tiêu thụ điện vẫn thay đổi theo từng tháng và từng loại ngày.",
            ]

            return insights
    

    season_month_map = {
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
        "Winter": [12, 1, 2],
    }

    selected_season = st.selectbox(
        "Select season to analyze",
        options=SEASON_ORDER,
        index=1
    )

    selected_months = season_month_map[selected_season]

    season_month_daytype = (
        df_analysis[
            (df_analysis["season"] == selected_season)
            & (df_analysis["month"].isin(selected_months))
        ]
        .groupby(["month", "day_type"])[TARGET_COL]
        .mean()
        .reset_index()
    )

    pivot_season = season_month_daytype.pivot_table(
        values=TARGET_COL,
        index="month",
        columns="day_type",
        aggfunc="mean"
    ).reindex(selected_months)

    valid_values = season_month_daytype[TARGET_COL].dropna()

    if not valid_values.empty:
        max_row = season_month_daytype.loc[season_month_daytype[TARGET_COL].idxmax()]
        min_row = season_month_daytype.loc[season_month_daytype[TARGET_COL].idxmin()]

        scol1, scol2 = st.columns(2)
        scol1.metric(
            "🔥 Highest in Selected Season",
            f"Month {int(max_row['month'])} - {max_row['day_type']}",
            f"{max_row[TARGET_COL]:,.2f} MW"
        )
        scol2.metric(
            "❄️ Lowest in Selected Season",
            f"Month {int(min_row['month'])} - {min_row['day_type']}",
            f"{min_row[TARGET_COL]:,.2f} MW"
        )

        fig, ax = plt.subplots(figsize=(12, 4.5))
        x = np.arange(len(selected_months))
        width = 0.25

        if "Weekday" in pivot_season.columns:
            ax.bar(x - width, pivot_season["Weekday"].values, width, label="Weekday", alpha=0.75)
        if "Weekend" in pivot_season.columns:
            ax.bar(x, pivot_season["Weekend"].values, width, label="Weekend", alpha=0.75)
        if "Holiday" in pivot_season.columns:
            ax.bar(x + width, pivot_season["Holiday"].values, width, label="Holiday", alpha=0.75)

        ax.set_xlabel("Month")
        ax.set_ylabel("MW")
        ax.set_title(f"Electricity Consumption by Month and Day Type in {selected_season}")
        ax.set_xticks(x)
        ax.set_xticklabels(selected_months)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        show_ai_insight(
            "Auto Insight for Selected Season Chart",
            generate_selected_season_insights(season_month_daytype, TARGET_COL, selected_season)
        )
    else:
        st.warning(f"No data available for season {selected_season}.")


# ===================== TAB FORECAST =====================
with tab_forecast:
    if df is None:
        st.warning("Choose a file.")
        st.stop()

    st.subheader("Base information about the dataset")
    st.markdown(f"""
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
        <tr>
            <td style="padding: 10px; text-align: center;"><strong>Number of Rows</strong><br>{len(df):,}</td>
            <td style="padding: 10px; text-align: center;"><strong>Start Date</strong><br>{str(df.index.min())}</td>
            <td style="padding: 10px; text-align: center;"><strong>End Date</strong><br>{str(df.index.max())}</td>
            <td style="padding: 10px; text-align: center;"><strong>Target Column</strong><br>{TARGET_COL}</td>     
        </tr>
    </table>           
    """, unsafe_allow_html=True)
    st.markdown("### Select Forecasting Time Range")

    min_date = (df.index.max() + pd.Timedelta(hours=1)).date()
    source_signature = f"{len(df)}|{df.index.min()}|{df.index.max()}"

    if st.session_state.forecast_source_signature != source_signature:
        st.session_state.forecast_result = None
        st.session_state.forecast_start_date = None
        st.session_state.forecast_end_date = None
        st.session_state.forecast_source_signature = source_signature
        st.session_state.selected_forecast_idx = 0

    with st.form("forecast_form"):
        fcol1, fcol2 = st.columns(2)
        start_date = fcol1.date_input(
            "Start date",
            value=st.session_state.forecast_start_date or min_date,
            key="forecast_start_input",
        )
        end_date = fcol2.date_input(
            "End date",
            value=st.session_state.forecast_end_date or min_date,
            key="forecast_end_input",
        )
        submitted = st.form_submit_button("Forecast", type="primary")

    if end_date < start_date:
        st.error("End date must be greater than or equal to Start date.")
        st.stop()

    if submitted:
        with st.spinner("Running forecast..."):
            fc = forecast_by_date(
                df=df,
                time_col=TIME_COL,
                target_col=TARGET_COL,
                start_date=str(start_date),
                end_date=str(end_date),
                model_path=MODEL_PATH,
                feature_config_path=CFG_PATH,
            )

        if fc is None or len(fc) == 0:
            st.error("No forecast results available. Please check the date range or the forecast function.")
            st.stop()

        st.session_state.forecast_result = fc.copy()
        st.session_state.forecast_start_date = start_date
        st.session_state.forecast_end_date = end_date
        st.session_state.selected_forecast_idx = 0

    if st.session_state.forecast_result is None:
        st.caption("Press Forecast to run the forecast.")
        st.stop()

    fc = st.session_state.forecast_result.copy()

    st.caption(
        f"Đang hiển thị kết quả dự báo từ {st.session_state.forecast_start_date} đến {st.session_state.forecast_end_date}."
    )

    fc["Datetime"] = pd.to_datetime(fc["Datetime"])
    fc = fc.sort_values("Datetime")
    fc["date"] = fc["Datetime"].dt.date
    fc["hour"] = fc["Datetime"].dt.hour

    avg_value = float(fc["yhat"].mean())
    max_point = fc.loc[fc["yhat"].idxmax()]
    min_point = fc.loc[fc["yhat"].idxmin()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average Value", f"{avg_value:,.2f} MW")
    m2.metric("Highest Peak", f"{max_point['yhat']:,.2f} MW", str(max_point["Datetime"]))
    m3.metric("Lowest Point", f"{min_point['yhat']:,.2f} MW", str(min_point["Datetime"]))
    m4.metric("Forecast Points", f"{len(fc):,}")

    st.markdown("### Line Chart of Forecast")
    fig_line = px.line(
        fc,
        x="Datetime",
        y="yhat",
        title="Predicted Electricity Load Over Time",
        labels={"Datetime": "Time", "yhat": "MW"},
        hover_data={"Datetime": "|%Y-%m-%d %H:%M", "yhat": ":,.2f"}
    )
    fig_line.update_layout(
        xaxis_title="Time",
        yaxis_title="MW",
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    show_ai_insight("Auto Insight for Forecast Chart", generate_forecast_insights(fc))



    st.markdown("### Giải thích dự báo bằng SHAP")
    st.caption("Phần này cho biết vì sao mô hình dự báo tăng hoặc giảm tại một thời điểm cụ thể.")

    try:
        model, feature_names = load_model_and_feature_names(MODEL_PATH, CFG_PATH)

        _, X_future = build_forecast_feature_matrix(
            history_df=df,
            forecast_df=fc,
            target_col=TARGET_COL,
            feature_names=feature_names,
        )

        explainer = make_shap_explainer(model, X_future)
        shap_values = explainer(X_future)

        show_ai_insight(
            "Giải thích tổng quát của mô hình",
            generate_simple_global_shap_insights(shap_values, feature_names)
        )

        st.markdown("#### Chọn thời điểm cần giải thích")
        max_idx = len(fc) - 1
        if st.session_state.selected_forecast_idx > max_idx:
            st.session_state.selected_forecast_idx = 0

        selected_idx = st.slider(
            "Điểm dự báo",
            min_value=0,
            max_value=max_idx,
            key="selected_forecast_idx"
        )

        selected_time = fc.iloc[selected_idx]["Datetime"]
        selected_pred = float(fc.iloc[selected_idx]["yhat"])

        base_value, positive_df, negative_df, top_abs, summary_text = explain_single_prediction_simple(
            shap_row=shap_values[selected_idx],
            feature_row=X_future.iloc[selected_idx],
            pred_value=selected_pred,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Thời điểm", str(selected_time))
        c2.metric("Giá trị gốc của mô hình", f"{base_value:,.2f} MW")
        c3.metric("Dự báo cuối cùng", f"{selected_pred:,.2f} MW")

        st.markdown("#### Cách hiểu nhanh")
        st.info(summary_text)

        chart_fig = plot_simple_shap_contributions(top_abs)
        st.pyplot(chart_fig, use_container_width=True)
        plt.close(chart_fig)

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### 3 yếu tố làm dự báo tăng")
            if positive_df.empty:
                st.write("Không có yếu tố tăng rõ rệt.")
            else:
                increase_table = positive_df[["feature_name", "feature_value", "shap_value"]].copy()
                increase_table.columns = ["Yếu tố", "Giá trị", "Tác động tăng (MW)"]
                increase_table["Tác động tăng (MW)"] = increase_table["Tác động tăng (MW)"].map(lambda x: f"{x:,.2f}")
                st.dataframe(increase_table, use_container_width=True, hide_index=True)

        with right_col:
            st.markdown("#### 3 yếu tố làm dự báo giảm")
            if negative_df.empty:
                st.write("Không có yếu tố giảm rõ rệt.")
            else:
                decrease_table = negative_df[["feature_name", "feature_value", "shap_value"]].copy()
                decrease_table.columns = ["Yếu tố", "Giá trị", "Tác động giảm (MW)"]
                decrease_table["Tác động giảm (MW)"] = decrease_table["Tác động giảm (MW)"].map(lambda x: f"{x:,.2f}")
                st.dataframe(decrease_table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Không thể tạo phần giải thích SHAP: {e}")




    st.markdown("### Cảnh báo bất thường bằng Isolation Forest")
    st.caption("Các điểm bị đánh dấu là những giờ dự báo có hành vi khác đáng kể so với lịch sử quen thuộc.")

    try:
        anomaly_df = detect_forecast_anomalies(
            history_df=df,
            forecast_df=fc,
            target_col=TARGET_COL,
        )

        show_ai_insight(
            "Tóm tắt cảnh báo bất thường",
            generate_anomaly_insights(anomaly_df)
        )

        fig_anomaly = plot_anomaly_chart(anomaly_df)
        st.pyplot(fig_anomaly, use_container_width=True)
        plt.close(fig_anomaly)

        flagged = anomaly_df[anomaly_df["is_anomaly"]].copy()
        flagged = flagged.sort_values(["anomaly_score", "Datetime"]).reset_index(drop=True)

        st.markdown("#### Danh sách điểm cảnh báo")
        if flagged.empty:
            st.success("Không có điểm forecast bất thường trong giai đoạn đã chọn.")
        else:
            alert_table = flagged[[
                "Datetime",
                "yhat",
                "anomaly_score",
                "hour_mean",
                "hour_zscore",
            ]].copy()
            alert_table.columns = [
                "Thời điểm",
                "Dự báo (MW)",
                "Điểm bất thường",
                "Trung bình cùng giờ (MW)",
                "Z-score cùng giờ",
            ]
            alert_table["Dự báo (MW)"] = alert_table["Dự báo (MW)"].map(lambda x: f"{x:,.2f}")
            alert_table["Điểm bất thường"] = alert_table["Điểm bất thường"].map(lambda x: f"{x:,.4f}")
            alert_table["Trung bình cùng giờ (MW)"] = alert_table["Trung bình cùng giờ (MW)"].map(lambda x: f"{x:,.2f}")
            alert_table["Z-score cùng giờ"] = alert_table["Z-score cùng giờ"].map(lambda x: f"{x:,.2f}")
            st.dataframe(alert_table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Không thể chạy Isolation Forest: {e}")

    st.markdown("### Quick Analysis")

    daily = fc.groupby("date")["yhat"].mean()
    hourly = fc.groupby("hour")["yhat"].mean()

    max_day = daily.idxmax()
    min_day = daily.idxmin()
    max_hour = int(hourly.idxmax())
    min_hour = int(hourly.idxmin())

    peak_hour_by_day = (
        fc.loc[fc.groupby("date")["yhat"].idxmax(), ["date", "hour"]]
        .set_index("date")["hour"]
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.write("**Summary Insights**")
        st.write(
            f"""
- Highest forecast day: {max_day} (Average: {daily.max():,.2f} MW)
- Lowest forecast day: {min_day} (Average: {daily.min():,.2f} MW)
- Highest peak hour: {max_hour}:00 (Average: {hourly.max():,.2f} MW)
- Lowest peak hour: {min_hour}:00 (Average: {hourly.min():,.2f} MW)
"""
        )

    with right:
        st.write("**Hourly Average Distribution**")
        fig_h, ax_h = plt.subplots(figsize=(8, 3.2))
        ax_h.plot(hourly.index, hourly.values, marker="o", linewidth=1.8)
        ax_h.set_xlabel("Hour")
        ax_h.set_ylabel("MW")
        ax_h.set_title("Average Predicted Load by Hour")
        ax_h.grid(axis="y", linestyle="--", alpha=0.35)
        ax_h.set_xticks(list(range(0, 24, 2)))
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)

    st.markdown("### Daily Average Forecast Comparison")

    fig_bar, ax = plt.subplots(figsize=(14, 5))
    x_labels = daily.index.astype(str)
    bars = ax.bar(x_labels, daily.values, alpha=0.78)

    ax.axhline(avg_value, linestyle="--", linewidth=2, label="Average Value")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Daily Average Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("MW")
    ax.set_ylim(0, daily.max() * 1.25)
    plt.xticks(rotation=30)

    for i, d in enumerate(daily.index):
        peak_h = int(peak_hour_by_day.loc[d])
        if bar_label_mode == "Trong cột":
            y_pos = daily.values[i] * 0.15
            ax.text(
                i,
                y_pos,
                f"{daily.values[i]:,.0f} MW\nPeak {peak_h:02d}h",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

    ax.legend()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    st.markdown("### Forecast Results")

    if show_fc_table:
        st.dataframe(fc[["Datetime", "yhat"]].reset_index(drop=True), use_container_width=True)

    out_path = os.path.join(ROOT_DIR, "outputs", "forecasts", save_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fc.to_csv(out_path, index=False)

    st.download_button(
        f"Tải file {save_name}",
        data=fc.to_csv(index=False).encode("utf-8"),
        file_name=save_name,
        mime="text/csv",
    )