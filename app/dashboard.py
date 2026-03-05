import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("src"))

import pandas as pd
import streamlit as st

from preprocess import preprocess_csv
from forecast import forecast_by_date

st.set_page_config(page_title="Electricity Forecast XGBoost", layout="wide")
st.title("Dự báo nhu cầu điện bằng XGBoost")

TIME_COL = "Datetime"
TARGET_COL = "PJME_MW"

MODEL_PATH = os.path.join("artifacts", "model.pkl")
CFG_PATH = os.path.join("artifacts", "feature_config.json")

os.makedirs(os.path.join("uploads", "raw"), exist_ok=True)
os.makedirs(os.path.join("outputs", "forecasts"), exist_ok=True)

uploaded = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded:
    save_path = os.path.join("uploads", "raw", uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"Đã upload: {save_path}")

    df = preprocess_csv(save_path, TIME_COL, TARGET_COL)

    st.subheader("Xem nhanh dữ liệu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng", len(df))
    col2.metric("Từ ngày", str(df.index.min()))
    col3.metric("Đến ngày", str(df.index.max()))
    st.dataframe(df.head(20))

    st.subheader("Chọn khoảng ngày muốn dự báo")
    min_date = (df.index.max() + pd.Timedelta(hours=1)).date()
    start_date = st.date_input("Start date", value=min_date)
    end_date = st.date_input("End date", value=min_date)

    if st.button("Forecast"):
        fc = forecast_by_date(
            df=df,
            time_col=TIME_COL,
            target_col=TARGET_COL,
            start_date=str(start_date),
            end_date=str(end_date),
            model_path=MODEL_PATH,
            feature_config_path=CFG_PATH,
        )

        st.subheader("Kết quả dự báo")
        st.dataframe(fc)
        st.line_chart(fc.set_index("Datetime")["yhat"])

        # ===== PHÂN TÍCH KẾT QUẢ DỰ BÁO =====
        st.subheader("Phân tích kết quả dự báo")

        fc["Datetime"] = pd.to_datetime(fc["Datetime"])
        fc["date"] = fc["Datetime"].dt.date
        fc["hour"] = fc["Datetime"].dt.hour

        daily = fc.groupby("date")["yhat"].mean()
        hourly = fc.groupby("hour")["yhat"].mean()

        max_day = daily.idxmax()
        min_day = daily.idxmin()
        max_hour = hourly.idxmax()
        min_hour = hourly.idxmin()
        avg_value = fc["yhat"].mean()

        st.write("### Nhận xét tổng quan")
        st.write(
            f"""
- Ngày có nhu cầu tiêu thụ điện **cao nhất** dự kiến là **{max_day}** với mức trung bình khoảng **{daily.max():,.2f} MW**.
- Ngày có nhu cầu tiêu thụ điện **thấp nhất** dự kiến là **{min_day}** với mức trung bình khoảng **{daily.min():,.2f} MW**.
- Mức tiêu thụ điện **trung bình trong giai đoạn dự báo** khoảng **{avg_value:,.2f} MW**.
            """
        )

        st.write(
            f"""
- Trong ngày, **giờ tiêu thụ điện cao nhất** dự kiến là khoảng **{max_hour}:00**, khi nhu cầu điện đạt trung bình khoảng **{hourly.max():,.2f} MW**.
- **Giờ tiêu thụ điện thấp nhất** thường rơi vào khoảng **{min_hour}:00**, với mức tiêu thụ trung bình khoảng **{hourly.min():,.2f} MW**.
            """
        )


        peak_hour_by_day = (
            fc.loc[fc.groupby("date")["yhat"].idxmax(), ["date", "hour"]]
            .set_index("date")["hour"]
        )


        st.subheader("So sánh nhu cầu điện giữa các ngày")

        fig, ax = plt.subplots(figsize=(14,6))

        x_labels = daily.index.astype(str)
        bars = ax.bar(x_labels, daily.values, color="#2E86C1")

        # đường trung bình
        ax.axhline(avg_value, color="red", linestyle="--", linewidth=2, label="Trung bình giai đoạn")

        # grid cho dễ nhìn
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # hiển thị giá trị MW + peak hour
        for i, d in enumerate(daily.index):

            peak_h = int(peak_hour_by_day.loc[d])

            ax.text(
                i,
                daily.values[i]*0.1,
                f"{daily.values[i]:,.0f} MW\nPeak {peak_h:02d}h",
                ha='center',
                color="white",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_title("So sánh nhu cầu điện dự báo giữa các ngày", fontsize=16, fontweight="bold")
        ax.set_xlabel("Ngày", fontsize=12)
        ax.set_ylabel("Công suất (MW)", fontsize=12)
        ax.set_ylim(0, daily.max()*1.2)

        plt.xticks(rotation=35)

        ax.legend()

        st.pyplot(fig)


        # ===== LƯU FILE + DOWNLOAD (để ngoài vòng for) =====
        out_path = os.path.join("outputs", "forecasts", "forecast_result.csv")
        fc.to_csv(out_path, index=False)
        st.success(f"Đã lưu file: {out_path}")

        st.download_button(
            "Tải file forecast_result.csv",
            data=fc.to_csv(index=False).encode("utf-8"),
            file_name="forecast_result.csv",
            mime="text/csv",
        )
