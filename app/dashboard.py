import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("src"))

import pandas as pd
import streamlit as st

from preprocess import preprocess_csv
from forecast import forecast_by_date

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Electricity Forecast XGBoost", layout="wide")
st.title("Dự báo nhu cầu điện bằng XGBoost")

TIME_COL = "Datetime"
TARGET_COL = "PJME_MW"

MODEL_PATH = os.path.join("artifacts", "model.pkl")
CFG_PATH = os.path.join("artifacts", "feature_config.json")

os.makedirs(os.path.join("uploads", "raw"), exist_ok=True)
os.makedirs(os.path.join("outputs", "forecasts"), exist_ok=True)

# ===================== SIDEBAR =====================
st.sidebar.header("Cấu hình")
st.sidebar.caption("Upload dữ liệu, chọn khoảng dự báo, xem kết quả và tải file.")

uploaded = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

show_raw = st.sidebar.toggle("Hiện dữ liệu gốc (head)", value=False)
show_fc_table = st.sidebar.toggle("Hiện bảng dự báo", value=True)
save_name = st.sidebar.text_input("Tên file xuất", value="forecast_result.csv")

# Style options
chart_height = st.sidebar.slider("Độ cao biểu đồ", 280, 520, 360, 20)
bar_label_mode = st.sidebar.selectbox(
    "Ghi chú cột (biểu đồ ngày)",
    ["Trong cột", "Dưới cột"],
    index=1
)

# ===================== LOAD DATA =====================
df = None
save_path = None

if uploaded:
    save_path = os.path.join("uploads", "raw", uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    df = preprocess_csv(save_path, TIME_COL, TARGET_COL)

# ===================== TABS =====================
tab_eda, tab_forecast = st.tabs(["📊 EDA dữ liệu upload", "🔮 Dự đoán"])

# ===================== TAB EDA (để sẵn, làm sau) =====================
with tab_eda:
    st.subheader("EDA dữ liệu upload")
    st.info("Phân tích EDA dữ liệu điện ")
    if df is not None and show_raw:
        st.dataframe(df.head(30), use_container_width=True)


    st.markdown("### Xu hướng tiêu thụ điện theo thời gian")

    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df.index, df[TARGET_COL])
    ax.set_title("Electricity Consumption Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)






    st.markdown("### Tiêu thụ điện trung bình theo giờ")

    hourly = df.groupby(df.index.hour)[TARGET_COL].mean()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hourly.index, hourly.values, marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.set_title("Average Electricity Load by Hour")
    ax.grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)





    st.markdown("### Tiêu thụ điện theo ngày trong tuần")

    dow = df.groupby(df.index.dayofweek)[TARGET_COL].mean()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(dow.index, dow.values)

    ax.set_xticks(range(7))
    ax.set_xticklabels(
        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    )

    ax.set_title("Average Load by Day of Week")
    ax.set_ylabel("MW")

    st.pyplot(fig, use_container_width=True)



    from statsmodels.tsa.seasonal import seasonal_decompose

    recent = df[TARGET_COL].iloc[-24*90:]

    decomp = seasonal_decompose(recent, model="additive", period=24)

    fig = decomp.plot()
    fig.set_size_inches(12,8)

    st.pyplot(fig)
# ===================== TAB FORECAST (làm chính) =====================
with tab_forecast:
    st.subheader("Dự đoán")
    if df is None:
        st.warning("Anh upload file CSV ở sidebar để bắt đầu.")
        st.stop()

    # ===== Header cards =====
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng", f"{len(df):,}")
    c2.metric("Bắt đầu", str(df.index.min()))
    c3.metric("Kết thúc", str(df.index.max()))
    c4.metric("Cột mục tiêu", TARGET_COL)

    if show_raw:
        with st.expander("Xem dữ liệu gốc (30 dòng đầu)", expanded=False):
            st.dataframe(df.head(30), use_container_width=True)

    st.markdown("### Chọn khoảng thời gian dự báo")

    # Default: dự báo từ giờ tiếp theo
    min_date = (df.index.max() + pd.Timedelta(hours=1)).date()

    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
    start_date = fcol1.date_input("Start date", value=min_date)
    end_date = fcol2.date_input("End date", value=min_date)

    # Validate input
    if end_date < start_date:
        st.error("End date phải lớn hơn hoặc bằng Start date.")
        st.stop()

    btn_col1, btn_col2 = st.columns([1, 5])
    run = btn_col1.button("Forecast", type="primary")

    if not run:
        st.caption("Nhấn Forecast để chạy dự báo.")
        st.stop()

    # ===================== FORECAST =====================
    with st.spinner("Đang dự báo..."):
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
        st.error("Không có kết quả dự báo. Kiểm tra lại khoảng ngày hoặc hàm forecast_by_date().")
        st.stop()

    # Normalize columns
    fc["Datetime"] = pd.to_datetime(fc["Datetime"])
    fc = fc.sort_values("Datetime")
    fc["date"] = fc["Datetime"].dt.date
    fc["hour"] = fc["Datetime"].dt.hour

    # ===================== SUMMARY METRICS =====================
    avg_value = float(fc["yhat"].mean())
    max_point = fc.loc[fc["yhat"].idxmax()]
    min_point = fc.loc[fc["yhat"].idxmin()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trung bình giai đoạn", f"{avg_value:,.2f} MW")
    m2.metric("Đỉnh cao nhất", f"{max_point['yhat']:,.2f} MW", str(max_point["Datetime"]))
    m3.metric("Thấp nhất", f"{min_point['yhat']:,.2f} MW", str(min_point["Datetime"]))
    m4.metric("Số điểm dự báo", f"{len(fc):,}")

    st.markdown("### Biểu đồ dự báo theo thời gian")

    # Line chart (matplotlib for nicer control)
    fig_line, ax_line = plt.subplots(figsize=(14, 4.2))
    ax_line.plot(fc["Datetime"], fc["yhat"])
    ax_line.set_title("Dự báo nhu cầu điện (yhat)")
    ax_line.set_xlabel("Thời gian")
    ax_line.set_ylabel("MW")
    ax_line.grid(axis="y", linestyle="--", alpha=0.35)
    st.pyplot(fig_line, use_container_width=True)

    # ===================== ANALYSIS =====================
    st.markdown("### Phân tích nhanh")

    daily = fc.groupby("date")["yhat"].mean()
    hourly = fc.groupby("hour")["yhat"].mean()

    max_day = daily.idxmax()
    min_day = daily.idxmin()
    max_hour = int(hourly.idxmax())
    min_hour = int(hourly.idxmin())

    # Peak hour by day
    peak_hour_by_day = (
        fc.loc[fc.groupby("date")["yhat"].idxmax(), ["date", "hour"]]
        .set_index("date")["hour"]
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.write("**Nhận xét tổng quan**")
        st.write(
            f"""
- Ngày dự báo cao nhất: {max_day} (TB khoảng {daily.max():,.2f} MW)
- Ngày dự báo thấp nhất: {min_day} (TB khoảng {daily.min():,.2f} MW)
- Giờ cao điểm thường gặp: {max_hour}:00 (TB khoảng {hourly.max():,.2f} MW)
- Giờ thấp điểm thường gặp: {min_hour}:00 (TB khoảng {hourly.min():,.2f} MW)
            """
        )

    with right:
        st.write("**Phân bố trung bình theo giờ**")
        fig_h, ax_h = plt.subplots(figsize=(8, 3.2))
        ax_h.plot(hourly.index, hourly.values, marker="o")
        ax_h.set_xlabel("Giờ")
        ax_h.set_ylabel("MW")
        ax_h.set_title("Trung bình yhat theo giờ")
        ax_h.grid(axis="y", linestyle="--", alpha=0.35)
        ax_h.set_xticks(list(range(0, 24, 2)))
        st.pyplot(fig_h, use_container_width=True)

    # ===================== DAILY BAR CHART =====================
    st.markdown("### So sánh nhu cầu điện giữa các ngày (TB ngày)")

    fig_bar, ax = plt.subplots(figsize=(14, 5.0))
    x_labels = daily.index.astype(str)
    bars = ax.bar(x_labels, daily.values)

    ax.axhline(avg_value, linestyle="--", linewidth=2, label="Trung bình giai đoạn")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("So sánh nhu cầu điện dự báo giữa các ngày")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("MW")
    ax.set_ylim(0, daily.max() * 1.25)
    plt.xticks(rotation=30)

    # Labels
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
        else:
            # Ghi dưới cột (dễ đọc hơn)
            ax.text(
                i,
                -daily.max() * 0.06,
                f"{daily.values[i]:,.0f} MW\nPeak {peak_h:02d}h",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

    if bar_label_mode == "Dưới cột":
        # chừa khoảng dưới để không bị cắt chữ
        fig_bar.subplots_adjust(bottom=0.22)

    ax.legend()
    st.pyplot(fig_bar, use_container_width=True)

    # ===================== TABLE + DOWNLOAD =====================
    st.markdown("### Kết quả dự báo")

    if show_fc_table:
        st.dataframe(fc[["Datetime", "yhat"]].reset_index(drop=True), use_container_width=True)

    out_path = os.path.join("outputs", "forecasts", save_name)
    fc.to_csv(out_path, index=False)
    st.success(f"Đã lưu file: {out_path}")

    st.download_button(
        f"Tải file {save_name}",
        data=fc.to_csv(index=False).encode("utf-8"),
        file_name=save_name,
        mime="text/csv",
    )