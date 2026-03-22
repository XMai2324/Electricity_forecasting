import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("src"))

import pandas as pd
import streamlit as st
import plotly.express as px

from preprocess import preprocess_csv
from forecast import forecast_by_date

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Electricity Forecast XGBoost", layout="wide")
st.title("Electricity Analysis & Forecasting System")

TIME_COL = "Datetime"
TARGET_COL = "PJME_MW"

MODEL_PATH = os.path.join("artifacts", "model.pkl")
CFG_PATH = os.path.join("artifacts", "feature_config.json")

os.makedirs(os.path.join("uploads", "raw"), exist_ok=True)
os.makedirs(os.path.join("outputs", "forecasts"), exist_ok=True)

# ===================== SIDEBAR =====================
st.sidebar.header("SETTING")
st.sidebar.caption("Upload file data")

uploaded = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

show_raw = st.sidebar.toggle("Show raw data (head)", value=False)
show_fc_table = st.sidebar.toggle("Show forecast table", value=True)
save_name = st.sidebar.text_input("Eport file name", value="forecast_result.csv")

# Style options
bar_label_mode = "Trong cột"
#bar_label_mode = st.sidebar.selectbox("Chế độ nhãn cột", options=["Trong cột", "outside"], index=0)





#===================== LOAD DATA =====================
df = None
save_path = None
current_raw_path = None

# Load file mặc định: KHÔNG preprocess
default_file = r"D:\ĐACN\Electricity_forecasting\data\sample\PJME_hourly.csv"
if os.path.exists(default_file):
    current_raw_path = default_file
    raw_default_df = pd.read_csv(default_file)

    raw_default_df[TIME_COL] = pd.to_datetime(raw_default_df[TIME_COL], errors="coerce")
    raw_default_df = raw_default_df.dropna(subset=[TIME_COL])
    raw_default_df = raw_default_df.sort_values(TIME_COL)

    df = raw_default_df.set_index(TIME_COL).copy()
    st.sidebar.success(f"✓ Default file loaded: {default_file}")

# Upload file khác sẽ ghi đè
if uploaded:
    raw_path = os.path.join("uploads", "raw", uploaded.name)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())

    current_raw_path = raw_path

    # Chỉ file upload mới preprocess
    df = preprocess_csv(raw_path, TIME_COL, TARGET_COL)

    if df is not None:
        processed_path = os.path.join("uploads", "processed", uploaded.name)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path)
        st.sidebar.success(f"✓ Uploaded file processed: {uploaded.name}")
    else:
        st.sidebar.error("File upload không hợp lệ hoặc không có dữ liệu. Vẫn dùng file mặc định.")

# ===================== TABS =====================
tab_eda, tab_forecast = st.tabs(["📊 EDA", "🔮 Forecast"])

# ===================== TAB EDA (để sẵn, làm sau) =====================
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA)")

    if df is None:
        st.warning("Choose a file.")
        st.stop()

    if current_raw_path is None:
        st.warning("Raw file not found.")
        st.stop()

    raw_df = pd.read_csv(current_raw_path)

    st.write("Raw rows:", len(raw_df))
    st.write("Rows used in analysis:", len(df))

    raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], errors="coerce")
    raw_df = raw_df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

    full_range = pd.date_range(
        start=raw_df[TIME_COL].min(),
        end=raw_df[TIME_COL].max(),
        freq="H"
    )

    st.write("Expected hourly rows:", len(full_range))
    st.write("Missing timestamps:", len(full_range) - len(raw_df))
    st.write("Duplicated timestamps:", raw_df[TIME_COL].duplicated().sum())

    st.markdown("### Basic information about the dataset")
    st.markdown(f"""
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
        <tr>
            <td style="padding: 10px; text-align: center;"><strong>Number of Rows</strong><br>{len(df):,}</td>
            <td style="padding: 10px; text-align: center;"><strong>Start</strong><br>{str(df.index.min())}</td>
            <td style="padding: 10px; text-align: center;"><strong>End</strong><br>{str(df.index.max())}</td>
            <td style="padding: 10px; text-align: center;"><strong>Target Column</strong><br>{TARGET_COL}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    if show_raw:
        with st.expander("Show raw data (first 30 rows)", expanded=False):
            st.dataframe(raw_df.head(30), use_container_width=True)

#==============================================================================


    st.markdown("### Electricity Consumption Trend")

    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df.index, df[TARGET_COL])
    ax.set_title("Electricity Consumption Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    st.caption("This chart shows the electricity consumption trend over time")




    st.markdown("### Average Electricity Load by Hour")

    # Tạo features thời gian
    df_analysis = df.copy()
    df_analysis["hour"] = df_analysis.index.hour
    df_analysis["dayofweek"] = df_analysis.index.dayofweek
    df_analysis["month"] = df_analysis.index.month
    df_analysis["day_name"] = df_analysis.index.day_name()

    hourly = df_analysis.groupby("hour")[TARGET_COL].mean()

    # Metrics
    peak_hour = hourly.idxmax()
    peak_value = hourly.max()
    low_hour = hourly.idxmin()
    low_value = hourly.min()

    hcol1, hcol2 = st.columns(2)
    hcol1.metric("⬆️ Peak Hour", f"{peak_hour}:00", f"{peak_value:,.2f} MW")
    hcol2.metric("⬇️ Low Hour", f"{low_hour}:00", f"{low_value:,.2f} MW")

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(hourly.index, hourly.values, color="steelblue", alpha=0.7)
    # Tô màu cho giờ cao điểm và thấp điểm
    bars[peak_hour].set_color("red")
    bars[low_hour].set_color("green")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.set_title("Average Electricity Load by Hour")
    ax.set_xticks(range(24))
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    st.caption(f"""
    **Insight:** Electricity consumption varies significantly throughout the day. 
                The peak hour occurs at **{peak_hour}:00 with {peak_value:,.2f} MW**, 
                while the lowest demand occurs at **{low_hour}:00 with {low_value:,.2f} MW**.
                """)





# =========================Lượng tiêu thụ điện theo giờ trong ngày cụ thể=========================
    st.markdown("### Analysis of Electricity Consumption by Day")
    min_date = df_analysis.index.min().date()
    max_date = df_analysis.index.max().date()

    selected_date = st.date_input(
        "Select a date to analyze",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    # Lọc dữ liệu của ngày được chọn
    selected_date_data = df_analysis[df_analysis.index.date == selected_date].copy()

    # Tạo đủ 24 giờ từ 0 đến 23h
    full_day = pd.date_range(
        start=pd.Timestamp(selected_date),
        periods=24,
        freq="h"
    )

    selected_date_data = selected_date_data.reindex(full_day)
    selected_date_data["hour"] = selected_date_data.index.hour

    if not selected_date_data[TARGET_COL].dropna().empty:
        # Tính các chỉ số trong ngày
        avg_day = selected_date_data[TARGET_COL].mean(skipna=True)
        max_day_val = selected_date_data[TARGET_COL].max(skipna=True)
        min_day_val = selected_date_data[TARGET_COL].min(skipna=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Daily Load", f"{avg_day:,.2f} MW")
        col2.metric("Maximum Daily Load", f"{max_day_val:,.2f} MW")
        col3.metric("Minimum Daily Load", f"{min_day_val:,.2f} MW")

        # Tìm giờ cao nhất
        peak_data = selected_date_data[TARGET_COL].dropna()
        peak_time = peak_data.idxmax()
        peak_hour = peak_time.hour
        peak_value = peak_data.max()

        # Vẽ biểu đồ giống mẫu
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            selected_date_data["hour"],
            selected_date_data[TARGET_COL],
            marker="o",
            linewidth=2
        )

        ax.set_title(f"Electricity Consumption by Hour on {selected_date}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("MW")
        ax.set_xticks(range(24))
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Nhận xét
        st.markdown("### Insight")
        st.write(
            f"On **{selected_date}**, the highest electricity consumption occurs at "
            f"**{peak_hour}:00** with a value of approximately **{peak_value:,.2f} MW**."
        )

    else:
        st.warning(f"No data available for {selected_date}")




    st.markdown("### Analysis of Electricity Consumption by Month")

    # Tạo features thời gian
    df_analysis = df.copy()
    df_analysis["hour"] = df_analysis.index.hour
    df_analysis["dayofweek"] = df_analysis.index.dayofweek
    df_analysis["month"] = df_analysis.index.month
    df_analysis["year"] = df_analysis.index.year
    df_analysis["day_name"] = df_analysis.index.day_name()

    # Tính trung bình theo năm và tháng
    monthly_year = df_analysis.groupby(["year", "month"])[TARGET_COL].mean().reset_index()

    # Tìm tháng cao nhất và thấp nhất (theo năm)
    max_row = monthly_year.loc[monthly_year[TARGET_COL].idxmax()]
    min_row = monthly_year.loc[monthly_year[TARGET_COL].idxmin()]

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("🔥 The highest month", f"Month {int(max_row['month'])} {int(max_row['year'])}", f"{max_row[TARGET_COL]:,.2f} MW")
    mcol2.metric("❄️ The lowest month", f"Month {int(min_row['month'])} {int(min_row['year'])}", f"{min_row[TARGET_COL]:,.2f} MW")

    # Vẽ biểu đồ trung bình theo tháng (không phân biệt năm)
    monthly = df_analysis.groupby("month")[TARGET_COL].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(monthly.index, monthly.values, color="steelblue", alpha=0.7)
    # Tô màu cho tháng cao nhất và thấp nhất (dựa trên trung bình tổng)
    bars[int(max_row['month']) - 1].set_color("red")
    bars[int(min_row['month']) - 1].set_color("green")
    ax.set_xlabel("Month")
    ax.set_ylabel("MW")
    ax.set_title("Average Electricity Load by Month")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    st.caption(f"""
    **Insight:** **Month {int(max_row['month'])} {int(max_row['year'])}** has the highest consumption ({max_row[TARGET_COL]:,.2f} MW), 
                while **Month {int(min_row['month'])} {int(min_row['year'])}** has the lowest ({min_row[TARGET_COL]:,.2f} MW). 
                This difference reflects the clear seasonal pattern, influenced by weather conditions and actual demand.
                """)




    st.markdown("### Analysis of Electricity Consumption by Season")
    
    # Thêm biến mùa
    def get_season(month):
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Winter"
    
    df_analysis["season"] = df_analysis["month"].apply(get_season)

    # Thêm biến ngày lễ Việt Nam (chỉ dựa trên tháng/ngày, không phụ thuộc năm)
    def is_vn_holiday(row):
        month = row["month"]
        day = row.name.day  # Lấy ngày từ index datetime
        
        # Danh sách ngày lễ VN (tháng-ngày)
        vn_holidays_list = [
            (1, 1),    # Tết dương lịch
            (1, 30), (2, 1),  # Tết âm lịch (gần đúng, 30 Tết - 2 Tết)
            (4, 30),   # 30/4 - Ngày Giải phóng
            (5, 1),    # 1/5 - Quốc tế lao động
            (9, 2),    # 2/9 - Quốc khánh
        ]
        
        return (month, day) in vn_holidays_list
    
    df_analysis["is_holiday"] = df_analysis.apply(is_vn_holiday, axis=1)
    df_analysis["day_type"] = df_analysis.apply(
        lambda row: "Holiday" if row["is_holiday"] else ("Weekday" if row["dayofweek"] < 5 else "Weekend"),
        axis=1
    )
    
    # ===== Tiêu thụ theo mùa x tháng =====
    season_month = df_analysis.groupby(["season", "month"])[TARGET_COL].mean().reset_index()
    season_month = season_month.sort_values("month")
    season_month["label"] = season_month.apply(
        lambda row: f"{row['season']}\n(Tháng {int(row['month'])})", axis=1
    )
    
    fig_sm, ax_sm = plt.subplots(figsize=(12, 4))
    colors = {"Spring": "green", "Summer": "red", "Autumn": "orange", "Winter": "blue"}
    bar_colors = [colors[s] for s in season_month["season"]]
    ax_sm.bar(range(len(season_month)), season_month[TARGET_COL].values, color=bar_colors, alpha=0.7)
    ax_sm.set_xticks(range(len(season_month)))
    ax_sm.set_xticklabels(season_month["label"], fontsize=9)
    ax_sm.set_ylabel("MW")
    ax_sm.set_title("Analysis of Electricity Consumption by Season and Month")
    ax_sm.grid(axis="y", alpha=0.3)
    st.pyplot(fig_sm, use_container_width=True)
    
    st.caption("""
    **Insight:** The grouped chart of seasons and months helps to see the clear trend of electricity consumption in each season. 
    Each column represents a month, colored according to the season (Spring-green, Summer-red, Autumn-orange, Winter-blue).
    """)

    # ===== Tiêu thụ theo ngày lễ x tháng =====
    st.markdown("### Analysis of Electricity Consumption by Holiday and Month")
    
    holiday_month = df_analysis.groupby(["is_holiday", "month"])[TARGET_COL].mean().reset_index()
    holiday_month["day_type_label"] = holiday_month["is_holiday"].map({True: "Holiday", False: "Weekday"})
    holiday_month = holiday_month.sort_values("month")
    
    # Pivot để so sánh side-by-side
    pivot_hm = holiday_month.pivot_table(
        values=TARGET_COL,
        index="month",
        columns="day_type_label",
        aggfunc="mean"
    )
    
    fig_hm, ax_hm = plt.subplots(figsize=(12, 4))
    x = range(len(pivot_hm))
    width = 0.35
    
    ax_hm.bar([i - width/2 for i in x], pivot_hm["Weekday"].values, width, label="Weekday", color="steelblue", alpha=0.7)
    ax_hm.bar([i + width/2 for i in x], pivot_hm["Holiday"].values, width, label="Holiday", color="coral", alpha=0.7)
    
    ax_hm.set_xlabel("Month")
    ax_hm.set_ylabel("MW")
    ax_hm.set_title("Comparison of Electricity Consumption: Weekday vs Holiday by Month")
    ax_hm.set_xticks(x)
    ax_hm.set_xticklabels(pivot_hm.index)
    ax_hm.legend()
    ax_hm.grid(axis="y", alpha=0.3)
    st.pyplot(fig_hm, use_container_width=True)
    
    st.caption("""
    **Insight:** The comparison chart of electricity consumption between weekdays (blue) and holidays (coral) by month. 
    Overall, holidays have lower consumption than weekdays due to reduced economic and production activities.
    """)


# ===================== TAB FORECAST (làm chính) =====================
with tab_forecast:
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

    # Default: dự báo từ giờ tiếp theo
    min_date = (df.index.max() + pd.Timedelta(hours=1)).date()

    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
    start_date = fcol1.date_input("Start date", value=min_date)
    end_date = fcol2.date_input("End date", value=min_date)

    # Validate input
    if end_date < start_date:
        st.error("End date must be greater than or equal to Start date.")
        st.stop()

    btn_col1, btn_col2 = st.columns([1, 5])
    run = btn_col1.button("Forecast", type="primary")

    if not run:
        st.caption("Press Forecast to run the forecast.")
        st.stop()

    # ===================== FORECAST =====================
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
        st.error("No forecast results available. Please check the date range or the forecast_by_date() function.")
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
    m1.metric("Average Value", f"{avg_value:,.2f} MW")
    m2.metric("Highest Peak", f"{max_point['yhat']:,.2f} MW", str(max_point["Datetime"]))
    m3.metric("Lowest Point", f"{min_point['yhat']:,.2f} MW", str(min_point["Datetime"]))
    m4.metric("Number of Forecast Points", f"{len(fc):,}")






    st.markdown("### Line Chart of Forecast")
    # Line chart (matplotlib for nicer control)
    fig_line = px.line(
        fc,
        x="Datetime",
        y="yhat",
        title="Predicted Electricity Load Over Time",
        labels={"Datetime": "Thời gian", "yhat": "MW"},
        hover_data={"Datetime": "|%Y-%m-%d %H:%M", "yhat": ":,.2f"}  # Tùy chỉnh format hover
    )
    fig_line.update_layout(
        xaxis_title="Time",
        yaxis_title="MW",
        hovermode="x unified"  # Hover trên toàn bộ x-axis
    )
    st.plotly_chart(fig_line, use_container_width=True)



    # ===================== ANALYSIS =====================
    st.markdown("### Quick Analysis")

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
        ax_h.plot(hourly.index, hourly.values, marker="o")
        ax_h.set_xlabel("Hour")
        ax_h.set_ylabel("MW")
        ax_h.set_title("Average yhat by Hour")
        ax_h.grid(axis="y", linestyle="--", alpha=0.35)
        ax_h.set_xticks(list(range(0, 24, 2)))
        st.pyplot(fig_h, use_container_width=True)




    # ===================== DAILY BAR CHART =====================
    st.markdown("### Daily Average Forecast Comparison")

    fig_bar, ax = plt.subplots(figsize=(14, 5.0))
    x_labels = daily.index.astype(str)
    bars = ax.bar(x_labels, daily.values)

    ax.axhline(avg_value, linestyle="--", linewidth=2, label="Average Value")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Daily Average Forecast Comparison")
    ax.set_xlabel("Date")
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
        

    if bar_label_mode == "Dưới cột":
        # chừa khoảng dưới để không bị cắt chữ
        fig_bar.subplots_adjust(bottom=0.22)

    ax.legend()
    st.pyplot(fig_bar, use_container_width=True)

    # ===================== TABLE + DOWNLOAD =====================
    st.markdown("### Forecast Results")

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