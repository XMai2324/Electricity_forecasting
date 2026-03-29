# Electricity_forecasting

Hệ Thống Phân Tích và Dự Báo Phụ Tải Điện (XGBoost & SHAP)
Hệ thống này được phát triển để dự báo nhu cầu tiêu thụ điện năng dựa trên dữ liệu lịch sử, đồng thời cung cấp khả năng giải thích các quyết định của mô hình AI và phát hiện các điểm dữ liệu bất thường.

🌟 Tính năng chính
Dự báo chuỗi thời gian (Time-series Forecasting): Sử dụng thuật toán XGBoost để dự báo lượng điện tiêu thụ (PJME_MW) trong tương lai dựa trên các đặc trưng thời gian và độ trễ (lags).

Giải thích mô hình (Explainable AI - XAI): Tích hợp thuật toán SHAP (TreeExplainer) để minh bạch hóa lý do tại sao mô hình đưa ra con số dự báo cụ thể, giúp người dùng hiểu rõ tác động của từng biến như nhiệt độ, giờ cao điểm, hay ngày lễ.

Phát hiện bất thường (Anomaly Detection): Sử dụng thuật toán Isolation Forest để cảnh báo các điểm dự báo có dấu hiệu bất thường (quá cao hoặc quá thấp) so với quy luật tiêu thụ thông thường trong quá khứ.

Dashboard tương tác: Giao diện web được xây dựng bằng Streamlit, cho phép người dùng upload dữ liệu, xem biểu đồ phân tích (EDA) và thực hiện dự báo trực quan.



## cau truc thu muc 

electricity_forecast_system/
├─ README.md                         # mô tả dự án + cách chạy
├─ requirements.txt                  # thư viện cần cài
├─ .env.example                      # cấu hình mẫu (đường dẫn, chế độ chạy)
│
├─ notebooks/
│  └─ xgboost.ipynb                  # notebook hiện có (EDA, train, thử nghiệm)
│
├─ data/
│  ├─ sample/                        # file mẫu để người dùng upload thử
│  └─ reference/       (chưa có)              # file phụ trợ: holiday, mapping cột (nếu có)
│
├─ uploads/
│  ├─ raw/                           # file người dùng vừa upload (giữ nguyên)
│  └─ processed/                     # file sau khi hệ thống làm sạch và chuẩn hóa
│
├─ artifacts/
│  ├─ model.pkl                      # mô hình XGBoost đã huấn luyện
│  └─ feature_config.json            # danh sách/thứ tự features, cột target, freq
│
├─ src/
│  ├─ preprocess.py                  # đọc file upload, parse datetime, xử lý thiếu, sort
│  ├─ features.py                    # tạo lag, rolling, time features
│  ├─ forecast.py                    # dự báo theo ngày tùy chọn (start_date, end_date)
│  ├─ evaluate.py  (chưa có)                   # tính MAE, RMSE, MAPE (khi có ground truth)
│  └─ io_utils.py                    # save/load csv, json, model, helper paths
│
├─ app/
│  ├─ api.py      (chưa có)                    # FastAPI: upload file + forecast theo ngày
│  └─ dashboard.py                   # Streamlit: upload + chọn ngày + xem bảng/biểu đồ
│
└─ outputs/
   ├─ forecasts/                     # kết quả dự báo (csv) theo từng lần chạy
   ├─ reports/    (chưa có)                   # metrics, log tóm tắt (json/csv)
   └─ figures/    (chưa có)                    # biểu đồ xuất ra để đưa vào báo cáo






## Luồng hệ thống
user upload CSV
        ↓
preprocess.py
        ↓
features.py
        ↓
model.pkl (XGBoost)
        ↓
forecast.py
        ↓
dashboard hiển thị dự báo



## Luồng hoạt động của hệ thống
Hệ thống dự báo nhu cầu điện sử dụng mô hình XGBoost đã được huấn luyện trước.  
Quy trình hoạt động của hệ thống như sau:

1. Người dùng tải lên file CSV chứa dữ liệu tiêu thụ điện.
2. `preprocess.py` đọc file dữ liệu, làm sạch dữ liệu, chuyển đổi định dạng thời gian và xử lý các giá trị thiếu.
3. `features.py` tạo các đặc trưng cần thiết cho mô hình như đặc trưng thời gian và các biến trễ (lag features).
4. Hệ thống nạp mô hình đã huấn luyện `model.pkl` để thực hiện dự báo.
5. `forecast.py` xử lý dữ liệu đầu vào, áp dụng mô hình và tạo kết quả dự báo theo khoảng thời gian người dùng lựa chọn.
6. Dashboard hiển thị kết quả dự báo dưới dạng bảng dữ liệu và biểu đồ trực quan.
