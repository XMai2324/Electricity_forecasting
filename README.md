# Electricity_forecasting
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
│  └─ reference/                     # file phụ trợ: holiday, mapping cột (nếu có)
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
│  ├─ evaluate.py                    # tính MAE, RMSE, MAPE (khi có ground truth)
│  └─ io_utils.py                    # save/load csv, json, model, helper paths
│
├─ app/
│  ├─ api.py                         # FastAPI: upload file + forecast theo ngày
│  └─ dashboard.py                   # Streamlit: upload + chọn ngày + xem bảng/biểu đồ
│
└─ outputs/
   ├─ forecasts/                     # kết quả dự báo (csv) theo từng lần chạy
   ├─ reports/                       # metrics, log tóm tắt (json/csv)
   └─ figures/                       # biểu đồ xuất ra để đưa vào báo cáo
