import os
import pandas as pd

from preprocess import preprocess_csv
from forecast import forecast_by_date


if __name__ == "__main__":
    # chỉnh lại đúng đường dẫn file csv mẫu
    csv_path = os.path.join("data", "sample", "PJME_hourly.csv")

    TIME_COL = "Datetime"
    TARGET_COL = "PJME_MW"

    df = preprocess_csv(csv_path, TIME_COL, TARGET_COL)

    # chọn khoảng ngày muốn dự báo
    start_date = "2018-01-01"
    end_date = "2018-01-07"

    fc = forecast_by_date(
        df=df,
        time_col=TIME_COL,
        target_col=TARGET_COL,
        start_date=start_date,
        end_date=end_date,
        model_path=os.path.join("artifacts", "model.pkl"),
        feature_config_path=os.path.join("artifacts", "feature_config.json"),
    )

    print(fc.head())
    os.makedirs(os.path.join("outputs", "forecasts"), exist_ok=True)
    out_path = os.path.join("outputs", "forecasts", "forecast_result.csv")
    fc.to_csv(out_path, index=False)
    print("Saved:", out_path)