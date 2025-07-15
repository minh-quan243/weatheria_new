import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

# 📌 Đọc file CSV chứa tỉnh và tọa độ
province_df = pd.read_csv("D:/Pycharm/weather-new/data/Raw/japan_lot_lat.csv")

# Chuyển dữ liệu tỉnh và tọa độ thành dictionary
provinces_coords = dict(zip(province_df['Province'], zip(province_df['lat'], province_df['lon'])))

# 📌 Danh sách cột đặc trưng dùng chung cho tất cả model
FEATURE_COLUMNS = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection', 'Month',
                   'Season']

# Load models
model_onset = joblib.load('models/model_use/onset_model_xgboost.pkl')
model_intensity = joblib.load('models/model_use/storm_intensity_xgboost.pkl')
model_regression = joblib.load('models/model_use/regression_xgboost_maxwind.pkl')
model_duration = joblib.load('models/model_use/duration_rf_model.pkl')

# Load 3 new models
model_1 = joblib.load('models/model_use/istyphoon_model.pkl')  # Model 1
model_2 = joblib.load('models/model_use/numtyphoon_model.pkl')  # Model 2
model_3 = joblib.load('models/model_use/region_model.pkl')  # Model 3

# Load scalers
scaler_intensity = joblib.load('models/model_use/scaler_intensity.pkl')
scaler_regression = joblib.load('models/model_use/regression_scaler.pkl')
scaler_duration = joblib.load('models/model_use/duration_scaler.pkl')

# Load SHAP explainers
explainer_onset = shap.Explainer(model_onset)
explainer_intensity = shap.Explainer(model_intensity)
explainer_regression = shap.Explainer(model_regression)
explainer_duration = shap.Explainer(model_duration)
explainer_model_1 = shap.Explainer(model_1)
explainer_model_2 = shap.Explainer(model_2)
explainer_model_3 = shap.Explainer(model_3.estimators_[0])


# Chuẩn hóa input cho 4 mô hình cũ
def prepare_features_old_models(rain, temp, wind_speed, pressure, humidity, cloud_cover, wind_direction, month, season):
    # Tạo input cho các mô hình dựa trên năm, tháng và mùa
    season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
    season_num = season_mapping.get(season, -1)  # Mã hóa mùa

    df = pd.DataFrame([{
        'Rain': rain,
        'Temp': temp,
        'WindSpeed': wind_speed,
        'Pressure': pressure,
        'Humidity': humidity,
        'CloudCover': cloud_cover,
        'WindDirection': wind_direction,
        'Month': month,
        'Season': season_num
    }])

    return df[FEATURE_COLUMNS]


# Chuẩn hóa input cho 3 mô hình mới (chỉ có năm, tháng và mùa)
def prepare_features_new_models(year, month, season):
    # Tạo input cho các mô hình mới chỉ dựa trên năm, tháng và mùa
    season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
    season_num = season_mapping.get(season, -1)  # Mã hóa mùa

    df = pd.DataFrame([{
        'year': year,
        'month': month,
        'Season': season_num
    }])

    # One-Hot Encoding cho 'Month' và 'Season'
    df = pd.get_dummies(df, columns=['month', 'Season'], drop_first=True)  # drop_first để tránh dummy variable trap

    # Đảm bảo các cột đúng thứ tự giống như khi huấn luyện mô hình
    expected_columns = ['year', 'month', 'season_Spring', 'season_Summer', 'season_Winter']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Thêm cột thiếu vào nếu chưa có (set giá trị là 0)

    df = df[expected_columns]  # Đảm bảo thứ tự cột

    return df


# Vẽ SHAP waterfall plot
def show_shap_plot(explainer, features, model_name):
    shap_values = explainer(features)

    st.subheader(f"🔍 SHAP - {model_name}")
    fig, ax = plt.subplots()

    try:
        if len(shap_values.values.shape) == 3:
            sv = shap.Explanation(
                values=shap_values.values[0, 1],
                base_values=shap_values.base_values[0, 1],
                data=shap_values.data[0],
                feature_names=shap_values.feature_names
            )
        else:
            sv = shap_values[0]
        shap.plots.waterfall(sv, max_display=6, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Không thể vẽ biểu đồ SHAP: {e}")


# Dự đoán cho 4 mô hình cũ
def predict_for_old_models():
    # Lấy năm, tháng và mùa theo thời gian thực
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Logic lấy tháng hiện tại hoặc tháng sau
    if current_date.day <= 15:
        month = current_month
    else:
        month = current_month % 12 + 1

    season = 'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Autumn'

    # Tạo input cho các mô hình dựa trên năm, tháng và mùa
    st.write(f"📅 Dự đoán cho tháng {month} - Mùa {season}")

    # Tạo dữ liệu chỉ với năm, tháng và mùa
    input_data = prepare_features_old_models(0, 0, 0, 0, 0, 0, 0, month, season)

    # Dự đoán trên 4 mô hình cũ
    onset_input = input_data
    storm = model_onset.predict(onset_input)[0]
    st.write("🌀 Phát hiện bão:", "🚨 Có thể có bão" if storm == 1 else "✅ Không có dấu hiệu bão")

    if storm == 1:
        input_i = input_data
        X_i = scaler_intensity.transform(input_i)
        intensity = model_intensity.predict(X_i)[0]
        st.write(f"🌪️ Cường độ: `{intensity}`")

        input_r = input_data
        X_r = scaler_regression.transform(input_r)
        max_wind = model_regression.predict(X_r)[0]
        st.write(f"💨 Gió mạnh nhất: `{max_wind:.2f}` km/h")

        input_d = input_data
        X_d = scaler_duration.transform(input_d)
        duration = model_duration.predict(X_d)[0]
        st.write(f"⏳ Thời gian ảnh hưởng: `{duration:.1f}` giờ")
    else:
        st.write("Thời tiết bình thường")


# Dự đoán cho 3 mô hình mới
def predict_for_new_models():
    # Lấy năm, tháng và mùa theo thời gian thực
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Logic lấy tháng hiện tại hoặc tháng sau
    if current_date.day <= 15:
        month = current_month
    else:
        month = current_month % 12 + 1

    season = 'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7,
                                                                                                            8] else 'Autumn'

    # Tạo input cho các mô hình mới chỉ dựa trên năm, tháng và mùa
    st.write(f"📅 Dự đoán cho tháng {month} - Mùa {season}")

    # Tạo dữ liệu chỉ với năm, tháng và mùa cho các mô hình mới
    input_data = prepare_features_new_models(current_year, month, season)

    # Dự đoán trên 3 mô hình mới
    input_m1 = input_data
    prediction_model_1 = model_1.predict(input_m1)[0]
    st.write(f"🔮 Dự đoán từ Model 1: `{prediction_model_1}`")

    if prediction_model_1 == 1:
        input_m2 = input_data
        prediction_model_2 = model_2.predict(input_m2)[0]
        st.write(f"🔮 Dự đoán từ Model 2: `{prediction_model_2}`")
        show_shap_plot(explainer_model_2, input_m2, "Model 2")

        input_m3 = input_data
        prediction_model_3 = model_3.predict(input_m3)[0]

        # Mã hóa ngược lại kết quả dự đoán từ Model 3 (ví dụ: 0 -> North, 1 -> South, ...)
        region_mapping = {0: 'North', 1: 'South', 2: 'East', 3: 'West', 4: 'Unknown', 5: 'Central'}
        predicted_region = region_mapping.get(prediction_model_3, 'Unknown')

        st.write(f"🔮 Dự đoán từ Model 3: `{predicted_region}`")  # In tên vùng bão thay vì số
        show_shap_plot(explainer_model_3, input_m3, "Model 3")
    else:
        st.write("⚠️ Model 1 dự đoán không có bão, không chạy các mô hình tiếp theo.")

    show_shap_plot(explainer_model_1, input_m1, "Model 1")

# Giao diện Streamlit
st.set_page_config(layout="wide")
st.title("🌀 Dự báo Bão Hiện Tại & SHAP Giải thích")

# Dự đoán dựa trên năm, tháng và mùa cho các mô hình mới (tự động chạy ngay khi trang được tải)
predict_for_new_models()

# Dự đoán từ tỉnh cho 4 mô hình cũ (chỉ chạy khi nhấn nút)
selected_province = st.selectbox("📍 Chọn tỉnh tại Nhật Bản", options=list(provinces_coords.keys()))
if st.button("▶️ Dự đoán cho tương lai gần"):
    st.write("🔄 Đang dự đoán...")
    predict_for_old_models()
