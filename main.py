import streamlit as st
import requests
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 📌 Danh sách cột đặc trưng dùng chung cho tất cả model
FEATURE_COLUMNS = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection']

# Load models
model_onset = joblib.load('models/model_use/onset_model_random_forest.pkl')
model_intensity = joblib.load('models/model_use/storm_intensity_xgboost.pkl')
model_regression = joblib.load('models/model_use/regression_xgboost_maxwind.pkl')
model_duration = joblib.load('models/model_use/duration_rf_model.pkl')

# Load scalers
scaler_intensity = joblib.load('models/model_use/scaler_intensity.pkl')
scaler_regression = joblib.load('models/model_use/regression_scaler.pkl')
scaler_duration = joblib.load('models/model_use/duration_scaler.pkl')

# Load SHAP explainers
explainer_onset = shap.Explainer(model_onset)
explainer_intensity = shap.Explainer(model_intensity)
explainer_regression = shap.Explainer(model_regression)
explainer_duration = shap.Explainer(model_duration)

# Lấy dữ liệu thời tiết hiện tại từ Open-Meteo
def get_weather_data_current(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=cloudcover,temperature_2m,surface_pressure,rain,"
        f"wind_speed_10m,wind_direction_10m,relative_humidity_2m"
        f"&forecast_days=1&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    return df.iloc[-1]  # Lấy thời điểm mới nhất

# Chuẩn hóa input theo đúng danh sách cột cố định
def prepare_features(row):
    df = pd.DataFrame([{
        'Rain': row['rain'],
        'Temp': row['temperature_2m'],
        'WindSpeed': row['wind_speed_10m'],
        'Pressure': row['surface_pressure'],
        'Humidity': row['relative_humidity_2m'],
        'CloudCover': row['cloudcover'],
        'WindDirection': row['wind_direction_10m']
    }])
    return df[FEATURE_COLUMNS]

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

# Hàm chính
def predict(lat, lon):
    row = get_weather_data_current(lat, lon)
    st.write("⏰ Thời điểm dữ liệu:", row['time'])

    # Onset
    onset_input = prepare_features(row)
    storm = model_onset.predict(onset_input)[0]
    st.write("🌀 Phát hiện bão:", "🚨 Có thể có bão" if storm == 1 else "✅ Không có dấu hiệu bão")
    show_shap_plot(explainer_onset, onset_input, "Phát hiện bão")

    if storm == 1:
        # INTENSITY
        input_i = prepare_features(row)
        X_i = scaler_intensity.transform(input_i)
        intensity = model_intensity.predict(X_i)[0]
        st.write(f"🌪️ Cường độ: `{intensity}`")
        show_shap_plot(explainer_intensity, X_i, "Cường độ")

        # MAX WIND
        input_r = prepare_features(row)
        X_r = scaler_regression.transform(input_r)
        max_wind = model_regression.predict(X_r)[0]
        st.write(f"💨 Gió mạnh nhất: `{max_wind:.2f}` km/h")
        show_shap_plot(explainer_regression, X_r, "Gió mạnh nhất")
        if intensity == 0 and max_wind >= 100:
            st.warning("⚠️ Dự đoán cường độ thấp, nhưng gió rất mạnh. Có thể mô hình chưa khớp tốt!")

        # DURATION
        input_d = prepare_features(row)
        X_d = scaler_duration.transform(input_d)
        duration = model_duration.predict(X_d)[0]
        st.write(f"⏳ Thời gian ảnh hưởng: `{duration:.1f}` giờ")
        show_shap_plot(explainer_duration, X_d, "Thời gian ảnh hưởng")
    else: st.write("Thời tiết bình thươờng")

# Giao diện Streamlit
st.set_page_config(layout="wide")
st.title("🌀 Dự báo Bão Hiện Tại & SHAP Giải thích")

lat = st.number_input("📍 Nhập vĩ độ (latitude)", value=33.59, format="%.4f")
lon = st.number_input("📍 Nhập kinh độ (longitude)", value=130.401, format="%.4f")

if st.button("▶️ Dự đoán thời điểm hiện tại"):
    st.write("🔄 Đang lấy dữ liệu thời tiết và dự đoán...")
    predict(lat, lon)
