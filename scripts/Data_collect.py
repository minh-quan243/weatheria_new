import pandas as pd
import requests
import time
import os

# 📥 Định nghĩa thông tin của 4 tỉnh có bão nhiều nhất ở Nhật Bản
cities_info = [
    {"City": "Okinawa", "Latitude": 26.512, "Longitude": 127.933},
    {"City": "Kagoshima", "Latitude": 31.596, "Longitude": 130.558},
    {"City": "Kochi", "Latitude": 33.558, "Longitude": 133.531},
    {"City": "Fukuoka", "Latitude": 33.590, "Longitude": 130.401},
]

# ⚙️ Các yếu tố cần thiết cho dự báo bão (bao gồm Hướng gió và Cloudcover)
hourly_vars = [
    "precipitation",  # Lượng mưa
    "temperature_2m",  # Nhiệt độ tại 2m
    "windspeed_10m",  # Tốc độ gió tại 10m
    "surface_pressure",  # Áp suất bề mặt
    "relative_humidity_2m",  # Độ ẩm tại 2m
    "cloudcover",  # Độ che phủ mây
    "winddirection_10m"  # Hướng gió tại 10m
]
hourly_str = ",".join(hourly_vars)


def fetch_weather(city, base_lat, base_lon):
    dfs = []

    for year in range(2001, 2025):
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={base_lat}&longitude={base_lon}"
            f"&start_date={year}-01-01&end_date={year}-12-31"
            f"&hourly={hourly_str}&timezone=auto"
        )

        success = False
        retry_count = 0
        while not success:
            try:
                res = requests.get(url, timeout=60)
                if res.status_code == 429:
                    retry_count += 1
                    if retry_count > 10:
                        print(f"⚠️ Bỏ qua {city} sau 10 lần bị chặn.")
                        break
                    print(f"[WAIT] {city} - {year}: 429 Too Many Requests → đợi 60s")
                    time.sleep(60)
                    continue
                res.raise_for_status()
                hourly = res.json()["hourly"]
                df = pd.DataFrame({
                    'City': city,
                    'Latitude': base_lat,
                    'Longitude': base_lon,
                    'Datetime': hourly['time'],
                    'Rain': hourly['precipitation'],
                    'Temp': hourly['temperature_2m'],
                    'WindSpeed': hourly['windspeed_10m'],
                    'Pressure': hourly['surface_pressure'],
                    "Humidity": hourly["relative_humidity_2m"],
                    "CloudCover": hourly["cloudcover"],
                    "WindDirection": hourly["winddirection_10m"],
                })
                dfs.append(df)
                success = True
                time.sleep(2)
            except Exception as e:
                print(f"[RETRY] {city} - {year}: {e}")
                time.sleep(10)

    return dfs


# 📍 Lặp qua 4 tỉnh và lấy dữ liệu
all_dfs = []

for city_info in cities_info:
    city = city_info["City"]
    lat = city_info["Latitude"]
    lon = city_info["Longitude"]
    city_dfs = fetch_weather(city, lat, lon)
    all_dfs.extend(city_dfs)

# 📦 Gộp tất cả dữ liệu và lưu vào 1 tệp CSV
if all_dfs:
    result = pd.concat(all_dfs)

    # Chuyển sang long format
    result_long = result.melt(
        id_vars=['City', 'Latitude', 'Longitude', 'Datetime'],
        var_name='Variable',
        value_name='Value'
    )

    # Pivot lại nhưng giữ Latitude, Longitude
    meta_cols = result[['City', 'Datetime', 'Latitude', 'Longitude']].drop_duplicates()
    pivot = result_long.pivot_table(
        index=['City', 'Datetime'],
        columns='Variable',
        values='Value'
    ).reset_index()
    result_wide = pivot.merge(meta_cols, on=['City', 'Datetime'], how='left')

    # Thêm cột tháng và mùa
    result_wide["Datetime"] = pd.to_datetime(result_wide["Datetime"])
    result_wide["Month"] = result_wide["Datetime"].dt.month

    # Hàm gán mùa theo tháng
    def assign_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"

    result_wide["Season"] = result_wide["Month"].apply(assign_season)

    # Đảm bảo thư mục lưu trữ tồn tại
    save_path = 'D:\\Pycharm\\weather-new\\data\\Raw'
    os.makedirs(save_path, exist_ok=True)

    # Lưu kết quả vào tệp CSV duy nhất
    filename = os.path.join(save_path, "weather_all_cities_2001_2024.csv")
    result_wide.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✅ Đã lưu xong file: {filename}")
