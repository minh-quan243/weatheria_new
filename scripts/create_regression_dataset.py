import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 📥 1. Load dữ liệu thời tiết
# =======================
weather_df = pd.read_csv(
    "D:/Pycharm/weather-new/data/Raw/weather_all_cities_2001_2024.csv",
    parse_dates=["Datetime"]
)
weather_df = weather_df.dropna(subset=["Latitude", "Longitude"])

# Tạo GeoDataFrame
weather_gdf = gpd.GeoDataFrame(
    weather_df,
    geometry=gpd.points_from_xy(weather_df["Longitude"], weather_df["Latitude"]),
    crs="EPSG:4326"
)

# =======================
# 📥 2. Load dữ liệu bão
# =======================
storm_gdf = gpd.read_file("D:/Pycharm/weather-new/data/Raw/storm_buffers_with_time.gpkg")
storm_gdf = storm_gdf.to_crs("EPSG:4326")

# Mở rộng vùng bão nhẹ (~20km)
# Chuyển sang hệ tọa độ phẳng (đơn vị: mét) – Web Mercator
storm_gdf_proj = storm_gdf.to_crs(epsg=3857)

# Buffer bán kính 20km = 20000m
storm_gdf_proj["geometry"] = storm_gdf_proj.buffer(20000)

# Đổi về lại hệ địa lý
storm_gdf = storm_gdf_proj.to_crs(epsg=4326)


# =======================
# 🌀 3. Gán nhãn max_wind
# =======================
weather_gdf["max_wind"] = weather_df["WindSpeed"]  # mặc định = 0 nếu không ảnh hưởng bão

print("✅ Đang gán nhãn max_wind...")
for idx, storm in tqdm(storm_gdf.iterrows(), total=storm_gdf.shape[0]):
    start_time = pd.to_datetime(storm["start_time"])
    end_time = pd.to_datetime(storm["end_time"])
    geometry = storm.geometry
    storm_max_wind = storm.get("max_wind")

    mask_time = weather_gdf["Datetime"].between(start_time, end_time)
    mask_space = weather_gdf["geometry"].within(geometry)
    affected_idx = weather_gdf[mask_time & mask_space].index

    # Chỉ cập nhật nếu storm_max_wind là số hợp lệ
    if pd.notna(storm_max_wind) and storm_max_wind > 0:
        weather_gdf.loc[affected_idx, "max_wind"] = storm_max_wind

# =======================
# 💾 4. Ghi dữ liệu ra CSV
# =======================
output_path = "D:/Pycharm/weather-new/data/Processed/Regression/storm_regression_dataset.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
weather_gdf.drop(columns=["geometry"]).to_csv(output_path, index=False)
print(f"✅ Đã lưu dữ liệu tại: {output_path}")

# =======================
# 📊 5. BIỂU ĐỒ THỐNG KÊ & HỒI QUY
# =======================
df = pd.read_csv(output_path)
features = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection']

# 1️⃣ Histogram phân bố max_wind
plt.figure(figsize=(6, 4))
sns.histplot(df["max_wind"], bins=50, color="skyblue", kde=True)
plt.title("Phân bố max_wind (gió cực đại)")
plt.xlabel("max_wind")
plt.tight_layout()
plt.show()

# 2️⃣ Boxplot max_wind theo City (nếu muốn xem vùng bị ảnh hưởng mạnh)
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[df["max_wind"] > 0], x="City", y="max_wind", color="skyblue")
plt.title("max_wind theo từng City (chỉ lấy điểm có bão)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3️⃣ Scatter plot: WindSpeed thực tế vs max_wind của bão
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df[df["max_wind"] > 0], x="WindSpeed", y="max_wind", hue="City", alpha=0.5)
plt.title("Tương quan WindSpeed thực tế với max_wind của bão")
plt.tight_layout()
plt.show()

# 4️⃣ Heatmap tương quan
plt.figure(figsize=(8, 6))
corr = df[features + ["max_wind"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng và max_wind")
plt.tight_layout()
plt.show()
