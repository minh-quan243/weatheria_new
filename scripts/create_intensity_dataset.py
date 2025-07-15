import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 📥 1. Load dữ liệu thời tiết
# ================================
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

# ================================
# 📥 2. Load vùng ảnh hưởng bão
# ================================
storm_gdf = gpd.read_file("D:/Pycharm/weather-new/data/Raw/storm_buffers_with_time.gpkg")
storm_gdf = storm_gdf.to_crs("EPSG:4326")

# ================================
# 📊 3. Hàm phân loại theo cấp gió
# ================================
def classify_wind(wind_speed):
    if pd.isna(wind_speed):
        return 1
    if wind_speed < 17:
        return 1
    elif 17 <= wind_speed < 34:
        return 2
    elif 34 <= wind_speed < 64:
        return 3
    else:
        return 4

# ================================
# 🔍 4. Gán cấp độ bão theo khoảng cách
# ================================
def get_storm_category_by_distance(point, storm_geom, max_wind):
    try:
        dist = point.distance(storm_geom)
    except:
        return 0

    if storm_geom.contains(point):
        return classify_wind(max_wind)
    elif dist < 0.2:
        return 3
    elif dist < 0.5:
        return 2
    else:
        return 0

# Gán nhãn mặc định
weather_gdf["storm_category"] = 0

# ================================
# 🌀 5. Gán nhãn từng điểm thời tiết
# ================================
print("✅ Đang gán nhãn storm_category theo khoảng cách...")
for idx, storm in tqdm(storm_gdf.iterrows(), total=storm_gdf.shape[0]):
    start_time = pd.to_datetime(storm["start_time"])
    end_time = pd.to_datetime(storm["end_time"])
    storm_geom = storm.geometry
    max_wind = storm.get("max_wind", None)

    condition_time = weather_gdf["Datetime"].between(start_time, end_time)
    sub_df = weather_gdf[condition_time]

    categories = sub_df["geometry"].apply(
        lambda pt: get_storm_category_by_distance(pt, storm_geom, max_wind)
    )

    weather_gdf.loc[sub_df.index, "storm_category"] = categories.combine(
        weather_gdf.loc[sub_df.index, "storm_category"], max
    )

# ================================
# 📆 6. Thêm cột tháng và mùa
# ================================
weather_gdf["Month"] = weather_gdf["Datetime"].dt.month

def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

weather_gdf["Season"] = weather_gdf["Month"].apply(assign_season)

# ================================
# 💾 7. Ghi dữ liệu ra CSV
# ================================
out_path = "D:/Pycharm/weather-new/data/Processed/Intensity/storm_intensity_dataset.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
if os.path.exists(out_path):
    os.remove(out_path)

weather_gdf.drop(columns=["geometry"]).to_csv(out_path, index=False)
print(f"✅ Đã lưu dữ liệu tại: {out_path}")

# ================================
# 📊 8. VẼ BIỂU ĐỒ THỐNG KÊ
# ================================
df = pd.read_csv(out_path)
features = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection', 'Month']

# 1️⃣ Countplot theo storm_category
plt.figure(figsize=(6, 4))
sns.countplot(x="storm_category", hue="storm_category", data=df, palette="Set2", legend=False)
plt.title("Phân bố điểm theo cấp độ bão (storm_category)")
plt.xlabel("Cấp độ bão")
plt.ylabel("Số lượng điểm")
plt.tight_layout()
plt.show()

# 2️⃣ Histogram đặc trưng
df[features].hist(bins=50, figsize=(14, 10), color="lightblue", edgecolor="black")
plt.suptitle("Phân bố các đặc trưng thời tiết")
plt.tight_layout()
plt.show()

# 3️⃣ Heatmap tương quan
plt.figure(figsize=(8, 6))
corr = df[features + ["storm_category"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng và storm_category")
plt.tight_layout()
plt.show()
