import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from tqdm import tqdm

# ================================
# 📥 Load dữ liệu thời tiết
# ================================
weather_df = pd.read_csv("D:/Pycharm/weather-new/data/Raw/weather_all_cities_2001_2024.csv", parse_dates=["Datetime"])
weather_df = weather_df.dropna(subset=["Latitude", "Longitude"])

# Tạo geometry cho thời tiết
weather_gdf = gpd.GeoDataFrame(
    weather_df,
    geometry=gpd.points_from_xy(weather_df["Longitude"], weather_df["Latitude"]),
    crs="EPSG:4326"
)

# ================================
# 📥 Load dữ liệu buffer bão
# ================================
storm_gdf = gpd.read_file("D:/Pycharm/weather-new/data/Raw/storm_buffers_with_time.gpkg")
storm_gdf = storm_gdf.to_crs("EPSG:4326")

# ================================
# 🌀 Gán nhãn bão
# ================================
weather_gdf["storm_onset"] = 0

print("✅ Đang gán nhãn...")
for idx, storm in tqdm(storm_gdf.iterrows(), total=storm_gdf.shape[0]):
    start_time = pd.to_datetime(storm["start_time"])
    end_time = pd.to_datetime(storm["end_time"])
    geometry = storm.geometry

    condition_time = weather_gdf["Datetime"].between(start_time, end_time)
    condition_space = weather_gdf["geometry"].within(geometry)

    weather_gdf.loc[condition_time & condition_space, "storm_onset"] = 1

# ================================
# 📆 Thêm cột tháng và mùa
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
# 💾 Lưu file CSV
# ================================
processed_df = weather_gdf.drop(columns=["geometry"])
save_path = "D:/Pycharm/weather-new/data/Processed/Onset/storm_onset_dataset.csv"
processed_df.to_csv(save_path, index=False)
print(f"✅ Đã lưu tập dữ liệu huấn luyện tại: {save_path}")

# ================================
# 📊 VẼ CÁC BIỂU ĐỒ THỐNG KÊ DỮ LIỆU
# ================================
features = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection']

# 1️⃣ Biểu đồ phân bố nhãn
plt.figure(figsize=(5, 3))
sns.countplot(x="storm_onset", hue="storm_onset", data=processed_df, palette="Set2", legend=False)
plt.title("Phân bố nhãn storm_onset (0 = Không bão, 1 = Có bão)")
plt.xlabel("storm_onset")
plt.ylabel("Số lượng")
plt.tight_layout()
plt.show()

# 2️⃣ Histogram các đặc trưng
processed_df[features].hist(bins=50, figsize=(14, 10), color="skyblue", edgecolor="black")
plt.suptitle("Phân bố các đặc trưng thời tiết")
plt.tight_layout()
plt.show()

# 3️⃣ Ma trận tương quan
plt.figure(figsize=(8, 6))
corr = processed_df[features + ['storm_onset']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng và storm_onset")
plt.tight_layout()
plt.show()

# 4️⃣ Boxplot đặc trưng theo storm_onset
plt.figure(figsize=(12, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=processed_df, x="storm_onset", y=feature, hue="storm_onset", palette="Set3", legend=False)
    plt.title(f"{feature} theo storm_onset")
plt.tight_layout()
plt.show()
