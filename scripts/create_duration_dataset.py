import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ 1. Load dữ liệu thời tiết
weather_df = pd.read_csv("D:/Pycharm/weather-new/data/Raw/weather_all_cities_2001_2024.csv", parse_dates=["Datetime"])
weather_df = weather_df.dropna(subset=["Latitude", "Longitude"])

# ✅ 2. Tạo GeoDataFrame
weather_gdf = gpd.GeoDataFrame(
    weather_df,
    geometry=gpd.points_from_xy(weather_df["Longitude"], weather_df["Latitude"]),
    crs="EPSG:4326"
)

# ✅ 3. Load vùng ảnh hưởng bão
storm_gdf = gpd.read_file("D:/Pycharm/weather-new/data/Raw/storm_buffers_with_time.gpkg")
storm_gdf = storm_gdf.to_crs("EPSG:4326")

# ✅ 4. Gắn nhãn in_storm = 1 nếu điểm thời tiết nằm trong vùng bão & đúng thời gian
weather_gdf["in_storm"] = 0
weather_gdf["storm_id"] = None  # Nếu muốn truy vết ID bão

print("🌀 Gán nhãn điểm bị ảnh hưởng bão...")

for idx, storm in tqdm(storm_gdf.iterrows(), total=storm_gdf.shape[0]):
    start_time = pd.to_datetime(storm["start_time"])
    end_time = pd.to_datetime(storm["end_time"])
    geometry = storm.geometry
    storm_id = storm.get("SID", f"storm_{idx}")

    condition_time = weather_gdf["Datetime"].between(start_time, end_time)
    condition_space = weather_gdf["geometry"].intersects(geometry)  # CHỈNH TẠI ĐÂY

    affected = condition_time & condition_space
    weather_gdf.loc[affected, "in_storm"] = 1
    weather_gdf.loc[affected, "storm_id"] = storm_id

# ✅ 5. Tính duration: số giờ liên tục trong vùng bão
print("📦 Tính duration...")
rows = []
for city, group in weather_gdf.groupby("City"):
    group = group.sort_values("Datetime").reset_index(drop=True)
    in_storm = group["in_storm"].values
    start_idx = None

    for i in range(len(in_storm)):
        if in_storm[i] == 1 and (i == 0 or in_storm[i - 1] == 0):
            start_idx = i
        elif in_storm[i] == 0 and start_idx is not None:
            duration = i - start_idx
            if duration >= 1:
                start_row = group.iloc[start_idx].copy()
                start_row["duration_in_storm"] = duration
                rows.append(start_row)
            start_idx = None

    # ✅ Trường hợp bão kéo dài đến dòng cuối
    if start_idx is not None:
        duration = len(in_storm) - start_idx
        if duration >= 1:
            start_row = group.iloc[start_idx].copy()
            start_row["duration_in_storm"] = duration
            rows.append(start_row)

# ✅ 6. Tạo DataFrame mới với nhãn
duration_df = pd.DataFrame(rows)

# ✅ 7. Chỉ giữ lại các cột cần thiết
cols_to_keep = [
    'Datetime', 'City', 'Latitude', 'Longitude',
    'Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity',
    'CloudCover', 'WindDirection', 'duration_in_storm', 'storm_id'
]
duration_df = duration_df[cols_to_keep]

# ✅ 8. Lưu ra file
out_path = "D:/Pycharm/weather-new/data/Processed/Duration/storm_duration_dataset.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
duration_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"✅ Đã lưu dữ liệu duration tại: {out_path}")

# 📈 Phân phối duration_in_storm
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
sns.histplot(duration_df["duration_in_storm"], bins=20, kde=True, color='skyblue')
plt.title("Phân phối số giờ bị ảnh hưởng bởi bão")
plt.xlabel("Số giờ ảnh hưởng")
plt.ylabel("Số lượng mẫu")
plt.tight_layout()
plt.show()

# 🔍 Ma trận tương quan các biến
plt.figure(figsize=(10, 8))
corr_matrix = duration_df.drop(columns=["Datetime", "City", "storm_id"]).corr()
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
plt.title("Ma trận tương quan giữa các biến đầu vào và nhãn")
plt.tight_layout()
plt.show()
