import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import os

def get_buffer_km_from_wind(wind):
    """Xác định bán kính ảnh hưởng (km) dựa vào sức gió của cơn bão."""
    if wind < 63:
        return 50   # Áp thấp nhiệt đới
    elif wind < 118:
        return 100  # Bão nhiệt đới
    elif wind < 154:
        return 150  # Bão mạnh
    else:
        return 200  # Siêu bão

def generate_buffers_and_add_time(input_csv, output_gpkg):
    """Tạo buffer cho các cơn bão và thêm thông tin thời gian vào buffer."""
    df = pd.read_csv(input_csv, parse_dates=["time"])
    df = df.rename(columns={"id": "SID"})  # dùng cột ID làm mã bão
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "wind"])

    # Tạo GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    buffers = []

    # Tạo buffer (vùng ảnh hưởng) cho từng cơn bão
    for sid, storm_points in gdf.groupby("SID"):
        if len(storm_points) < 2:
            continue
        max_wind = storm_points["wind"].max()
        buffer_km = get_buffer_km_from_wind(max_wind)
        buffer_deg = buffer_km / 111.0  # Chuyển km thành độ

        track = LineString(storm_points.geometry.tolist())
        buffer = track.buffer(buffer_deg)

        buffers.append({
            "SID": sid,
            "max_wind": max_wind,
            "buffer_km": buffer_km,
            "geometry": buffer
        })

    # Tạo GeoDataFrame cho các vùng ảnh hưởng
    buffer_gdf = gpd.GeoDataFrame(buffers, crs="EPSG:4326")

    # Thêm thời gian bắt đầu và kết thúc cho mỗi cơn bão
    time_df = df.groupby("SID")["time"].agg(["min", "max"]).reset_index()
    time_df.columns = ["SID", "start_time", "end_time"]
    buffer_gdf = buffer_gdf.merge(time_df, on="SID", how="left")

    # Lưu kết quả vào GeoPackage
    os.makedirs(os.path.dirname(output_gpkg), exist_ok=True)
    buffer_gdf.to_file(output_gpkg, driver="GPKG")
    print(f"✅ Đã tạo buffer và thêm thời gian vào: {output_gpkg}")

if __name__ == "__main__":
    generate_buffers_and_add_time(
        input_csv="D:/Pycharm/weather-new/data/Raw/ibtracs_japan.csv",
        output_gpkg="D:/Pycharm/weather-new/data/Raw/storm_buffers_with_time.gpkg"
    )
